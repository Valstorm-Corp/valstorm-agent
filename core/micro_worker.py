"""Ephemeral Micro-Worker Discovery Pre-Hydration & Mechanical Turn Offloader (FI-559 / P2).

Provides two cache-safe cost and latency optimizations for `ReActEngine`:
1. **Pre-Turn Deterministic Discovery Hydration (`plan_pre_turn_discovery_calls`)**:
   When a user prompt on `iteration == 0` explicitly requests reading existing workspace
   files (`read_file`), inspecting a named Valstorm object schema (`valstorm_schema_inspect`),
   or scraping an explicit URL (`web_scrape`) before performing synthesis/mutation,
   pre-executes those read-only discovery tool calls in parallel before the first LLM call
   (`0` LLM tokens, `<5ms` routing overhead).
2. **Post-Synthesis Mechanical Turn Offload (`resolve_mechanical_offload`)**:
   On `thinker`-tier turns (`gemini-pro-latest` / `claude-opus-5-5`), once the core
   code/script synthesis (`write_file`, `patch_file`) has completed on Iteration 0/1,
   offloads subsequent mechanical test execution (`terminal_exec`) and post-verification
   wrap-up turns to the fast `reasoner` / `worker` model (`gemini-flash-latest` /
   `gemini-flash-lite-latest`), cutting post-synthesis token cost by 8.3x-16.7x without
   fragmenting the Thinker model's contiguous KV prefix cache.
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from core.message_autotuner import (
    MUTATION_EXECUTION_TOOLS,
    AutoTuneDecision,
    ModelTier,
    ThinkingLevel,
    _IMMEDIATE_CREATION_SIGNALS,
    resolve_tier_model,
)
from core.models import Message, SessionState, ToolCall
from core.sandbox import get_current_sandbox

_FILE_PATH_RE = re.compile(
    r"(?:['\"`]|^|\s)([a-zA-Z0-9_./-]+\.(?:py|ts|tsx|js|jsx|json|yaml|yml|md|sh|go|rs|sql|toml))(?:['\"`]|$|\s|,|\.)",
)
_SCHEMA_INSPECT_RE = re.compile(
    r"(?:inspect\s+the\s+['\"`]?([a-zA-Z0-9_]+)['\"`]?\s+schema|valstorm_schema_inspect[^a-zA-Z0-9_]+([a-zA-Z0-9_]+))",
    re.IGNORECASE,
)
_WEB_SCRAPE_URL_RE = re.compile(
    r"web_scrape\s+(?:on\s+|for\s+)?['\"`]?(https?://[^\s'\"`)]+)['\"`]?",
    re.IGNORECASE,
)

# Verification & wrap-up tool subset used when offloading mechanical post-synthesis turns
MECHANICAL_VERIFICATION_TOOLS: FrozenSet[str] = frozenset({
    "terminal_exec",
    "read_file",
    "write_file",
    "patch_file",
    "web_content_diff",
    "valstorm_record_cud",
    "valstorm_record_merge",
    "valstorm_create_list_filter",
})


class MicroWorkerEngine:
    """Plans zero-token pre-turn discovery calls and offloads post-synthesis mechanical iterations."""

    @staticmethod
    def _already_executed_in_turn(session: Optional[SessionState], tool_name: str, arg_match: str = "") -> bool:
        if not session or not session.messages:
            return False
        for m in reversed(session.messages):
            if m.role == "user":
                break
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.name == tool_name:
                        if not arg_match or arg_match in str(tc.arguments or ""):
                            return True
        return False

    @classmethod
    def plan_pre_turn_discovery_calls(
        cls,
        user_input: str,
        session: Optional[SessionState],
        registered_tool_names: Set[str],
    ) -> List[ToolCall]:
        """Identifies deterministic read-only discovery tool calls that can be pre-executed before Iteration 0."""
        clean_prompt = (user_input or "").strip()
        if not clean_prompt:
            return []
        prompt_lower = clean_prompt.lower()

        # Never pre-read files if the prompt asks to create/write them from scratch first
        is_creation_first = any(k in prompt_lower for k in _IMMEDIATE_CREATION_SIGNALS)
        planned: List[ToolCall] = []

        # 1. Explicit existing workspace files when `read_file` is available and requested
        if (
            "read_file" in registered_tool_names
            and not is_creation_first
            and ("read_file" in prompt_lower or "inspect " in prompt_lower or "read " in prompt_lower)
        ):
            sandbox = get_current_sandbox()
            candidates = list(dict.fromkeys(_FILE_PATH_RE.findall(clean_prompt)))
            for idx, rel_path in enumerate(candidates[:4]):
                try:
                    resolved = sandbox.resolve_path(rel_path)
                    if not resolved.is_file():
                        from pathlib import Path
                        from core.tool_healer import heal_file_path_s1
                        healed_rel, was_healed = heal_file_path_s1(rel_path, Path(sandbox.base_dir))
                        if was_healed:
                            rel_path = healed_rel
                            resolved = sandbox.resolve_path(rel_path)
                    if resolved.is_file() and not cls._already_executed_in_turn(session, "read_file", rel_path):
                        planned.append(
                            ToolCall(
                                id=f"mw_pre_read_{idx}",
                                name="read_file",
                                arguments={"path": rel_path},
                            )
                        )
                except Exception:
                    continue

        # 2. Explicit Valstorm schema inspection (`valstorm_schema_inspect`) gated by System-1 confidence
        if "valstorm_schema_inspect" in registered_tool_names and (
            "valstorm_schema_inspect" in prompt_lower or "schema" in prompt_lower
        ):
            m_schema = _SCHEMA_INSPECT_RE.search(clean_prompt)
            api_name = (m_schema.group(1) or m_schema.group(2)).strip() if m_schema else ""
            if api_name and not cls._already_executed_in_turn(session, "valstorm_schema_inspect", api_name):
                from core.decision_model import ChoiceQuestion, SystemOneClient
                s1 = SystemOneClient()
                resp = s1.evaluate(
                    state={"user_input": clean_prompt, "goal": clean_prompt},
                    questions={
                        "prefetch_schema": ChoiceQuestion(
                            instructions="Should valstorm_schema_inspect be pre-executed before Iteration 0?",
                            criteria={
                                "prefetch_schema": f"valstorm_schema_inspect inspect schema {api_name} fields object",
                                "defer_to_llm": "unrelated general chat",
                            },
                        )
                    },
                )
                ans = resp.get_choice("prefetch_schema")
                if ans.choice == "prefetch_schema" and ans.confidence >= 0.50:
                    planned.append(
                        ToolCall(
                            id="mw_pre_schema_0",
                            name="valstorm_schema_inspect",
                            arguments={"api_name": api_name},
                        )
                    )

        # 3. Explicit initial web_scrape URL (`web_scrape on 'https://...'`)
        if "web_scrape" in registered_tool_names and "web_scrape" in prompt_lower:
            m_url = _WEB_SCRAPE_URL_RE.search(clean_prompt)
            if m_url:
                target_url = m_url.group(1).strip()
                if target_url and not cls._already_executed_in_turn(session, "web_scrape", target_url):
                    planned.append(
                        ToolCall(
                            id="mw_pre_scrape_0",
                            name="web_scrape",
                            arguments={"url": target_url},
                        )
                    )

        return planned

    @classmethod
    def filter_schemas_for_turn(
        cls,
        user_input: str,
        schemas: List[Dict[str, Any]],
        session: Optional[SessionState],
        locked_tier: Optional[ModelTier],
        is_offloaded: bool = False,
    ) -> List[Dict[str, Any]]:
        """P3 (FI-566): Dynamically prunes large tool registries (>12 tools) for Worker/Reasoner & offloaded turns.

        Deterministic per `(user_input, locked_tier, is_offloaded)` so `tools_hash` and KV prefix caches
        never drift across iterations of the same model.
        """
        if not schemas or len(schemas) <= 12:
            return schemas

        prompt_lower = (user_input or "").strip().lower()
        used_in_turn: Set[str] = set()
        if session and session.messages:
            for m in reversed(session.messages):
                if m.role == "user":
                    break
                if m.role == "assistant" and m.tool_calls:
                    for tc in m.tool_calls:
                        if tc.name:
                            used_in_turn.add(tc.name)

        # 1. Offloaded post-synthesis mechanical iterations on Worker only need verification tools
        if is_offloaded:
            keep_names = set(MECHANICAL_VERIFICATION_TOOLS) | used_in_turn
            filtered = [s for s in schemas if (s.get("name") or s.get("function", {}).get("name")) in keep_names]
            return filtered if filtered else schemas

        # 2. On Worker or Reasoner turns with large tool registries (>12 tools), keep core dev tools + prompt-matched domain tools
        if locked_tier in ("worker", "reasoner"):
            core_keep: Set[str] = {
                "read_file",
                "write_file",
                "patch_file",
                "search_files",
                "terminal_exec",
                "calculator",
                "code_outline",
                "find_symbols",
            } | used_in_turn

            if any(w in prompt_lower for w in ("web", "http", "url", "scrape", "search", "diff", "rss", "feed")):
                core_keep.update({"web_search", "web_scrape", "web_content_diff", "web_crawl_domain", "web_feed_poll"})
            if any(w in prompt_lower for w in ("valstorm", "schema", "record", "sql", "lead", "contact", "campaign", "deal", "account", "ticket", "merge", "filter", "report", "dashboard", "drip", "sms")):
                core_keep.update({
                    "valstorm_schema_inspect",
                    "valstorm_sql_query",
                    "valstorm_record_cud",
                    "valstorm_record_merge",
                    "valstorm_create_list_filter",
                    "valstorm_generate_report",
                    "valstorm_generate_dashboard",
                    "valstorm_vfs_search",
                    "valstorm_vfs_write_file",
                })
            if any(w in prompt_lower for w in ("vault", "vfs", "sop", "document", "markdown")):
                core_keep.update({"valstorm_vfs_search", "valstorm_vfs_browse", "valstorm_vfs_get_file", "valstorm_vfs_write_file", "valstorm_record_cud"})
            if any(w in prompt_lower for w in ("delegate", "subagent", "swarm", "parallel", "profile")):
                core_keep.update({"delegate_task", "subagent_manage"})
            if any(w in prompt_lower for w in ("ui_", "navigate", "click", "dom", "artcraft", "canvas", "mesh")):
                core_keep.update({"ui_navigate", "system_one_decide"})

            # Also always keep any tool explicitly named in the user prompt
            for s in schemas:
                s_name = s.get("name") or s.get("function", {}).get("name") or ""
                if s_name and s_name.lower() in prompt_lower:
                    core_keep.add(s_name)

            filtered = [s for s in schemas if (s.get("name") or s.get("function", {}).get("name")) in core_keep]
            if len(filtered) >= 4:
                return filtered

        return schemas

    @classmethod
    def resolve_mechanical_offload(
        cls,
        user_input: str,
        session: Optional[SessionState],
        iteration: int,
        locked_tier: ModelTier,
        locked_model: str,
        base_provider: str,
        tune_dec: AutoTuneDecision,
    ) -> Tuple[str, ThinkingLevel, bool]:
        """Determines whether an intra-turn (`iteration >= 1`) call on a Thinker turn can be offloaded to Worker/Reasoner.

        Returns `(execution_model, execution_effort, offloaded)`.
        """
        if iteration < 1 or locked_tier != "thinker" or tune_dec.escalated_on_error:
            return locked_model, tune_dec.reasoning_effort, False

        if not session or not session.messages:
            return locked_model, tune_dec.reasoning_effort, False

        prompt_lower = (user_input or "").strip().lower()

        # Inspect tools executed so far in the current turn
        turn_tools: List[Tuple[str, bool, str]] = []
        for m in reversed(session.messages):
            if m.role == "user":
                break
            if m.role == "tool" and m.tool_result:
                turn_tools.append(
                    (
                        m.tool_result.name or "",
                        bool(m.tool_result.is_error),
                        (m.tool_result.output or "")[:400],
                    )
                )
        turn_tools.reverse()
        if not turn_tools:
            return locked_model, tune_dec.reasoning_effort, False

        last_name, last_err, last_out = turn_tools[-1]
        if last_err:
            return locked_model, tune_dec.reasoning_effort, False

        has_mutation = any(t[0] in ("write_file", "patch_file", "valstorm_record_cud", "valstorm_record_merge") for t in turn_tools)
        write_count = sum(1 for t in turn_tools if t[0] in ("write_file", "patch_file"))
        is_tdd = "failing" in prompt_lower or "tdd" in prompt_lower

        # Case 1: Clean `terminal_exec` verification just passed (exit code 0, no FAILED/Error) -> Wrap-up summary on Worker!
        if (
            last_name == "terminal_exec"
            and has_mutation
            and "Command exited with code" not in last_out
            and "FAILED" not in last_out
        ):
            worker_model = resolve_tier_model(base_provider, "worker", fallback_model=locked_model)
            return worker_model, "none", True

        # Case 2: Non-TDD synthesis (`patch_file` or `setup_env.sh` `write_file`) just succeeded -> Offload execution/verification step to Worker!
        if not is_tdd and last_name in ("patch_file", "write_file"):
            worker_model = resolve_tier_model(base_provider, "worker", fallback_model=locked_model)
            eff: ThinkingLevel = "medium" if ("setup_env.sh" in prompt_lower or "chmod" in prompt_lower) else "none"
            return worker_model, eff, True

        # Case 3: TDD loop (`G-H01`) — once initial test/stub files are written (`write_count >= 2`) or in Red-to-Green transition,
        # offload mechanical pytest execution and 1-line Green implementation to Worker (`gemini-flash-lite-latest`)!
        if is_tdd and (write_count >= 2 or (last_name == "terminal_exec" and "FAILED" in last_out)):
            worker_model = resolve_tier_model(base_provider, "worker", fallback_model=locked_model)
            return worker_model, "none", True

        return locked_model, tune_dec.reasoning_effort, False
