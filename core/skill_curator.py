"""Hermes-Style Post-Trajectory Skill & Pitfall Curator for Valstorm Agent Runtime.

Provides:
1. Fast System-1 Trajectory Gate (`evaluate_trajectory_curation_candidate`):
   - Evaluates tool execution traces in <1ms with zero LLM overhead for routine turns.
   - Triggers only on:
     a) `error_recovery`: A tool call failed (`is_error=True` or traceback) and the agent subsequently
        recovered and succeeded (`is_error=False`).
     b) `multi_step_workflow`: Complex multi-tool procedural execution (>= 5 tool calls across >= 2 tools).
     c) `explicit_directive`: User explicitly asked to codify/save a skill or invoked `/learn`.
2. Untrusted Ingestion Quarantine Gate (`is_session_tainted_by_untrusted_ingestion`):
   - Prevents persistent prompt-injection attacks by blocking autonomous background skill creation
     or patching when the turn ingested external web/feed content (`UNTRUSTED_INGESTION_TOOLS`).
3. Prefer-Patch-Over-Create Distillation (`curate_and_commit_turn_trajectory`):
   - Inspects the active skill index (`_index_all_skills`) and prefers appending discovered debugging
     lessons to an existing skill's `## Pitfalls` section (auto-adopting `bundled` skills first)
     over creating duplicate micro-skills.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from core.models import Message, SessionState
from core.security_guard import SecurityGuard
from providers.base import BaseProvider
from tools.skill_tool import _index_all_skills, execute_skill_mutation

logger = logging.getLogger("vsagent.skill_curator")

_EXPLICIT_SKILL_DIRECTIVE_RE = re.compile(
    r"(^/learn\b|"
    r"\b(save|turn|codify|record|capture|document|extract)\s+(this|that|the)\s+(workflow|procedure|runbook|pattern|process|approach|fix)\s+(as|into)\s+(a\s+)?skill\b|"
    r"\bcreate\s+a\s+(reusable\s+)?skill\s+(for|from)\s+this\b|"
    r"\bremember\s+how\s+we\s+(did|fixed|solved|ran)\s+this\b)",
    re.IGNORECASE,
)

_ERROR_OUTPUT_HINT_RE = re.compile(
    r"(Traceback \(most recent call last\)|ModuleNotFoundError:|ImportError:|SyntaxError:|TypeError:|ValueError:|"
    r"PermissionError:|Command exited with code [1-9]|^Error:)",
    re.IGNORECASE | re.MULTILINE,
)

CURATOR_SYSTEM_PROMPT = """You are the Hermes Procedural Skill Curator for the Valstorm AI Agent Runtime.
Your job is to review a completed multi-step agent trajectory (User Input, Tool Execution Trace, and Final Resolution) and decide whether a reusable procedural skill or debugging pitfall should be saved for future sessions.

### Strict Curation Rules:
1. **PREFER `patch` OVER `create`**:
   - Review the `Existing Skills Index` carefully.
   - If the lesson, command flag, environment variable, or gotcha belongs to an existing skill's domain, return `"action": "patch"` with `"slug": "<existing_slug>"`, `"section": "Pitfalls"`, and `"content_to_append": "- <concise 1-2 sentence actionable rule>"`.
2. **ONLY `create` FOR DISTINCT REUSABLE WORKFLOWS**:
   - Only return `"action": "create"` when the trajectory solved a reusable multi-step procedure not covered by any existing skill, or when the user explicitly asked to create a new skill.
   - Never create skills for one-off typos, trivial file reads, or transient data lookups.
3. **RETURN `noop` WHEN NOTHING GENERALIZES**:
   - If the trajectory contains no reusable engineering/operational lesson, return `{"action": "noop"}`.
4. **OUTPUT FORMAT**:
   - Respond ONLY with a single valid JSON object (no markdown fences needed):
     - For patching an existing skill:
       `{"action": "patch", "slug": "existing-skill-slug", "section": "Pitfalls", "content_to_append": "- Always set PYTHONPATH=apps/api/app before running apps/api pytest suites."}`
     - For creating a new skill:
       `{"action": "create", "slug": "kebab-case-slug", "category": "valstorm-internal", "name": "Human Title", "description": "1-line trigger condition", "body": "# Human Title\\n\\n## Procedure\\n1. Step one...\\n\\n## Pitfalls\\n- Watch out for...", "tags": ["tag1", "tag2"]}`
     - For no-op:
       `{"action": "noop"}`
"""


def extract_turn_tool_trace(session: Optional[SessionState]) -> List[Dict[str, Any]]:
    """Extracts the ordered list of tool calls and outcomes from the latest turn in `session`."""
    if not session or not getattr(session, "messages", None):
        return []

    messages = session.messages
    last_user_idx = 0
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].role == "user":
            last_user_idx = idx
            break

    call_args_by_id: Dict[str, Dict[str, Any]] = {}
    trace: List[Dict[str, Any]] = []

    for msg in messages[last_user_idx:]:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                call_args_by_id[tc.id] = tc.arguments or {}
        if msg.role == "tool" and msg.tool_result is not None:
            tr = msg.tool_result
            out_str = str(tr.output or "")
            inferred_error = bool(tr.is_error or _ERROR_OUTPUT_HINT_RE.search(out_str[:600]))
            trace.append({
                "call_id": tr.call_id,
                "name": tr.name,
                "arguments": call_args_by_id.get(tr.call_id, {}),
                "output_preview": out_str[:400],
                "is_error": inferred_error,
            })

    return trace


def is_session_tainted_by_untrusted_ingestion(
    session: Optional[SessionState] = None,
    tool_trace: Optional[List[Dict[str, Any]]] = None,
    untrusted_tainted: bool = False,
) -> bool:
    """Returns True if the session or turn ingested untrusted external web/feed content."""
    if untrusted_tainted:
        return True
    if session and isinstance(getattr(session, "metadata", None), dict):
        if session.metadata.get("untrusted_content_tainted"):
            return True

    effective_trace = tool_trace if tool_trace is not None else extract_turn_tool_trace(session)
    for item in effective_trace:
        if SecurityGuard.is_untrusted_ingestion_tool(item.get("name")):
            return True

    return False


def evaluate_trajectory_curation_candidate(
    user_input: str,
    assistant_output: str,
    session: Optional[SessionState] = None,
    tool_trace: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fast (<1ms) System-1 trajectory complexity gate for post-turn skill curation."""
    clean_input = (user_input or "").strip()
    clean_output = (assistant_output or "").strip()

    effective_trace = tool_trace if tool_trace is not None else extract_turn_tool_trace(session)
    tool_names = [str(t.get("name") or "") for t in effective_trace if t.get("name")]

    # 1. Skip if the agent already called skill_manage directly during this turn
    if "skill_manage" in tool_names:
        return {
            "should_curate": False,
            "trigger_reason": "already_managed_in_turn",
            "tool_count": len(effective_trace),
        }

    # 2. Check explicit user directive (/learn or "save this workflow as a skill")
    if clean_input and _EXPLICIT_SKILL_DIRECTIVE_RE.search(clean_input):
        return {
            "should_curate": True,
            "trigger_reason": "explicit_directive",
            "tool_count": len(effective_trace),
        }

    if not effective_trace or not clean_output:
        return {
            "should_curate": False,
            "trigger_reason": "trivial_trajectory",
            "tool_count": len(effective_trace),
        }

    # 3. Check error-recovery signal (at least one error followed by at least one success)
    first_error_idx = None
    last_success_idx = None
    for idx, step in enumerate(effective_trace):
        if step.get("is_error") and first_error_idx is None:
            first_error_idx = idx
        elif not step.get("is_error"):
            last_success_idx = idx

    if first_error_idx is not None and last_success_idx is not None and last_success_idx > first_error_idx:
        return {
            "should_curate": True,
            "trigger_reason": "error_recovery",
            "tool_count": len(effective_trace),
        }

    # 4. Check multi-step workflow signal (>= 5 tool calls across >= 2 distinct tools)
    distinct_tools = set(tool_names)
    if len(effective_trace) >= 5 and len(distinct_tools) >= 2:
        return {
            "should_curate": True,
            "trigger_reason": "multi_step_workflow",
            "tool_count": len(effective_trace),
        }

    return {
        "should_curate": False,
        "trigger_reason": "trivial_trajectory",
        "tool_count": len(effective_trace),
    }


def _parse_curator_json(raw_text: str) -> Dict[str, Any]:
    """Extracts a JSON object from the curator LLM output."""
    text = (raw_text or "").strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"action": "noop"}


async def curate_and_commit_turn_trajectory(
    user_input: str,
    assistant_output: str,
    session: Optional[SessionState],
    provider: BaseProvider,
    model: Optional[str] = None,
    untrusted_tainted: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Evaluates a completed turn trajectory and autonomously patches or creates a skill when warranted."""
    try:
        trace = extract_turn_tool_trace(session)

        # 1. Quarantine Gate: Never auto-curate skills from turns tainted by untrusted web/feed ingestion
        if is_session_tainted_by_untrusted_ingestion(
            session=session,
            tool_trace=trace,
            untrusted_tainted=untrusted_tainted,
        ):
            logger.info("[SkillCurator] Quarantine gate triggered: skipping autonomous skill curation on untrusted turn.")
            return {
                "status": "quarantined",
                "reason": "untrusted_external_content_ingested",
            }

        # 2. Fast System-1 Trajectory Gate
        gate = evaluate_trajectory_curation_candidate(
            user_input=user_input,
            assistant_output=assistant_output,
            session=session,
            tool_trace=trace,
        )
        if not force and not gate["should_curate"]:
            return {
                "status": "skipped",
                "reason": gate["trigger_reason"],
            }

        # 3. Build existing skills catalog for Prefer-Patch-Over-Create grounding
        skills = await asyncio.to_thread(_index_all_skills)
        skills_catalog_lines = []
        skills_by_slug: Dict[str, Dict[str, Any]] = {}
        for s in skills[:80]:
            s_slug = s["slug"]
            skills_by_slug[s_slug] = s
            skills_catalog_lines.append(
                f"- {s_slug} (category={s.get('category')}, ownership={s.get('ownership', 'bundled')}): {s.get('description', '')[:120]}"
            )
        skills_catalog_text = "\n".join(skills_catalog_lines) if skills_catalog_lines else "(No existing skills indexed)"

        trace_lines = []
        for idx, step in enumerate(trace[:15], 1):
            status_tag = "ERROR" if step.get("is_error") else "OK"
            args_str = json.dumps(step.get("arguments") or {})[:200]
            out_str = str(step.get("output_preview") or "").replace("\n", " ")[:220]
            trace_lines.append(f"{idx}. [{status_tag}] {step.get('name')}({args_str}) -> {out_str}")
        trace_summary = "\n".join(trace_lines) if trace_lines else "(No tool calls recorded)"

        curator_prompt = (
            f"Trigger Reason: {gate['trigger_reason']}\n\n"
            f"### Existing Skills Index:\n{skills_catalog_text}\n\n"
            f"### User Input:\n{(user_input or '')[:1500]}\n\n"
            f"### Tool Execution Trajectory:\n{trace_summary}\n\n"
            f"### Final Assistant Resolution:\n{(assistant_output or '')[:2000]}\n\n"
            f"Decide whether to `patch` an existing skill's Pitfalls section, `create` a new skill, or `noop`."
        )

        target_model = model or getattr(provider, "default_model", None) or "gemini-flash-latest"
        response_msg = await provider.chat_complete(
            messages=[
                Message(role="system", content=CURATOR_SYSTEM_PROMPT),
                Message(role="user", content=curator_prompt),
            ],
            model=target_model,
            temperature=0.0,
        )

        decision = _parse_curator_json(response_msg.content or "")
        act = str(decision.get("action") or "noop").strip().lower()

        if act == "noop" or act not in ("patch", "create"):
            return {"status": "noop", "action": "noop", "trigger_reason": gate["trigger_reason"]}

        target_slug = str(decision.get("slug") or "").strip().lower()
        if not target_slug:
            return {"status": "noop", "action": "noop", "reason": "missing_slug"}

        if act == "patch":
            existing_skill = skills_by_slug.get(target_slug)
            # If the skill exists and is currently 'bundled', automatically adopt it first so the patch succeeds!
            if existing_skill and existing_skill.get("ownership") == "bundled":
                await execute_skill_mutation(action="adopt", slug=target_slug)

            receipt = await execute_skill_mutation(
                action="patch",
                slug=target_slug,
                section=str(decision.get("section") or "Pitfalls"),
                content_to_append=str(decision.get("content_to_append") or ""),
                old_text=decision.get("old_text"),
                new_text=decision.get("new_text"),
            )
            receipt["trigger_reason"] = gate["trigger_reason"]
            logger.info(f"[SkillCurator] Curated skill patch: {receipt}")
            return receipt

        # act == "create"
        receipt = await execute_skill_mutation(
            action="create",
            slug=target_slug,
            category=str(decision.get("category") or "valstorm-internal"),
            name=decision.get("name"),
            description=decision.get("description"),
            body=decision.get("body"),
            tags=decision.get("tags") if isinstance(decision.get("tags"), list) else None,
        )
        receipt["trigger_reason"] = gate["trigger_reason"]
        logger.info(f"[SkillCurator] Curated new skill: {receipt}")
        return receipt

    except Exception as e:
        logger.warning(f"[SkillCurator] Background trajectory curation failed: {e}")
        return {"status": "error", "message": str(e)}
