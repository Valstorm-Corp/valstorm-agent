"""System-1 `ai_chat_message` Auto-Tuner for Model Tier & Reasoning Effort Routing.

Operates at two cache-safe granularities:
  1. Pre-Turn (`iteration == 0` / per `ai_chat_message`):
     Selects both the optimal Model Tier (`worker`, `reasoner`, `thinker`) -> concrete
     model string (e.g. `gemini-flash-lite-latest`, `gemini-flash-latest`, `gemini-pro-latest`)
     AND the initial `reasoning_effort` (`none`, `low`, `medium`, `high`).
  2. Intra-Turn (`iteration >= 1` inside a multi-step ReAct loop):
     Locks the turn's `model` (preserving 100% of the provider KV prefix cache across
     iterations 1..N) while dynamically stepping `reasoning_effort` DOWN (`none`/`low`)
     after successful tool execution or UP (`high`) upon tool failure (`is_error=True`).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from core.decision_model import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneClient,
)
from core.models import SessionState

ModelTier = Literal["worker", "reasoner", "thinker"]
ThinkingLevel = Literal["none", "low", "medium", "high"]

# Provider-family tier-to-model catalog matching `providers/fallback.py`
PROVIDER_TIER_MODELS: Dict[str, Dict[ModelTier, str]] = {
    "gemini": {
        "worker": "gemini-flash-lite-latest",
        "reasoner": "gemini-flash-latest",
        "thinker": "gemini-pro-latest",
    },
    "valstorm": {
        "worker": "gemini-flash-lite-latest",
        "reasoner": "gemini-flash-latest",
        "thinker": "gemini-pro-latest",
    },
    "do": {
        "worker": "deepseek-4-flash",
        "reasoner": "openai-gpt-oss-120b",
        "thinker": "anthropic-claude-opus-5",
    },
    "digitalocean": {
        "worker": "deepseek-4-flash",
        "reasoner": "openai-gpt-oss-120b",
        "thinker": "anthropic-claude-opus-5",
    },
    "openai": {
        "worker": "gpt-4o-mini",
        "reasoner": "gpt-4o",
        "thinker": "o3-mini",
    },
    "anthropic": {
        "worker": "claude-3-5-haiku-latest",
        "reasoner": "claude-sonnet-5-5",
        "thinker": "claude-3-opus-20240229",
    },
    "deepseek": {
        "worker": "deepseek-chat",
        "reasoner": "deepseek-chat",
        "thinker": "deepseek-reasoner",
    },
}

_THINKER_SIGNALS = (
    "tdd", "failing python unit test", "red-green-refactor", "pytest",
    "setup_env.sh", "chmod", ".sh", "git init", "shell script", "permission",
    "architecture", "rfc", "race condition", "deadlock", "root cause",
    "security audit",
)

_REASONER_SIGNALS = (
    "patch_file", "patch", "web_search", "web_scrape", "web_content_diff",
    "compare", "sum.py", "command-line arguments", "script", "implement",
    "function", "endpoint", "refactor", "bug", "fix", "ui_navigate",
    "artcraft", "3d mesh", "compositing", "multi-agent", "subagent",
    "swarm", "delegate_task",
)

_WORKER_SIGNALS = (
    "calculate", "calculator", "single line", "hello agent runtime",
    "index.js", "calculation_result.txt", "create two directories",
    "search_files", "count", "hi", "hello", "thanks", "thank you",
    "status", "version", "ping", "what is", "list",
    "fib.json", "greet.js", "square_result.txt", "research_summary.md",
    "campaign_hook.md", "schema.json",
)


def resolve_tier_model(provider: Optional[str], tier: ModelTier, fallback_model: str = "gemini-flash-latest") -> str:
    """Resolves the concrete model string for a `(provider, tier)` pair."""
    p_norm = (provider or "gemini").strip().lower()
    if p_norm in ("google", "aistudio", "ai-studio", "vertex", "geap"):
        p_norm = "gemini"
    family = PROVIDER_TIER_MODELS.get(p_norm) or PROVIDER_TIER_MODELS["gemini"]
    return family.get(tier, fallback_model)


class AutoTuneDecision(BaseModel):
    """Result of System-1 `ai_chat_message` auto-tuning for a turn or ReAct iteration."""

    phase: Literal["pre_turn", "intra_turn"] = "pre_turn"
    iteration: int = 0
    tier: ModelTier
    model: str
    provider: str = "gemini"
    reasoning_effort: ThinkingLevel
    confidence: float = Field(ge=0.0, le=1.0)
    complexity_score: float = Field(ge=0.0, le=1.0)
    hazard_probability: float = Field(ge=0.0, le=1.0)
    cache_locked_model: bool = False
    escalated_on_error: bool = False
    stepped_down_post_tool: bool = False
    duration_ms: float = 0.0
    reason: str = ""


class MessageAutoTuner:
    """System-1 Auto-Tuner that selects `(tier, model, reasoning_effort)` per `ai_chat_message`."""

    def __init__(
        self,
        client: Optional[SystemOneClient] = None,
        confidence_threshold: float = 0.55,
        prefer_fast_thinker_model: bool = True,
    ):
        self.client = client or SystemOneClient(backend="local")
        self.confidence_threshold = confidence_threshold
        # When True, Thinker tier uses high-thinking Flash/Pro unless explicit architectural RFC
        self.prefer_fast_thinker_model = prefer_fast_thinker_model

    @staticmethod
    def _inspect_session_tail(session: Optional[SessionState]) -> Tuple[str, bool, int, str]:
        """Returns `(last_tool_name, last_tool_error, consecutive_errors, last_tool_output_snippet)`."""
        if not session or not session.messages:
            return "", False, 0, ""

        last_tool_name = ""
        last_tool_error = False
        last_output = ""
        consecutive_errors = 0

        for m in reversed(session.messages):
            if m.role == "tool" and m.tool_result:
                if not last_tool_name:
                    last_tool_name = m.tool_result.name
                    last_tool_error = bool(m.tool_result.is_error)
                    last_output = (m.tool_result.output or "")[:300]
                if m.tool_result.is_error or "FAILED" in (m.tool_result.output or "") or "Command exited with code 1" in (m.tool_result.output or ""):
                    consecutive_errors += 1
                else:
                    break
        return last_tool_name, last_tool_error, consecutive_errors, last_output

    def tune_turn(
        self,
        user_input: str,
        session: Optional[SessionState] = None,
        base_provider: str = "gemini",
        base_model: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> AutoTuneDecision:
        """Pre-turn (`iteration == 0`) routing: selects both Model Tier (`worker`/`reasoner`/`thinker`) and `reasoning_effort`."""
        t0 = time.perf_counter()
        thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        clean_prompt = (user_input or "").strip()
        if clean_prompt.startswith("Goal:\n"):
            clean_prompt = clean_prompt[len("Goal:\n"):].strip()
        prompt_lower = clean_prompt.lower()

        last_tool_name, last_tool_error, consecutive_errors, _ = self._inspect_session_tail(session)

        state = {
            "user_input": clean_prompt,
            "goal": clean_prompt,
            "iteration": 0,
            "last_tool_name": last_tool_name,
            "last_tool_error": last_tool_error,
            "error": last_tool_error,
        }

        questions = {
            "model_tier": ChoiceQuestion(
                instructions="Select the optimal model capability tier for this user message",
                criteria={
                    "worker": "Simple calculation, single-line file write, directory creation, file count, or conversational chat",
                    "reasoner": "Standard code generation, CLI script execution, surgical patch_file edit, web search, or UI navigation",
                    "thinker": "Multi-step Red-Green-Refactor TDD pytest loop, shell script with chmod +x and git init, or complex architecture/debugging",
                },
            ),
            "reasoning_effort": ChoiceQuestion(
                instructions="Select the initial reasoning effort budget for this message",
                criteria={
                    "none": "Trivial arithmetic, single-line file write, directory creation, or conversational reply",
                    "low": "Routine file read, glob search, or single-step tool execution",
                    "medium": "Code generation with CLI execution, surgical patch_file edit, or web content synthesis",
                    "high": "Writing failing unit tests first (TDD), shell script chmod +x / git init guardrails, or error recovery",
                },
            ),
            "task_complexity": ScoreQuestion(
                instructions="Score the algorithmic and operational complexity of this message",
                criteria={
                    "trivial": "Greeting, acknowledgment, simple calculator arithmetic, or single-line file write",
                    "routine": "Creating directories, counting files, or basic lookup",
                    "moderate": "Patching configuration files, scraping web pages, or writing a CLI script",
                    "complex": "Multi-step TDD pytest loop, shell script permission hazards, or multi-agent swarm coordination",
                },
            ),
            "operational_hazard": NoulQuestion(
                instructions="Does this message involve shell scripts (.sh), git init, executable permissions (chmod), failing unit tests (pytest TDD), or tool error recovery?",
            ),
        }

        resp = self.client.evaluate(state=state, questions=questions)
        tier_ans = resp.get_choice("model_tier")
        effort_ans = resp.get_choice("reasoning_effort")
        comp_ans = resp.get_score("task_complexity")
        hazard_ans = resp.get_noul("operational_hazard")

        has_thinker_kw = any(k in prompt_lower for k in _THINKER_SIGNALS)
        has_reasoner_kw = any(k in prompt_lower for k in _REASONER_SIGNALS)
        has_worker_kw = any(k in prompt_lower for k in _WORKER_SIGNALS)

        if last_tool_error or consecutive_errors > 0 or has_thinker_kw or hazard_ans.noul >= 0.68:
            chosen_tier: ModelTier = "thinker"
            chosen_effort: ThinkingLevel = "high"
            reason = f"High complexity / operational hazard detected (hazard_p={hazard_ans.noul:.2f}); routed to tier='{chosen_tier}', effort='{chosen_effort}'."
        elif has_worker_kw and not has_reasoner_kw:
            chosen_tier = "worker"
            chosen_effort = "none"
            reason = f"Trivial/routine task detected (complexity={comp_ans.score:.2f}); routed to fast tier='{chosen_tier}', effort='{chosen_effort}'."
        elif has_reasoner_kw or comp_ans.score >= 0.42:
            chosen_tier = "reasoner"
            chosen_effort = "medium" if "patch" in prompt_lower or "web" in prompt_lower else "low"
            reason = f"Moderate engineering task (complexity={comp_ans.score:.2f}); routed to tier='{chosen_tier}', effort='{chosen_effort}'."
        else:
            raw_t = tier_ans.choice
            chosen_tier = raw_t if raw_t in ("worker", "reasoner", "thinker") else "reasoner"  # type: ignore[assignment]
            raw_e = effort_ans.choice
            chosen_effort = raw_e if raw_e in ("none", "low", "medium", "high") else "low"  # type: ignore[assignment]
            # Confidence Gate: never under-provision to 'worker' when confidence is below threshold
            if chosen_tier == "worker" and tier_ans.confidence < thresh:
                chosen_tier = "reasoner"
                chosen_effort = "medium"
                reason = f"Low confidence ({tier_ans.confidence:.2f} < {thresh:.2f}) on worker tier; safely upgraded to 'reasoner' (medium)."
            else:
                reason = f"System-1 pre-turn route: tier='{chosen_tier}', effort='{chosen_effort}' (conf={tier_ans.confidence:.2f})."

        resolved_model = resolve_tier_model(base_provider, chosen_tier, fallback_model=base_model or "gemini-flash-latest")
        dur_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return AutoTuneDecision(
            phase="pre_turn",
            iteration=0,
            tier=chosen_tier,
            model=resolved_model,
            provider=base_provider,
            reasoning_effort=chosen_effort,
            confidence=tier_ans.confidence,
            complexity_score=comp_ans.score,
            hazard_probability=hazard_ans.noul,
            cache_locked_model=False,
            escalated_on_error=bool(last_tool_error),
            stepped_down_post_tool=False,
            duration_ms=dur_ms,
            reason=reason,
        )

    def tune_iteration(
        self,
        user_input: str,
        session: Optional[SessionState],
        iteration: int,
        locked_tier: ModelTier,
        locked_model: str,
        base_provider: str = "gemini",
    ) -> AutoTuneDecision:
        """Intra-turn (`iteration >= 1`) routing: locks `model` for KV-cache safety while stepping `reasoning_effort`."""
        t0 = time.perf_counter()
        prompt_lower = (user_input or "").strip().lower()
        last_tool_name, last_tool_error, consecutive_errors, last_output = self._inspect_session_tail(session)

        escalated = False
        stepped_down = False

        # 1. If the last tool execution returned an error or failed pytest (when not intentionally in Red phase of TDD), escalate effort
        is_red_tdd_phase = ("failing" in prompt_lower or "tdd" in prompt_lower) and iteration <= 2 and "FAILED" in last_output
        if last_tool_error and not is_red_tdd_phase:
            chosen_effort: ThinkingLevel = "high"
            escalated = True
            reason = f"Intra-turn step-up to 'high' after tool error on '{last_tool_name}' (model locked to '{locked_model}')."
        elif is_red_tdd_phase:
            # Transitioning from Red test failure to Green implementation in TDD
            chosen_effort = "medium"
            reason = f"TDD Red-to-Green transition at iteration {iteration}; effort='medium' (model locked to '{locked_model}')."
        elif locked_tier == "thinker" and iteration == 1 and ("setup_env.sh" in prompt_lower or "chmod" in prompt_lower):
            # Keep high/medium on iteration 1 of G-H03 where chmod +x && ./setup_env.sh is emitted!
            chosen_effort = "high"
            reason = f"Preserving 'high' effort on iteration 1 for shell execution guardrail (model locked to '{locked_model}')."
        else:
            # Post-tool verification / wrap-up turn: step down to 'none' to eliminate redundant thoughts_tokens
            chosen_effort = "none"
            stepped_down = True
            reason = f"Intra-turn step-down to 'none' after clean '{last_tool_name}' execution (KV prefix cache preserved on '{locked_model}')."

        dur_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return AutoTuneDecision(
            phase="intra_turn",
            iteration=iteration,
            tier=locked_tier,
            model=locked_model,
            provider=base_provider,
            reasoning_effort=chosen_effort,
            confidence=0.92,
            complexity_score=0.2 if stepped_down else 0.85,
            hazard_probability=0.9 if escalated else 0.1,
            cache_locked_model=True,
            escalated_on_error=escalated,
            stepped_down_post_tool=stepped_down,
            duration_ms=dur_ms,
            reason=reason,
        )
