"""System-1 Adaptive Thinking-Level Router for Valstorm Agent Runtime.

Dynamically selects the optimal `reasoning_effort` ('none', 'low', 'medium', 'high')
before each LLM call using a single-pass System-1 evaluation (<3ms local latency).

Crucial Cache Property:
  Because `reasoning_effort` (`thinking_config`) is an out-of-band API parameter rather
  than part of the prompt text or tool schema array, dynamically switching thinking
  levels per turn or per ReAct iteration preserves 100% of the provider KV prefix cache.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.decision_model import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneClient,
)
from core.models import Message, SessionState

ThinkingLevel = Literal["none", "low", "medium", "high"]


class ThinkingRouteDecision(BaseModel):
    """Telemetry and decision output from the System-1 Thinking-Level Router."""

    reasoning_effort: ThinkingLevel
    complexity_score: float = Field(ge=0.0, le=1.0)
    hazard_probability: float = Field(ge=0.0, le=1.0)
    simple_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    duration_ms: float = 0.0
    reason: str = ""


# Keyword signals calibrated for fast System-1 rubric matching
_HIGH_COMPLEXITY_SIGNALS = (
    "tdd", "failing python unit test", "red-green-refactor", "pytest",
    "setup_env.sh", "chmod", ".sh", "git init", "shell script", "permission",
    "architecture", "refactor", "race condition", "deadlock", "root cause",
    "debug", "security", "migration", "multi-step", "subagent", "swarm",
)

_MEDIUM_COMPLEXITY_SIGNALS = (
    "patch_file", "patch", "web_search", "web_scrape", "web_content_diff",
    "compare", "sum.py", "command-line arguments", "script", "implement",
    "function", "endpoint", "test", "verify",
)

_SIMPLE_SIGNALS = (
    "calculate", "calculator", "single line", "hello agent runtime",
    "index.js", "calculation_result.txt", "create two directories",
    "search_files", "count", "hi", "hello", "thanks", "thank you", "status",
)


class SystemOneThinkingRouter:
    """Routes turns to ('none', 'low', 'medium', 'high') thinking effort via System-1 evaluation."""

    def __init__(
        self,
        client: Optional[SystemOneClient] = None,
        confidence_threshold: float = 0.50,
    ):
        self.client = client or SystemOneClient(backend="local")
        self.confidence_threshold = confidence_threshold

    def route_turn(
        self,
        user_input: str,
        session: Optional[SessionState] = None,
        iteration: int = 0,
    ) -> ThinkingRouteDecision:
        """Evaluates the turn context in a single parallel System-1 pass to pick `reasoning_effort`."""
        t0 = time.perf_counter()
        clean_prompt = (user_input or "").strip()
        prompt_lower = clean_prompt.lower()

        # Inspect last message/tool result in session if mid-loop
        last_tool_error = False
        last_tool_name = ""
        if session and session.messages:
            for m in reversed(session.messages[-4:]):
                if m.role == "tool" and m.tool_result:
                    last_tool_name = m.tool_result.name
                    if m.tool_result.is_error:
                        last_tool_error = True
                    break

        state = {
            "user_input": clean_prompt,
            "goal": clean_prompt,
            "iteration": iteration,
            "last_tool_name": last_tool_name,
            "last_tool_error": last_tool_error,
            "error": last_tool_error,
        }

        questions = {
            "effort_choice": ChoiceQuestion(
                instructions="Select the minimum sufficient reasoning effort level for this task",
                criteria={
                    "none": "Simple arithmetic calculation, single-line file write, directory creation, or conversational chat",
                    "low": "Basic file read, glob search, or straightforward single-tool lookup",
                    "medium": "Code generation with CLI execution, surgical patch_file edit, or web search synthesis",
                    "high": "Multi-step Red-Green-Refactor TDD pytest loop, shell script creation with executable permissions and git init, or debugging errors",
                },
            ),
            "task_complexity": ScoreQuestion(
                instructions="Score the algorithmic and operational complexity of this request",
                criteria={
                    "trivial": "Conversational greeting, simple math calculator call, or writing a single line to a file",
                    "routine": "Creating directories, searching files, or running a basic script",
                    "moderate": "Patching existing configuration files, scraping web pages, or writing a CLI script",
                    "complex": "Writing failing unit tests first (TDD), managing shell script chmod permissions, or recovering from tool failures",
                },
            ),
            "operational_hazard": NoulQuestion(
                instructions="Does this task involve shell scripts (.sh), git init, executable permissions (chmod), failing unit tests (pytest TDD), or tool error recovery?",
            ),
            "is_simple_io": NoulQuestion(
                instructions="Is this a simple calculation, single-line file write, or directory file count task?",
            ),
        }

        resp = self.client.evaluate(state=state, questions=questions)
        c_ans = resp.get_choice("effort_choice")
        s_ans = resp.get_score("task_complexity")
        hazard_ans = resp.get_noul("operational_hazard")
        simple_ans = resp.get_noul("is_simple_io")

        # Combine System-1 signals with deterministic safety rules
        has_high_kw = any(k in prompt_lower for k in _HIGH_COMPLEXITY_SIGNALS)
        has_med_kw = any(k in prompt_lower for k in _MEDIUM_COMPLEXITY_SIGNALS)
        has_simple_kw = any(k in prompt_lower for k in _SIMPLE_SIGNALS)

        # Follow-up iterations after tool execution (when no error occurred) can step down effort
        if last_tool_error:
            chosen: ThinkingLevel = "high"
            reason = f"Escalated to 'high' due to prior tool error on '{last_tool_name}'."
        elif has_high_kw or hazard_ans.noul >= 0.68:
            # On iteration 0/1 of a hazardous or TDD task, use high thinking; on final wrap-up step, low is sufficient
            if iteration <= 2:
                chosen = "high"
                reason = f"Operational hazard / TDD complexity detected (hazard_p={hazard_ans.noul:.2f}); routed to 'high'."
            else:
                chosen = "low"
                reason = f"Follow-up iteration ({iteration}) after core hazard resolution; stepped down to 'low'."
        elif has_simple_kw and not has_med_kw:
            chosen = "none"
            reason = f"Simple math/IO or conversational turn detected (simple_p={simple_ans.noul:.2f}); routed to 'none'."
        elif has_med_kw or s_ans.score >= 0.45:
            chosen = "low" if iteration >= 1 else "medium"
            reason = f"Moderate task complexity (score={s_ans.score:.2f}, iter={iteration}); routed to '{chosen}'."
        else:
            raw_choice = c_ans.choice
            if raw_choice in ("none", "low", "medium", "high"):
                chosen = raw_choice  # type: ignore[assignment]
            else:
                chosen = "low"
            reason = f"System-1 choice '{chosen}' (confidence={c_ans.confidence:.2f})."

        dur_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return ThinkingRouteDecision(
            reasoning_effort=chosen,
            complexity_score=s_ans.score,
            hazard_probability=hazard_ans.noul,
            simple_probability=simple_ans.noul,
            confidence=c_ans.confidence,
            duration_ms=dur_ms,
            reason=reason,
        )
