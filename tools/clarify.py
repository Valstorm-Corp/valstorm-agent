"""Interactive User Dialog and Clarification Tool for Valstorm Agent Runtime.

Provides:
- clarify: Requests structured clarification from the user with optional choices
  and multi-select capabilities.
"""

import json
from typing import Any, Dict, List, Optional
import uuid

from core.tools import ToolRegistry, tool


@tool
def clarify(
    question: str,
    choices: Optional[List[str]] = None,
    multi_select: bool = False,
) -> str:
    """Asks the user a structured clarification question to resolve ambiguity before proceeding.

    Use this tool when user instructions are ambiguous, when critical configuration decisions
    need input, or when multiple valid approaches exist and user preference is required.

    Args:
        question: The clear, concise question to ask the user.
        choices: Optional list of predefined options/choices for the user to choose from.
        multi_select: If True, the user can select multiple choices; if False, single selection.

    Returns:
        A structured prompt notice with clarification_id and formatted options.
    """
    clean_question = (question or "").strip()
    if not clean_question:
        return "Error: A question must be provided for clarify."

    clarify_id = f"clarify_{uuid.uuid4().hex[:12]}"
    clean_choices = [c.strip() for c in choices if isinstance(c, str) and c.strip()] if choices else []

    mode = "multi_select" if (multi_select and clean_choices) else ("single_select" if clean_choices else "open_ended")

    payload: Dict[str, Any] = {
        "status": "CLARIFICATION_REQUESTED",
        "clarify_id": clarify_id,
        "question": clean_question,
        "mode": mode,
        "choices": clean_choices,
        "multi_select": multi_select if clean_choices else False,
        "awaiting_user_input": True,
    }

    formatted_json = json.dumps(payload, indent=2, ensure_ascii=False)

    parts = [
        f"❓ [CLARIFICATION REQUESTED - {mode.replace('_', ' ').upper()}]",
        f"Clarification ID: {clarify_id}",
        f"Question: {clean_question}",
    ]

    if clean_choices:
        parts.append("\nOptions:")
        for idx, choice in enumerate(clean_choices, 1):
            parts.append(f"  {idx}. {choice}")

    parts.append(f"\nPayload:\n{formatted_json}")
    parts.append(f"\nExecution paused. Awaiting user response for clarification_id '{clarify_id}'.")

    return "\n".join(parts)


def create_clarify_tools() -> List[Any]:
    """Returns list of clarify tool functions."""
    return [clarify]


def register_clarify_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers clarify tools into the provided ToolRegistry."""
    for tool_fn in create_clarify_tools():
        registry.register(tool_fn)
    return registry
