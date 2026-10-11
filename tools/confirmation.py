"""Confirmation and Human-in-the-Loop Safety Gate Tool for Valstorm Agent Runtime.

Provides:
- confirmation_required: Pauses execution and signals that human confirmation/approval
  is required for high-risk, destructive, or irreversible actions.
"""

import json
import os
from typing import Any, Dict, List, Literal, Optional
import uuid

from core.tools import ToolRegistry, tool


RISK_LEVELS = {"low", "medium", "high", "critical"}


@tool
def confirmation_required(
    action: str,
    details: Dict[str, Any],
    risk_level: str = "high",
) -> str:
    """Pauses agent execution and requests human-in-the-loop confirmation for high-risk actions.

    Use this tool whenever an action is potentially destructive, irreversible, security-sensitive,
    or modifies critical production resources (e.g. deleting databases, overwriting sensitive configs,
    force-pushing git branches, making irreversible API calls, or executing high-privilege commands).

    Args:
        action: A concise description or name of the high-risk action to be approved.
        details: A dictionary containing all relevant parameters, targeted resources, and context.
        risk_level: Risk classification ('low', 'medium', 'high', or 'critical'). Defaults to 'high'.

    Returns:
        A structured confirmation notice with confirmation_id and status PENDING_APPROVAL.
    """
    clean_action = (action or "").strip()
    if not clean_action:
        return "Error: An action description must be provided for confirmation_required."

    norm_risk = (risk_level or "high").strip().lower()
    if norm_risk not in RISK_LEVELS:
        norm_risk = "high"

    if os.environ.get("YOLO_MODE", "").lower() in ("true", "1", "yes"):
        return f"⚡ [YOLO Mode Active] High-risk action '{clean_action}' automatically APPROVED without human confirmation."

    confirmation_id = f"conf_{uuid.uuid4().hex[:12]}"
    details_dict = details if isinstance(details, dict) else {"raw_details": details}

    payload = {
        "status": "CONFIRMATION_REQUIRED",
        "confirmation_id": confirmation_id,
        "action": clean_action,
        "risk_level": norm_risk.upper(),
        "details": details_dict,
        "requires_human_approval": True,
        "message": f"Human confirmation required before proceeding with '{clean_action}'.",
    }

    formatted_json = json.dumps(payload, indent=2, ensure_ascii=False)
    
    return (
        f"⚠️ [CONFIRMATION REQUIRED - RISK LEVEL: {norm_risk.upper()}]\n"
        f"Confirmation ID: {confirmation_id}\n"
        f"Action: {clean_action}\n"
        f"Status: PENDING_APPROVAL\n\n"
        f"Payload:\n{formatted_json}\n\n"
        f"Action paused. Awaiting explicit user confirmation or rejection for confirmation_id '{confirmation_id}'."
    )


def create_confirmation_tools() -> List[Any]:
    """Returns list of confirmation tool functions."""
    return [confirmation_required]


def register_confirmation_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers confirmation tools into the provided ToolRegistry."""
    for tool_fn in create_confirmation_tools():
        registry.register(tool_fn)
    return registry
