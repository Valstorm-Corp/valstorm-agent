"""Agent Runtime Tools for System-1 Decision Evaluation and UI / Crafting Canvas Navigation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.decision_model import SystemOneClient
from core.sandbox import get_current_sandbox
from core.tools import ToolRegistry, tool
from core.ui_navigator import SystemOneUINavigator


def _resolve_sandbox_path(path_str: str) -> Path:
    """Resolves a relative file path against the active sandbox working directory."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    sb = get_current_sandbox()
    base = getattr(sb, "base_dir", None) or getattr(sb, "workdir", None)
    if base:
        return (Path(str(base)) / p).resolve()
    return p.resolve()


def _load_json_or_state(raw: Union[str, Dict[str, Any], List[Any]]) -> Any:
    """Loads state or questions from a dict, JSON string, or sandbox file path."""
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return {"value": str(raw)}

    stripped = raw.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except Exception:
            return stripped

    # Check if `stripped` is a file path in the active sandbox
    if len(stripped) < 260 and ("\n" not in stripped):
        candidate = _resolve_sandbox_path(stripped)
        if candidate.is_file():
            content = candidate.read_text(encoding="utf-8")
            try:
                return json.loads(content)
            except Exception:
                return content

    return stripped


@tool
def system_one_decide(
    state: str,
    questions: str,
    backend: str = "auto",
    output_path: Optional[str] = None,
) -> str:
    """Evaluate a batch of parallel System-1 questions (Choice, Score, Noul, Box) against a state.

    Args:
        state: JSON string, plain text, or file path representing the program, DOM, or canvas state.
        questions: JSON string mapping question keys to typed question definitions (choice, score, noul, box).
        backend: Execution backend ('local', 'jev', or 'auto'). Defaults to 'auto' (local daemon).
        output_path: Optional file path to save the structured JSON response.

    Returns:
        JSON string containing typed answers, calibrated probability distributions, confidence scores, and latency_ms.
    """
    parsed_state = _load_json_or_state(state)
    parsed_questions = _load_json_or_state(questions)
    if not isinstance(parsed_questions, dict):
        raise ValueError("'questions' must be a JSON object mapping question keys to question specs.")

    b_mode = backend.strip().lower() if backend else "auto"
    if b_mode not in ("local", "jev", "auto"):
        b_mode = "auto"

    client = SystemOneClient(backend=b_mode)  # type: ignore[arg-type]
    resp = client.evaluate(state=parsed_state, questions=parsed_questions)
    payload = resp.model_dump()

    if output_path:
        out_file = _resolve_sandbox_path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return json.dumps(payload, indent=2)


@tool
def ui_navigate(
    goal: str,
    ui_state: str,
    assertions: Optional[str] = None,
    mode: str = "workflow",
    max_steps: int = 10,
    confidence_threshold: float = 0.55,
    output_path: Optional[str] = None,
) -> str:
    """Navigate a DOM UI, Command Bus, or 2D/3D Crafting Canvas (e.g. Valstorm Desktop or ArtCraft) using System-1 parallel decisions.

    Selects actions strictly from the closed set of visible/enabled UI or canvas elements
    (0% selector hallucination), computes normalized and pixel click coordinates, checks for
    blocking modals/errors, and verifies E2E assertions in parallel.

    Args:
        goal: Single-step or multi-step UI/canvas navigation goal (e.g. '1. Convert image to 3D mesh -> 2. Pose actor -> 3. Render scene').
        ui_state: JSON string or file path to the UI DOM snapshot, Autodiscovery tree, or ArtCraft canvas state.
        assertions: Optional JSON string mapping assertion names to boolean statements to verify in parallel.
        mode: 'step' for a single decision step, or 'workflow' for a multi-step trajectory loop.
        max_steps: Maximum steps to execute when mode='workflow' (default: 10).
        confidence_threshold: Minimum confidence required before executing an action without escalating to System-2 (default: 0.55).
        output_path: Optional file path to write the resulting navigation decision or trajectory JSON.

    Returns:
        JSON string containing the selected element chain, action primitives, click coordinates, confidence scores, and assertion results.
    """
    parsed_state = _load_json_or_state(ui_state)
    if not isinstance(parsed_state, dict):
        raise ValueError("'ui_state' must resolve to a JSON object/dictionary.")

    parsed_assertions: Optional[Dict[str, str]] = None
    if assertions:
        loaded_a = _load_json_or_state(assertions)
        if isinstance(loaded_a, dict):
            parsed_assertions = {str(k): str(v) for k, v in loaded_a.items()}

    navigator = SystemOneUINavigator(confidence_threshold=float(confidence_threshold))
    clean_mode = (mode or "workflow").strip().lower()

    if clean_mode == "step":
        step_res = navigator.decide_step(
            goal=goal,
            ui_state=parsed_state,
            assertions=parsed_assertions,
            confidence_threshold=float(confidence_threshold),
        )
        payload = step_res.model_dump()
    else:
        traj_res = navigator.run_navigation_workflow(
            goal=goal,
            initial_state=parsed_state,
            assertions=parsed_assertions,
            max_steps=int(max_steps),
            confidence_threshold=float(confidence_threshold),
        )
        payload = traj_res.model_dump()

    if output_path:
        out_file = _resolve_sandbox_path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return json.dumps(payload, indent=2)


def create_ui_navigator_tools() -> List[Any]:
    """Returns the System-1 Decision Model and UI/Canvas Navigator tools."""
    return [system_one_decide, ui_navigate]


def register_ui_navigator_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers System-1 Decision Model and UI Navigator tools into a ToolRegistry."""
    for fn in create_ui_navigator_tools():
        registry.register(fn)
    return registry
