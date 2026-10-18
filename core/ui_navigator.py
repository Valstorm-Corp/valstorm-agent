"""System-1 UI, DOM, Command Bus & 2D/3D Crafting Canvas Navigator.

Uses the System-1 Decision Engine (`core/decision_model.py`) to navigate interactive
interfaces, E2E test workflows, and creative crafting canvases (e.g. Valstorm Desktop
and `storytold/artcraft`) in 5ms-50ms per step with zero selector hallucination.

Core Capabilities:
  1. Closed-Set Candidate Extraction: Normalizes DOM elements, Command Bus entries,
     routes, and 2D/3D canvas tools/actors/layers/cameras into typed candidate maps.
  2. Single-Pass Parallel Step Evaluation: Simultaneously evaluates target element `Choice`,
     action primitive `Choice`, `Box` click coordinates, `blocker_detected` (`Noul`),
     `screen_readiness` (`Score`), and N E2E assertions (`Noul`) in one forward pass.
  3. Confidence-Gated Routing & Loop Prevention: Escalates ambiguous states to System-2
     when confidence < threshold and penalizes already-executed actions in multi-step flows.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from core.decision_model import (
    BoxAnswer,
    BoxQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    QuestionSpec,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneClient,
    SystemOneResponse,
    _tokenize,
    _STOPWORDS,
)


# Standard action primitives across DOM UIs and 2D/3D Crafting Canvases (ArtCraft)
UI_ACTION_PRIMITIVES: Dict[str, str] = {
    "click": "Click, press, activate, or trigger a button, tab, menu item, or command",
    "type": "Type, input, fill, or enter text or search query into an input field",
    "select": "Select or switch an option from a dropdown, combobox, filter, or list",
    "drag": "Drag, move, reposition, or slider-adjust an element or layer on canvas",
    "pose_actor": "Pose, rig, rotate joints, or position a 3D character actor or mannequin",
    "camera_frame": "Adjust camera placement, lens FOV, orbit angle, or shot framing in 3D viewport",
    "composite_layer": "Add, mask, remove background, or blend 2D/3D image layers and props",
    "convert_3d_mesh": "Convert 2D image into a 3D mesh object or depth-mapped geometry",
    "render_scene": "Render, generate, synthesize, bake, or export the composed video or image scene",
    "dismiss_modal": "Close, dismiss, or acknowledge a blocking modal, dialog, or error toast",
}


class UICandidate(BaseModel):
    """Normalized representation of an actionable UI element, Command Bus item, or Canvas tool/node."""

    id: str
    label: str
    category: Literal["dom_element", "command_bus", "route", "canvas_tool", "scene_node", "layer", "timeline"] = "dom_element"
    role: str = "button"
    description: str = ""
    action_hint: Optional[str] = None
    enabled: bool = True
    visible: bool = True
    box: Optional[Tuple[float, float, float, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_criteria_string(self) -> str:
        parts = [f"[{self.category}:{self.role}]", self.label]
        if self.description and self.description.lower() != self.label.lower():
            parts.append(f"— {self.description}")
        if self.action_hint:
            parts.append(f"(action: {self.action_hint})")
        return " ".join(parts).strip()


class UIStepDecision(BaseModel):
    """Result of a single System-1 UI / Canvas navigation decision step."""

    route_status: Literal["execute_action", "handle_blocker", "goal_complete", "escalate_system2"]
    target_id: str
    target_label: str = ""
    target_probabilities: Dict[str, float] = Field(default_factory=dict)
    target_confidence: float = 0.0
    action_primitive: str = "click"
    action_confidence: float = 0.0
    click_center_norm: Optional[Tuple[float, float]] = None
    click_pixel: Optional[Tuple[int, int]] = None
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    goal_satisfied: bool = False
    goal_probability: float = 0.0
    blocker_detected: bool = False
    blocker_probability: float = 0.0
    screen_readiness_score: float = 1.0
    screen_readiness_level: str = "ready_interactive"
    assertions: Dict[str, bool] = Field(default_factory=dict)
    assertion_probabilities: Dict[str, float] = Field(default_factory=dict)
    duration_ms: float = 0.0
    backend: str = "local"
    reason: str = ""


class UINavigationTrajectory(BaseModel):
    """Telemetry and step history for a multi-step System-1 UI/Canvas workflow."""

    goal: str
    completed: bool
    steps_taken: int
    total_duration_ms: float
    avg_step_ms: float
    system2_escalations: int = 0
    selected_element_chain: List[str] = Field(default_factory=list)
    action_primitive_chain: List[str] = Field(default_factory=list)
    final_assertions: Dict[str, bool] = Field(default_factory=dict)
    steps: List[UIStepDecision] = Field(default_factory=list)
    final_state: Dict[str, Any] = Field(default_factory=dict)


def extract_ui_candidates(ui_state: Dict[str, Any]) -> List[UICandidate]:
    """Extracts actionable closed-set candidates from DOM snapshots, Autodiscovery JSON, or ArtCraft canvas states."""
    candidates: List[UICandidate] = []
    seen_ids: set[str] = set()

    vp = ui_state.get("viewport") if isinstance(ui_state.get("viewport"), dict) else {}
    vp_w = float(vp.get("width", ui_state.get("viewport_width", 1920)))
    vp_h = float(vp.get("height", ui_state.get("viewport_height", 1080)))

    def _add_item(raw: Any, default_cat: Literal["dom_element", "command_bus", "route", "canvas_tool", "scene_node", "layer", "timeline"], idx: int):
        if isinstance(raw, str):
            cid = raw.strip()
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                candidates.append(UICandidate(id=cid, label=cid, category=default_cat))
            return
        if not isinstance(raw, dict):
            return

        visible = raw.get("visible", not raw.get("hidden", False))
        enabled = raw.get("enabled", not raw.get("disabled", False))
        if visible is False or enabled is False:
            return

        cid = str(
            raw.get("id")
            or raw.get("test_id")
            or raw.get("data_testid")
            or raw.get("command_id")
            or raw.get("tool_id")
            or raw.get("node_id")
            or raw.get("layer_id")
            or raw.get("route")
            or raw.get("name")
            or f"{default_cat}_{idx}"
        ).strip()
        if not cid or cid in seen_ids:
            return
        seen_ids.add(cid)

        label = str(
            raw.get("label")
            or raw.get("text")
            or raw.get("title")
            or raw.get("aria_label")
            or raw.get("name")
            or cid
        ).strip()
        role = str(raw.get("role") or raw.get("type") or raw.get("tool_type") or "button").strip()
        desc_parts = [
            str(raw.get(f, "")).strip()
            for f in ("description", "placeholder", "hint", "shortcut", "target_view", "capabilities", "effect")
            if raw.get(f)
        ]
        action_hint = str(raw.get("action") or raw.get("action_hint") or "").strip() or None

        from core.decision_model import LocalSystemOneBackend
        box = LocalSystemOneBackend._normalize_item_box(raw, vp_w, vp_h)

        candidates.append(
            UICandidate(
                id=cid,
                label=label,
                category=raw.get("category") or default_cat,
                role=role,
                description=" ".join(desc_parts),
                action_hint=action_hint,
                enabled=True,
                visible=True,
                box=box,
                metadata=raw,
            )
        )

    # 1. Standard DOM & Autodiscovery elements
    for key in ("elements", "interactive_elements", "controls", "buttons", "inputs", "tabs", "candidates"):
        items = ui_state.get(key)
        if isinstance(items, list):
            for i, item in enumerate(items):
                _add_item(item, "dom_element", i)

    # 2. Valstorm Command Bus commands
    for key in ("commands", "command_bus", "palette_commands"):
        items = ui_state.get(key)
        if isinstance(items, list):
            for i, item in enumerate(items):
                _add_item(item, "command_bus", i)

    # 3. Navigation routes
    for key in ("routes", "nav_links"):
        items = ui_state.get(key)
        if isinstance(items, list):
            for i, item in enumerate(items):
                _add_item(item, "route", i)

    # 4. ArtCraft / 2D & 3D Creative Canvas tools, scene actors, layers, cameras, timeline
    for key in ("tools", "canvas_tools", "crafting_tools"):
        items = ui_state.get(key)
        if isinstance(items, list):
            for i, item in enumerate(items):
                _add_item(item, "canvas_tool", i)

    for key in ("actors", "nodes", "scene_nodes", "meshes", "cameras", "props"):
        items = ui_state.get(key)
        if isinstance(items, list):
            for i, item in enumerate(items):
                _add_item(item, "scene_node", i)

    for key in ("layers", "compositing_layers"):
        items = ui_state.get(key)
        if isinstance(items, list):
            for i, item in enumerate(items):
                _add_item(item, "layer", i)

    for key in ("timeline", "timeline_controls", "shots"):
        items = ui_state.get(key)
        if isinstance(items, list):
            for i, item in enumerate(items):
                _add_item(item, "timeline", i)

    return candidates


class SystemOneUINavigator:
    """High-speed System-1 Navigator for DOM UIs, Command Buses, and 2D/3D Crafting Canvases."""

    def __init__(
        self,
        client: Optional[SystemOneClient] = None,
        confidence_threshold: float = 0.55,
    ):
        self.client = client or SystemOneClient(backend="auto")
        self.confidence_threshold = confidence_threshold

    def decide_step(
        self,
        goal: str,
        ui_state: Dict[str, Any],
        assertions: Optional[Dict[str, str]] = None,
        visited_targets: Optional[List[str]] = None,
        subgoal_focus: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> UIStepDecision:
        """Evaluates the current UI/canvas state in a single parallel System-1 pass."""
        t0 = time.perf_counter()
        thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        visited = set(visited_targets or [])

        candidates = extract_ui_candidates(ui_state)
        if not candidates:
            dur = round((time.perf_counter() - t0) * 1000.0, 3)
            return UIStepDecision(
                route_status="escalate_system2",
                target_id="",
                reason="No actionable UI or canvas candidates found in state.",
                duration_ms=dur,
            )

        # Build criteria map for unvisited (or all if all visited) candidates
        unvisited_candidates = [c for c in candidates if c.id not in visited]
        active_candidates = unvisited_candidates if unvisited_candidates else candidates
        candidate_criteria = {c.id: c.to_criteria_string() for c in active_candidates}
        cand_by_id = {c.id: c for c in candidates}

        effective_goal = subgoal_focus or goal

        # Check if there is an active blocking modal or error toast with a dismiss/close button
        has_active_blocker = bool(
            ui_state.get("active_error")
            or ui_state.get("error_toast")
            or ui_state.get("blocking_modal")
        )

        # Construct parallel System-1 question batch
        questions: Dict[str, QuestionSpec] = {
            "target_element": ChoiceQuestion(
                instructions=f"Select the single best UI element, command, or canvas tool to interact with next to achieve: {effective_goal}",
                criteria=candidate_criteria,
            ),
            "action_primitive": ChoiceQuestion(
                instructions=f"Select the interaction primitive required for: {effective_goal}",
                criteria=UI_ACTION_PRIMITIVES,
            ),
            "blocker_detected": NoulQuestion(
                instructions="Is there an active unhandled error toast, crash alert, or blocking modal obstructing the screen?",
            ),
            "screen_readiness": ScoreQuestion(
                instructions="Rate the interactive readiness of the current UI/canvas state",
                criteria={
                    "broken_error": "Unhandled exception, crash screen, or fatal error toast is blocking the view",
                    "loading_skeleton": "Page or canvas is still loading with skeleton placeholders or spinner",
                    "modal_interstitial": "A confirmation dialog or modal overlay is open",
                    "ready_interactive": "Screen and interactive controls/canvas tools are loaded, visible, and ready",
                },
            ),
            "target_box": BoxQuestion(
                instructions=f"Locate the bounding box for the control matching: {effective_goal}",
            ),
        }

        if assertions:
            for a_key, a_stmt in assertions.items():
                questions[f"assert__{a_key}"] = NoulQuestion(instructions=a_stmt)

        # Prepare compact evaluation state
        eval_state = {
            **ui_state,
            "goal": effective_goal,
            "full_goal": goal,
        }

        resp: SystemOneResponse = self.client.evaluate(state=eval_state, questions=questions)

        target_ans = resp.get_choice("target_element")
        action_ans = resp.get_choice("action_primitive")
        blocker_ans = resp.get_noul("blocker_detected")
        readiness_ans = resp.get_score("screen_readiness")
        box_ans = resp.get_box("target_box")

        chosen_id = target_ans.choice
        chosen_cand = cand_by_id.get(chosen_id)

        # If the chosen candidate has an explicit action_hint or bounding box, refine with zero overhead
        chosen_action = chosen_cand.action_hint if (chosen_cand and chosen_cand.action_hint in UI_ACTION_PRIMITIVES) else action_ans.choice
        if chosen_cand and chosen_cand.box:
            b = chosen_cand.box
            box_ans = BoxAnswer(left=b[0], top=b[1], right=b[2], bottom=b[3], confidence=max(box_ans.confidence, target_ans.confidence), matched_id=chosen_id)

        # Parse E2E assertions evaluated in the same parallel pass
        assertion_bools: Dict[str, bool] = {}
        assertion_probs: Dict[str, float] = {}
        if assertions:
            for a_key in assertions:
                n_ans = resp.get_noul(f"assert__{a_key}")
                assertion_bools[a_key] = n_ans.is_true
                assertion_probs[a_key] = n_ans.noul

        # Check if goal is already explicitly marked complete in state
        goal_done = bool(ui_state.get("goal_completed") is True or ui_state.get("workflow_complete") is True)
        goal_prob = 0.96 if goal_done else 0.10

        # Determine Confidence-Gated Route Status
        if goal_done:
            route_status: Literal["execute_action", "handle_blocker", "goal_complete", "escalate_system2"] = "goal_complete"
            reason = "Goal completion criteria verified on UI/canvas state."
        elif has_active_blocker or blocker_ans.noul >= 0.75:
            # Look for a dismiss/close candidate in the closed set
            dismiss_cand = next(
                (c for c in active_candidates if any(w in f"{c.id} {c.label} {c.description}".lower() for w in ("dismiss", "close", "acknowledge", "clear_error", "cancel"))),
                None,
            )
            if dismiss_cand:
                chosen_id = dismiss_cand.id
                chosen_cand = dismiss_cand
                chosen_action = "dismiss_modal"
                route_status = "handle_blocker"
                reason = f"Blocker detected (p={blocker_ans.noul:.2f}); routing to '{chosen_id}' to dismiss blocker."
            else:
                route_status = "escalate_system2"
                reason = f"Blocker detected (p={blocker_ans.noul:.2f}) with no deterministic dismiss control; escalating to System-2."
        elif target_ans.confidence < thresh:
            route_status = "escalate_system2"
            reason = (
                f"Target choice '{chosen_id}' confidence ({target_ans.confidence:.2f}) is below "
                f"threshold ({thresh:.2f}); escalating ambiguous UI state to System-2."
            )
        else:
            route_status = "execute_action"
            reason = (
                f"Selected '{chosen_id}' ({chosen_action}) with confidence {target_ans.confidence:.2f} >= {thresh:.2f}."
            )

        vp = ui_state.get("viewport") if isinstance(ui_state.get("viewport"), dict) else {}
        vp_w = int(vp.get("width", ui_state.get("viewport_width", 1920)))
        vp_h = int(vp.get("height", ui_state.get("viewport_height", 1080)))

        dur_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return UIStepDecision(
            route_status=route_status,
            target_id=chosen_id,
            target_label=chosen_cand.label if chosen_cand else chosen_id,
            target_probabilities=target_ans.probabilities,
            target_confidence=target_ans.confidence,
            action_primitive=chosen_action,
            action_confidence=action_ans.confidence,
            click_center_norm=box_ans.center,
            click_pixel=box_ans.pixel_point(vp_w, vp_h),
            bounding_box=(box_ans.left, box_ans.top, box_ans.right, box_ans.bottom),
            goal_satisfied=goal_done,
            goal_probability=goal_prob,
            blocker_detected=has_active_blocker or blocker_ans.is_true,
            blocker_probability=blocker_ans.noul,
            screen_readiness_score=readiness_ans.score,
            screen_readiness_level=readiness_ans.level,
            assertions=assertion_bools,
            assertion_probabilities=assertion_probs,
            duration_ms=dur_ms,
            backend=resp.backend,
            reason=reason,
        )

    @staticmethod
    def decompose_goal_steps(goal: str) -> List[str]:
        """Splits a multi-step natural language UI or crafting goal into ordered atomic step intents."""
        import re
        # Check numbered steps like "1. ... 2. ... 3. ..." or "Step 1: ... -> Step 2: ..."
        numbered = re.split(r"(?:^|\s+)(?:\d+[\.\)]\s+|step\s+\d+[:\.\-]\s*)", goal, flags=re.IGNORECASE)
        cleaned = [s.strip().rstrip(".,;") for s in numbered if s and s.strip()]
        if len(cleaned) >= 2:
            return cleaned

        # Check arrow or "then" chains: "First ..., then ..., finally ..."
        arrow_parts = re.split(r"\s*(?:->|→|;\s*then\s+|\bthen\b|\band\s+then\b|\bfinally\b)\s*", goal, flags=re.IGNORECASE)
        cleaned_arrows = [s.strip().rstrip(".,;") for s in arrow_parts if s and len(s.strip()) > 3]
        if len(cleaned_arrows) >= 2:
            return cleaned_arrows

        return [goal.strip()]

    @staticmethod
    def apply_simulated_transition(
        ui_state: Dict[str, Any],
        decision: UIStepDecision,
    ) -> Dict[str, Any]:
        """Applies a state transition for a selected UI/canvas element (supports declarative `transitions` or default state updates)."""
        next_state = copy.deepcopy(ui_state)
        tid = decision.target_id

        # 1. If the state defines explicit `transitions` for `target_id`, merge them
        transitions = next_state.get("transitions")
        if isinstance(transitions, dict) and tid in transitions:
            patch = transitions[tid]
            if isinstance(patch, dict):
                for k, v in patch.items():
                    if k == "append_history" and isinstance(v, str):
                        hist = list(next_state.get("interaction_history") or [])
                        hist.append(v)
                        next_state["interaction_history"] = hist
                    else:
                        next_state[k] = copy.deepcopy(v)

        # 2. Record executed step in interaction_history & active state markers
        hist = list(next_state.get("interaction_history") or [])
        hist.append(f"{decision.action_primitive}:{tid}")
        next_state["interaction_history"] = hist
        next_state["last_target_id"] = tid
        next_state["last_action_primitive"] = decision.action_primitive

        # If dismissing a blocker, clear active_error / blocking_modal
        if decision.route_status == "handle_blocker" or decision.action_primitive == "dismiss_modal":
            next_state.pop("active_error", None)
            next_state.pop("error_toast", None)
            next_state.pop("blocking_modal", None)

        return next_state

    def run_navigation_workflow(
        self,
        goal: str,
        initial_state: Dict[str, Any],
        assertions: Optional[Dict[str, str]] = None,
        step_executor: Optional[Callable[[Dict[str, Any], UIStepDecision], Dict[str, Any]]] = None,
        max_steps: int = 10,
        confidence_threshold: Optional[float] = None,
    ) -> UINavigationTrajectory:
        """Executes a complete multi-step System-1 UI or ArtCraft canvas navigation loop."""
        t0 = time.perf_counter()
        subgoals = self.decompose_goal_steps(goal)
        state = copy.deepcopy(initial_state)
        visited: List[str] = []
        steps: List[UIStepDecision] = []
        selected_ids: List[str] = []
        action_primitives: List[str] = []
        escalations = 0

        total_planned = max(1, min(max_steps, len(subgoals) if len(subgoals) > 1 else max_steps))

        for step_idx in range(total_planned):
            # If state already signals workflow completion, evaluate final assertions and exit
            if state.get("goal_completed") is True or state.get("workflow_complete") is True:
                break

            subgoal = subgoals[step_idx] if step_idx < len(subgoals) else goal
            decision = self.decide_step(
                goal=goal,
                ui_state=state,
                assertions=assertions,
                visited_targets=visited,
                subgoal_focus=subgoal,
                confidence_threshold=confidence_threshold,
            )
            steps.append(decision)

            if decision.route_status == "goal_complete":
                break

            if decision.route_status == "escalate_system2":
                escalations += 1
                break

            if decision.target_id:
                visited.append(decision.target_id)
                selected_ids.append(decision.target_id)
                action_primitives.append(decision.action_primitive)

            # Advance UI/canvas state via custom step_executor or declarative transition
            if step_executor is not None:
                state = step_executor(state, decision)
            else:
                state = self.apply_simulated_transition(state, decision)

            # If we had a fixed list of decomposed subgoals and executed all of them, mark complete
            if len(subgoals) >= 1 and (step_idx + 1) == len(subgoals) and not state.get("active_error"):
                state["goal_completed"] = True

        # Final assertion verification pass over the post-navigation state
        final_asserts: Dict[str, bool] = {}
        if assertions:
            assert_questions: Dict[str, QuestionSpec] = {
                k: NoulQuestion(instructions=stmt) for k, stmt in assertions.items()
            }
            assert_resp = self.client.evaluate(state=state, questions=assert_questions)
            for k in assertions:
                final_asserts[k] = assert_resp.get_noul(k).is_true
            if steps:
                steps[-1].assertions = final_asserts

        total_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        n_steps = len(steps)
        completed = bool(
            state.get("goal_completed") is True
            or state.get("workflow_complete") is True
            or (steps and steps[-1].route_status == "goal_complete")
        )
        return UINavigationTrajectory(
            goal=goal,
            completed=completed,
            steps_taken=n_steps,
            total_duration_ms=total_ms,
            avg_step_ms=round(total_ms / max(1, n_steps), 3),
            system2_escalations=escalations,
            selected_element_chain=selected_ids,
            action_primitive_chain=action_primitives,
            final_assertions=final_asserts,
            steps=steps,
            final_state=state,
        )
