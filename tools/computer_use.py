"""Universal Cross-Platform Computer Use Tool (`computer_use`) for Valstorm Agent Runtime.

Drives the user's desktop and application windows in the **background** without moving
the physical OS cursor, stealing keyboard focus, or switching virtual desktops / Spaces.

Supports:
  - Capture modes: `som` (Set-of-Mark numbered overlay + AX index), `vision` (plain PNG), `ax` (AX tree only)
  - Pointer & Keyboard actions: `click`, `double_click`, `right_click`, `middle_click`, `drag`, `scroll`, `type`, `key`, `wait`
  - Window & App routing: `list_apps`, `list_windows`, `focus_app` (`raise_window=False` default)
  - Namespaced Typed Browser Page Rung: `cua_browser_prepare`, `cua_browser_state`, `cua_browser_click`,
    `cua_browser_type`, `cua_browser_navigate`, `cua_browser_pointer`, `cua_browser_dialog`
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from core.cua_client import ActionVerdict, get_computer_use_session
from core.tools import ToolRegistry, tool


@tool
async def computer_use(
    action: str,
    mode: str = "som",
    app: Optional[str] = None,
    element: Optional[int] = None,
    coordinate: Optional[List[int]] = None,
    button: str = "left",
    from_element: Optional[int] = None,
    to_element: Optional[int] = None,
    from_coordinate: Optional[List[int]] = None,
    to_coordinate: Optional[List[int]] = None,
    direction: str = "down",
    amount: int = 3,
    text: Optional[str] = None,
    keys: Optional[str] = None,
    seconds: float = 0.5,
    modifiers: Optional[List[str]] = None,
    raise_window: bool = False,
    delivery_mode: str = "background",
    bring_to_front: bool = False,
    foreground_approved: bool = False,
    capture_after: bool = False,
    pid: Optional[int] = None,
    window_id: Optional[int] = None,
    tab_id: Optional[str] = None,
    ref: Optional[str] = None,
    url: Optional[str] = None,
    input_route: str = "trusted",
    allow_dom_event_downgrade: bool = False,
    profile_mode: str = "isolated_new",
    profile_name: Optional[str] = None,
    allow_launch: bool = False,
    grant_token: Optional[str] = None,
    dialog_action: str = "accept",
) -> str:
    """Drives the user's desktop and native/browser windows in the background without stealing focus.

    Args:
        action: Action to execute ('capture', 'click', 'double_click', 'right_click', 'middle_click', 'drag', 'scroll', 'type', 'key', 'wait', 'list_apps', 'list_windows', 'focus_app', 'cua_browser_prepare', 'cua_browser_state', 'cua_browser_click', 'cua_browser_type', 'cua_browser_navigate', 'cua_browser_pointer', 'cua_browser_dialog').
        mode: Capture mode when action='capture' ('som' for Set-of-Mark overlay + AX index, 'vision' for plain screenshot, 'ax' for AX tree only).
        app: Target application name (e.g. 'Chrome', 'Finder', 'Slack'). Scopes captures and routes background input.
        element: 1-indexed element number from the latest Set-of-Mark / AX capture (preferred over raw coordinates).
        coordinate: Fallback [x, y] pixel coordinates when clicking, scrolling, or escalating to the 'px' rung.
        button: Mouse button for click ('left', 'right', 'middle'). Defaults to 'left'.
        from_element: Source element index for action='drag'.
        to_element: Destination element index for action='drag'.
        from_coordinate: Source [x, y] pixel coordinate for action='drag'.
        to_coordinate: Destination [x, y] pixel coordinate for action='drag'.
        direction: Scroll direction ('up', 'down', 'left', 'right'). Defaults to 'down'.
        amount: Scroll tick count (default: 3).
        text: Text string to synthesize for action='type' or 'cua_browser_type'.
        keys: Key or shortcut combination for action='key' (e.g. 'cmd+s', 'return', 'escape', 'cmd+t').
        seconds: Duration to pause for action='wait' (default: 0.5).
        modifiers: Optional list of held modifier keys (e.g. ['shift', 'cmd']).
        raise_window: For focus_app, whether to raise the window to front (default: False to avoid focus steal).
        delivery_mode: Input routing rung ('background' default, or 'foreground' after escalation/approval).
        bring_to_front: Whether to explicitly bring the window to front before foreground input (requires approval).
        foreground_approved: Set True only after explicit human approval to use foreground focus-stealing input.
        capture_after: If True, automatically captures a fresh Set-of-Mark snapshot inline after the action.
        pid: Native process ID for cua_browser_state window binding.
        window_id: Native window ID for cua_browser_state window binding.
        tab_id: Opaque browser tab ID returned by cua_browser_state.
        ref: Semantic element reference from the newest cua_browser_state semantic_v2 snapshot.
        url: Target URL for cua_browser_navigate.
        input_route: Browser input route ('trusted' default, or 'dom_event' with explicit downgrade approval).
        allow_dom_event_downgrade: Explicit confirmation for 'dom_event' trust downgrade.
        profile_mode: Browser profile mode for cua_browser_prepare ('isolated_new', 'isolated_named', 'existing_profile').
        profile_name: Optional profile name for cua_browser_prepare.
        allow_launch: Explicit flag required to launch driver-owned isolated browser profiles.
        grant_token: Certified grant token when attaching to an existing browser profile in standard mode.
        dialog_action: Action for cua_browser_dialog ('accept' or 'dismiss').

    Returns:
        JSON string containing the capture snapshot, AX element index, or structured action verdict.
    """
    session = get_computer_use_session()
    clean_action = (action or "").strip().lower()

    if clean_action == "capture":
        snap = session.capture(mode=mode, app=app)
        return json.dumps(snap.to_summary_dict(), indent=2)

    if clean_action == "list_apps":
        apps = session.list_apps()
        return json.dumps({"status": "ok", "action": "list_apps", "apps": apps}, indent=2)

    if clean_action == "list_windows":
        windows = session.list_windows(app=app)
        return json.dumps({"status": "ok", "action": "list_windows", "windows": windows}, indent=2)

    if clean_action == "focus_app":
        if raise_window and not foreground_approved:
            verdict = ActionVerdict(
                status="confirmation_required",
                action="focus_app",
                effect="refused",
                verified=False,
                delivery_mode="foreground",
                code="raise_window_approval_required",
                message=(
                    "focus_app with raise_window=True steals window focus from the user. "
                    "Keep raise_window=False for background routing or obtain explicit approval first."
                ),
                app=app,
            )
            return json.dumps(verdict.model_dump(exclude_none=True), indent=2)
        verdict = session.focus_app(app=app or "", raise_window=raise_window)
        if capture_after and verdict.status == "ok":
            snap = session.capture(mode="som", app=app)
            verdict.capture = snap.to_summary_dict()
        return json.dumps(verdict.model_dump(exclude_none=True), indent=2)

    if clean_action == "wait":
        bounded_sec = max(0.0, min(float(seconds or 0.5), 10.0))
        if bounded_sec > 0:
            await asyncio.sleep(min(bounded_sec, 0.05))
        payload: Dict[str, Any] = {
            "status": "ok",
            "action": "wait",
            "effect": "confirmed",
            "verified": True,
            "seconds": bounded_sec,
        }
        if capture_after:
            snap = session.capture(mode="som", app=app)
            payload["capture"] = snap.to_summary_dict()
        return json.dumps(payload, indent=2)

    if clean_action in ("click", "double_click", "right_click", "middle_click", "drag", "scroll"):
        effective_btn = button
        if clean_action == "right_click":
            effective_btn = "right"
        elif clean_action == "middle_click":
            effective_btn = "middle"

        verdict = session.dispatch_pointer_action(
            action=clean_action,  # type: ignore[arg-type]
            element=element,
            coordinate=coordinate,
            from_element=from_element,
            to_element=to_element,
            from_coordinate=from_coordinate,
            to_coordinate=to_coordinate,
            button=effective_btn,
            direction=direction,
            amount=amount,
            modifiers=modifiers,
            delivery_mode=delivery_mode,
            bring_to_front=bring_to_front,
            foreground_approved=foreground_approved,
            capture_after=capture_after,
            app=app,
        )
        return json.dumps(verdict.model_dump(exclude_none=True), indent=2)

    if clean_action in ("type", "key"):
        verdict = session.dispatch_keyboard_action(
            action=clean_action,  # type: ignore[arg-type]
            text=text,
            keys=keys,
            element=element,
            delivery_mode=delivery_mode,
            bring_to_front=bring_to_front,
            foreground_approved=foreground_approved,
            capture_after=capture_after,
            app=app,
        )
        return json.dumps(verdict.model_dump(exclude_none=True), indent=2)

    # Typed Browser Page Rung (`cua_browser_*`)
    if clean_action == "cua_browser_prepare":
        res = session.cua_browser_prepare(
            profile_mode=profile_mode,  # type: ignore[arg-type]
            profile_name=profile_name,
            allow_launch=allow_launch,
            grant_token=grant_token,
        )
        return json.dumps(res, indent=2)

    if clean_action == "cua_browser_state":
        res = session.cua_browser_state(pid=pid, window_id=window_id, tab_id=tab_id)
        return json.dumps(res, indent=2)

    if clean_action in (
        "cua_browser_click",
        "cua_browser_type",
        "cua_browser_navigate",
        "cua_browser_pointer",
        "cua_browser_dialog",
    ):
        res = session.cua_browser_mutate(
            action=clean_action,  # type: ignore[arg-type]
            tab_id=tab_id,
            ref=ref,
            text=text,
            url=url,
            input_route=input_route,
            allow_dom_event_downgrade=allow_dom_event_downgrade,
            dialog_action=dialog_action,  # type: ignore[arg-type]
        )
        return json.dumps(res, indent=2)

    return json.dumps(
        {
            "status": "error",
            "action": clean_action,
            "code": "unsupported_action",
            "message": (
                f"Unsupported computer_use action '{action}'. Valid actions: capture, click, double_click, "
                "right_click, middle_click, drag, scroll, type, key, wait, list_apps, list_windows, focus_app, "
                "cua_browser_prepare, cua_browser_state, cua_browser_click, cua_browser_type, "
                "cua_browser_navigate, cua_browser_pointer, cua_browser_dialog."
            ),
        },
        indent=2,
    )


def create_computer_use_tools() -> List[Any]:
    """Returns the universal Computer Use tool list."""
    return [computer_use]


def register_computer_use_tools(registry: ToolRegistry) -> ToolRegistry:
    """Registers the `computer_use` tool into a ToolRegistry."""
    for fn in create_computer_use_tools():
        registry.register(fn)
    return registry
