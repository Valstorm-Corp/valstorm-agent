"""Unit, integration, and benchmark tests for the System-1 Decision Model & UI/Canvas Navigator."""

from __future__ import annotations

import json
from pathlib import Path
from httpx import ASGITransport, AsyncClient
import pytest

from benchmarks.runner import get_ablation_configs, load_tasks, run_benchmark_matrix
from core.decision_model import (
    BoxQuestion,
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneClient,
    TypeSafeJevBackend,
    route_with_confidence,
)
from core.sandbox import HostSandbox, set_current_sandbox
from core.tools import ToolRegistry
from core.ui_navigator import SystemOneUINavigator, extract_ui_candidates
from server import app
from tools.ui_navigator_tools import register_ui_navigator_tools, system_one_decide, ui_navigate


class TestSystemOneDecisionModel:
    """Tests for core System-1 primitives (Choice, Score, Noul, Box) and calibration."""

    def test_parallel_mixed_primitives_single_pass(self):
        client = SystemOneClient(backend="local")
        state = {
            "current_view": "/view/vaults",
            "goal": "Share the Q4 financial report in Cloud Vaults",
            "error_toast": False,
            "elements": [
                {"id": "btn_share_doc", "label": "Share Document", "box": [0.75, 0.10, 0.85, 0.16]},
                {"id": "btn_archive_doc", "label": "Archive Document", "box": [0.86, 0.10, 0.95, 0.16]},
            ],
        }
        questions = {
            "next_control": ChoiceQuestion(
                instructions="Which button shares the document?",
                criteria={
                    "btn_share_doc": "Share Document button to open sharing permissions",
                    "btn_archive_doc": "Archive Document button to move file to archive",
                },
            ),
            "readiness": ScoreQuestion(
                instructions="Is the Cloud Vaults view ready for sharing?",
                criteria={
                    "error": "Fatal error screen",
                    "loading": "Loading skeleton",
                    "ready": "Cloud Vaults view active with Share Document button visible",
                },
            ),
            "has_error": NoulQuestion(
                instructions="Is there an unhandled error toast on screen?",
            ),
            "vaults_open": NoulQuestion(
                instructions="Is current_view set to /view/vaults?",
            ),
            "share_box": BoxQuestion(
                instructions="Locate the Share Document button",
            ),
        }

        resp = client.evaluate(state=state, questions=questions)
        assert resp.backend == "local"
        assert resp.output_tokens == 0
        assert resp.cost_usd == 0.0
        assert resp.duration_ms < 250.0

        # 1. Choice verification & probability sum == 1.0
        c_ans = resp.get_choice("next_control")
        assert c_ans.choice == "btn_share_doc"
        assert abs(sum(c_ans.probabilities.values()) - 1.0) < 1e-4
        assert c_ans.probabilities["btn_share_doc"] > c_ans.probabilities["btn_archive_doc"]
        assert c_ans.confidence > 0.50

        # 2. Score verification
        s_ans = resp.get_score("readiness")
        assert s_ans.level == "ready"
        assert s_ans.score > 0.60

        # 3. Noul boolean truth & negation verification
        err_ans = resp.get_noul("has_error")
        assert err_ans.is_true is False
        assert err_ans.noul < 0.25

        vault_ans = resp.get_noul("vaults_open")
        assert vault_ans.is_true is True
        assert vault_ans.noul > 0.60

        # 4. Box coordinates verification
        box_ans = resp.get_box("share_box")
        assert box_ans.matched_id == "btn_share_doc"
        assert box_ans.center == (0.8, 0.13)
        assert box_ans.pixel_point(1920, 1080) == (1536, 140)

    def test_high_cardinality_two_stage_beam_evaluation(self):
        """Verify >255 options automatically triggers 2-stage beam ranking like Jev's Wikiracing."""
        client = SystemOneClient(backend="local")
        criteria = {f"link_{i}": f"Unrelated navigation page topic {i}" for i in range(300)}
        criteria["link_artcraft_3d_camera"] = "Configure 3D viewport camera orbit angle and lens FOV in ArtCraft"

        resp = client.evaluate(
            state={"goal": "Adjust the 3D viewport camera orbit angle and lens FOV"},
            questions={
                "best_link": ChoiceQuestion(
                    instructions="Pick the link that configures the 3D camera orbit and lens FOV",
                    criteria=criteria,
                )
            },
        )
        ans = resp.get_choice("best_link")
        assert ans.stages_evaluated == 2
        assert ans.choice == "link_artcraft_3d_camera"
        assert len(ans.probabilities) == 301
        assert abs(sum(ans.probabilities.values()) - 1.0) < 1e-4

    def test_confidence_gated_routing(self):
        client = SystemOneClient(backend="local")
        resp = client.evaluate(
            state={"goal": "Click the Render Video Scene button"},
            questions={
                "clear_q": ChoiceQuestion(
                    instructions="Select the Render Video Scene button",
                    criteria={
                        "btn_render": "Render Video Scene button",
                        "btn_logout": "Sign out of account",
                    },
                ),
                "ambiguous_q": ChoiceQuestion(
                    instructions="Select the next action",
                    criteria={
                        "opt_a": "Generic control alpha",
                        "opt_b": "Generic control beta",
                    },
                ),
            },
        )
        clear_route = route_with_confidence(resp.get_choice("clear_q"), threshold=0.55)
        assert clear_route.action == "act"
        assert clear_route.selected_value == "btn_render"

        ambig_route = route_with_confidence(resp.get_choice("ambiguous_q"), threshold=0.90)
        assert ambig_route.action == "escalate_system2"

        fallback_route = route_with_confidence(
            resp.get_choice("ambiguous_q"),
            threshold=0.90,
            fallback_value="safe_default_btn",
        )
        assert fallback_route.action == "fallback"
        assert fallback_route.selected_value == "safe_default_btn"

    def test_typesafe_jev_backend_payload_and_box_header(self):
        jev = TypeSafeJevBackend(api_key="ts_test_key", model="jev-latest")
        payload, has_box = jev._build_payload(
            state={"screen": "main"},
            questions={
                "q_choice": ChoiceQuestion(instructions="Pick one", criteria={"a": "Option A", "b": "Option B"}),
                "q_box": BoxQuestion(instructions="Find mug"),
            },
        )
        assert has_box is True
        assert payload["model"] == "jev-preview"
        assert payload["questions"]["q_choice"]["type"] == "choice"
        assert payload["questions"]["q_box"]["type"] == "box"


class TestSystemOneUINavigator:
    """Tests for DOM, Command Bus, and ArtCraft 2D/3D Canvas navigation."""

    def test_extract_candidates_filters_disabled_and_hidden(self):
        ui_state = {
            "elements": [
                {"id": "btn_visible", "label": "Save Changes", "visible": True, "enabled": True},
                {"id": "btn_hidden", "label": "Hidden Secret", "visible": False},
                {"id": "btn_disabled", "label": "Disabled Submit", "disabled": True},
            ],
            "commands": [
                {"command_id": "cmd_open_vaults", "label": "Open Cloud Vaults"},
            ],
            "tools": [
                {"tool_id": "tool_3d_mesh", "label": "Image to 3D Mesh"},
            ],
        }
        cands = extract_ui_candidates(ui_state)
        ids = [c.id for c in cands]
        assert ids == ["btn_visible", "cmd_open_vaults", "tool_3d_mesh"]

    def test_blocker_detection_and_modal_dismissal_routing(self):
        navigator = SystemOneUINavigator(confidence_threshold=0.50)
        ui_state = {
            "active_error": "Rate limit warning modal is blocking the workspace",
            "elements": [
                {"id": "btn_save_record", "label": "Save CRM Record"},
                {"id": "btn_dismiss_error_modal", "label": "Dismiss & Close Warning Modal"},
            ],
        }
        step = navigator.decide_step(
            goal="Click Save CRM Record",
            ui_state=ui_state,
        )
        assert step.blocker_detected is True
        assert step.route_status == "handle_blocker"
        assert step.target_id == "btn_dismiss_error_modal"
        assert step.action_primitive == "dismiss_modal"

        # Applying transition clears the blocker so the next step can click Save CRM Record
        next_state = navigator.apply_simulated_transition(ui_state, step)
        assert "active_error" not in next_state
        step2 = navigator.decide_step(
            goal="Click Save CRM Record",
            ui_state=next_state,
            visited_targets=[step.target_id],
        )
        assert step2.route_status == "execute_action"
        assert step2.target_id == "btn_save_record"

    def test_valstorm_desktop_multi_step_workflow(self):
        ui_tasks = load_tasks(suite="ui", task_id="UI-01")
        assert len(ui_tasks) == 1
        task = ui_tasks[0]

        navigator = SystemOneUINavigator(confidence_threshold=0.50)
        traj = navigator.run_navigation_workflow(
            goal=task["goal"],
            initial_state=task["ui_state"],
            assertions=task["assertions"],
        )
        assert traj.completed is True
        assert traj.selected_element_chain == [
            "nav_tab_cloud_vaults",
            "vault_doc_q4_strategy",
            "btn_share_vault_doc",
        ]
        assert traj.final_assertions["no_error_toast"] is False
        assert traj.final_assertions["vault_view_active"] is True
        assert traj.final_assertions["share_modal_ready"] is True

    def test_artcraft_3d_scene_and_video_crafting_workflow(self):
        ui_tasks = load_tasks(suite="ui", task_id="UI-02")
        assert len(ui_tasks) == 1
        task = ui_tasks[0]

        navigator = SystemOneUINavigator(confidence_threshold=0.50)
        traj = navigator.run_navigation_workflow(
            goal=task["goal"],
            initial_state=task["ui_state"],
            assertions=task["assertions"],
        )
        assert traj.completed is True
        assert traj.selected_element_chain == [
            "tool_img_to_3d_mesh",
            "actor_hero_pose_rig",
            "cam_main_shot_rig",
            "btn_render_video_scene",
        ]
        assert traj.action_primitive_chain == [
            "convert_3d_mesh",
            "pose_actor",
            "camera_frame",
            "render_scene",
        ]
        assert traj.final_assertions == {
            "mesh_converted": True,
            "actor_posed": True,
            "video_rendered": True,
        }


class TestToolsServerAndBenchmarkIntegration:
    """Tests for ToolRegistry execution, FastAPI daemon routes, and benchmark matrix ablations."""

    @pytest.mark.asyncio
    async def test_ui_navigator_tools_in_sandbox(self, tmp_path: Path):
        sb = HostSandbox(base_dir=str(tmp_path))
        tok = set_current_sandbox(sb)
        try:
            reg = ToolRegistry()
            register_ui_navigator_tools(reg)
            assert "system_one_decide" in reg.list_tools()
            assert "ui_navigate" in reg.list_tools()

            ui_tasks = load_tasks(suite="ui", task_id="UI-02")
            task = ui_tasks[0]

            res = await reg.execute_async(
                "ui_navigate",
                {
                    "goal": task["goal"],
                    "ui_state": json.dumps(task["ui_state"]),
                    "assertions": json.dumps(task["assertions"]),
                    "mode": "workflow",
                    "output_path": "artcraft_out.json",
                },
            )
            assert res.is_error is False
            out_file = tmp_path / "artcraft_out.json"
            assert out_file.is_file()
            saved = json.loads(out_file.read_text(encoding="utf-8"))
            assert saved["completed"] is True
            assert saved["selected_element_chain"] == [
                "tool_img_to_3d_mesh",
                "actor_hero_pose_rig",
                "cam_main_shot_rig",
                "btn_render_video_scene",
            ]
        finally:
            from core.sandbox import _current_sandbox
            _current_sandbox.reset(tok)

    @pytest.mark.asyncio
    async def test_local_daemon_system_one_endpoints(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            eval_resp = await ac.post(
                "/v1/system-one/evaluate",
                json={
                    "state": {"goal": "Open Cloud Vaults"},
                    "questions": {
                        "route": {
                            "type": "choice",
                            "instructions": "Select the target view",
                            "criteria": {
                                "vaults": "Open Cloud Vaults workspace",
                                "settings": "Open user profile settings",
                            },
                        }
                    },
                },
            )
            assert eval_resp.status_code == 200
            data = eval_resp.json()
            assert data["answers"]["route"]["choice"] == "vaults"

            ui_tasks = load_tasks(suite="ui", task_id="UI-01")
            nav_resp = await ac.post(
                "/v1/system-one/ui-navigate",
                json={
                    "goal": ui_tasks[0]["goal"],
                    "ui_state": ui_tasks[0]["ui_state"],
                    "assertions": ui_tasks[0]["assertions"],
                    "mode": "workflow",
                },
            )
            assert nav_resp.status_code == 200
            nav_data = nav_resp.json()
            assert nav_data["completed"] is True
            assert nav_data["selected_element_chain"] == [
                "nav_tab_cloud_vaults",
                "vault_doc_q4_strategy",
                "btn_share_vault_doc",
            ]

    @pytest.mark.asyncio
    async def test_benchmark_decision_model_ablation_matrix(self):
        """Verify System-1 UI Navigator vs Full LLM ReAct UI Navigation across all 4 UI/ArtCraft tasks."""
        ui_tasks = load_tasks(suite="ui")
        assert len(ui_tasks) == 4

        configs = get_ablation_configs("decision_model")
        assert [c.config_id for c in configs] == ["system_one_ui_nav", "llm_react_ui_nav"]

        report = await run_benchmark_matrix(tasks=ui_tasks, configs=configs, mode="mock", trials=1)
        summaries = {s["config_id"]: s for s in report["config_summaries"]}

        s1 = summaries["system_one_ui_nav"]
        s2 = summaries["llm_react_ui_nav"]

        # Both achieve 100% pass rate, but System-1 takes fewer turns, lower latency, and far fewer tokens/cost
        assert s1["pass_rate"] == 1.0
        assert s2["pass_rate"] == 1.0
        assert s1["avg_turns"] < s2["avg_turns"]
        assert s1["avg_duration_sec"] < s2["avg_duration_sec"]
        assert s1["avg_tokens"] < s2["avg_tokens"]
        assert s1["total_cost_usd"] < s2["total_cost_usd"]

    @pytest.mark.asyncio
    async def test_adaptive_thinking_router_and_ablation(self):
        """Verify SystemOneThinkingRouter dynamically allocates effort and achieves 100% pass rate at lower cost/latency."""
        from core.thinking_router import SystemOneThinkingRouter

        router = SystemOneThinkingRouter()
        r_simple = router.route_turn("Calculate (42 * 3.14159) / 7 and write to calculation_result.txt")
        assert r_simple.reasoning_effort == "none"

        r_med = router.route_turn("Use patch_file to change the URL in config.json")
        assert r_med.reasoning_effort == "medium"

        r_high = router.route_turn("Write a shell script setup_env.sh, run git init, set executable permissions, and execute it.")
        assert r_high.reasoning_effort == "high"

        core_tasks = load_tasks(suite="core")
        configs = get_ablation_configs("adaptive_thinking")
        report = await run_benchmark_matrix(tasks=core_tasks, configs=configs, mode="mock", trials=1)
        summaries = {s["config_id"]: s for s in report["config_summaries"]}

        s_adapt = summaries["thinking_adaptive_s1"]
        s_high = summaries["thinking_high_static"]
        s_none = summaries["thinking_none_static"]

        # Adaptive matches Static HIGH's 100% pass rate (9/9) while Static NONE fails G-H03 (8/9 = 88.9%)
        assert s_adapt["pass_rate"] == 1.0
        assert s_high["pass_rate"] == 1.0
        assert s_none["pass_rate"] < 1.0

        # Adaptive is significantly faster and cheaper than Static HIGH
        assert s_adapt["avg_duration_sec"] < s_high["avg_duration_sec"]
        assert s_adapt["avg_tokens"] < s_high["avg_tokens"]
        assert s_adapt["total_cost_usd"] < s_high["total_cost_usd"]

