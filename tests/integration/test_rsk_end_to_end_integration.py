"""End-to-end integration tests for RSK and Panda tasks through TaskOrchestrator.

Validates the full Agent A + Agent B integration path:
Prompt -> Plan Generation -> Shared Contract -> Orchestrator -> Simulator Adapter
-> Real Physics Stepping / Readiness Gating -> Measured Telemetry -> UI Reporting.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from robosahaya.evidence_models import TaskStage
from robosahaya.orchestrator import TaskOrchestrator
from robosahaya.task_store import TaskStore
from robosahaya.workspace_manager import WorkspaceManager
from robosahaya.tasks.contracts import TaskType
from robosahaya.ui import RoboSahayaStudioApp


def _rsk_scene_available() -> bool:
    candidates = (
        Path.home() / ".cache/robot_descriptions/mujoco_menagerie/robot_soccer_kit/scene.xml",
        Path.home() / ".cache/robosahaya/robot_models/rsk/v1.0.0/model/scene.xml",
    )
    return any(path.is_file() for path in candidates)


@pytest.fixture
def orchestrator(tmp_path: Path) -> TaskOrchestrator:
    repo_root = Path(__file__).resolve().parents[2]
    templates_dir = repo_root / "simulation" / "templates"
    store = TaskStore(tmp_path / "tasks")
    ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)
    return TaskOrchestrator(task_store=store, workspace_manager=ws_mgr, mode="mock")


def test_e2e_rsk_wheel_velocity_pipeline(orchestrator: TaskOrchestrator):
    """Reproduce: 'Build RSK with wheel1_speed as 5.44'.
    
    Verifies that:
    1. Canonical model is identified as 'rsk'.
    2. Task is recognized as wheel-velocity control.
    3. Action plan is generated using shared contracts.
    4. Execution commands wheel1_speed at 5.44 rad/s on loaded MJCF model.
    5. Physics steps advance and joint velocity responds near 5.44 rad/s.
    6. Verification confirms actuator response from measured joint state.
    """
    if not _rsk_scene_available():
        pytest.skip("RSK MuJoCo scene is not present in local cache.")
    pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")

    async def _run():
        task_id = "test_rsk_e2e_01"
        query = "Build RSK with wheel1_speed as 5.44"
        events = []

        def on_event(evt):
            events.append(evt)

        # 1. Plan Formulation
        plan = orchestrator.formulate_plan(task_id, query)
        assert plan.is_supported is True
        assert plan.robot_template == "rsk"
        assert plan.world_template == "soccer_pitch"
        assert plan.simulator_adapter == "mujoco"
        assert plan.task_spec is not None
        assert plan.task_spec["task_type"] == TaskType.MOBILE_WHEEL_CONTROL
        assert plan.task_spec["canonical_robot_id"] == "rsk"
        assert len(plan.task_spec["action_sequence"]) == 1

        step0 = plan.task_spec["action_sequence"][0]
        assert step0["skill_id"] == "set_wheel_velocity"
        assert step0["parameters"]["wheel_name"] == "wheel1_speed"
        assert step0["parameters"]["velocity"] == pytest.approx(5.44)

        # 2. Pipeline Execution
        res = await orchestrator.execute_studio_pipeline(
            task_id=task_id,
            query=query,
            plan=plan,
            on_event_cb=on_event,
            headless=True,
        )

        # 3. Verification of Results
        assert res["stage"] == TaskStage.COMPLETED
        assert res["simulator"] == "mujoco"
        assert res["model_id"] == "rsk"
        test_result = res["test_result"]
        assert test_result.passed is True
        assert test_result.source_mode == "live"
        assert test_result.status == "passed"

        meas = test_result.measured_values
        assert meas["steps_completed"] == 1
        assert meas["total_steps"] == 1
        assert abs(meas["measured_joint_velocity"] - 5.44) < 1.0
        assert meas["sim_time"] > 0.0

        # Verify action result observations
        assert len(res["observations"]) == 1
        obs = res["observations"][0]
        assert obs.step_success is True
        assert obs.observed_state["actuator_name"] == "wheel1_speed"
        assert obs.observed_state["motion_verified"] is True
        assert abs(obs.observed_state["measured_joint_velocity_rad_s"] - 5.44) < 1.0

        # Verify event sequence
        stages = [e.stage for e in events]
        assert TaskStage.RECEIVED in stages
        assert TaskStage.PLANNING in stages
        assert TaskStage.PREPARING in stages
        assert TaskStage.BUILDING in stages
        assert TaskStage.LAUNCHING in stages
        assert TaskStage.TESTING in stages
        assert TaskStage.COMPLETED in stages

    asyncio.run(_run())


def test_e2e_panda_pick_and_place_remains_honestly_blocked(orchestrator: TaskOrchestrator):
    """Reproduce: 'Pick up the object in front of the arm and place it on the plate.'
    With robot set to Franka Emika Panda.
    
    Verifies that:
    1. Task is parsed as manipulation_pick_place.
    2. System checks required preconditions: arm controller, gripper controller, MoveIt.
    3. Execution reports task is blocked due to missing MoveIt / controller setup.
    4. Task does not claim success.
    5. Readiness limitations are clearly detailed.
    """
    async def _run():
        task_id = "test_panda_blocked_01"
        query = "Pick up the object in front of the arm and place it on the plate."
        events = []

        def on_event(evt):
            events.append(evt)

        # 1. Plan Formulation with Franka Panda
        plan = orchestrator.formulate_plan(
            task_id,
            query,
            template_id="franka_panda",
            world_id="tabletop_world",
        )
        assert plan.is_supported is True
        assert plan.robot_template == "franka_panda"
        assert plan.world_template == "tabletop_world"
        assert plan.task_spec is not None
        assert plan.task_spec["task_type"] == TaskType.MANIPULATION_PICK_PLACE
        assert plan.task_spec["canonical_robot_id"] == "franka_panda"
        assert "target_cube_red" in plan.task_spec["target_objects"]
        assert plan.task_spec["destination"] == "target_bin_green"

        # 2. Pipeline Execution
        res = await orchestrator.execute_studio_pipeline(
            task_id=task_id,
            query=query,
            plan=plan,
            on_event_cb=on_event,
            headless=True,
        )

        # 3. Honest Gating: Must be BLOCKED, never completed
        assert res["stage"] == TaskStage.BLOCKED
        test_result = res["test_result"]
        assert test_result.passed is False
        assert test_result.status == "blocked"

        err = res.get("error", "")
        assert "ros2_control" in err or "controller" in err or "MoveIt" in err

        # Verify event sequence contains BLOCKED
        stages = [e.stage for e in events]
        assert TaskStage.BLOCKED in stages
        assert TaskStage.COMPLETED not in stages

    asyncio.run(_run())


def test_e2e_ui_telemetry_rendering(orchestrator: TaskOrchestrator):
    """Verifies UI telemetry formatting for both completed RSK and blocked Panda tasks."""
    if not _rsk_scene_available():
        pytest.skip("RSK MuJoCo scene is not present in local cache.")
    pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")

    async def _run():
        app = RoboSahayaStudioApp()
        try:
            # 1. RSK Success Telemetry
            rsk_plan = orchestrator.formulate_plan("task_ui_rsk", "Build RSK with wheel1_speed as 5.44")
            rsk_res = await orchestrator.execute_studio_pipeline(
                "task_ui_rsk",
                "Build RSK with wheel1_speed as 5.44",
                plan=rsk_plan,
                headless=True,
            )
            rsk_md = app.format_sensor_status(rsk_res["test_result"], rsk_res["stage"])
            assert "Closed-Loop Task Execution & Verification Telemetry" in rsk_md
            assert "set_wheel_velocity" in rsk_md
            assert "🟢 Passed" in rsk_md
            assert "VERIFIED SUCCESS" in rsk_md

            # 2. Panda Blocked Telemetry
            panda_plan = orchestrator.formulate_plan(
                "task_ui_panda",
                "Pick up the object in front of the arm and place it on the plate.",
                template_id="franka_panda",
                world_id="tabletop_world",
            )
            panda_res = await orchestrator.execute_studio_pipeline(
                "task_ui_panda",
                "Pick up the object in front of the arm and place it on the plate.",
                plan=panda_plan,
                headless=True,
            )
            panda_md = app.format_sensor_status(panda_res["test_result"], panda_res["stage"])
            assert "Closed-Loop Task Execution & Verification Telemetry" in panda_md
            assert "🔴 Failed" in panda_md
            assert "FAILED / DISCREPANCY DETECTED" in panda_md
        finally:
            if hasattr(app, "inspector") and app.inspector and hasattr(app.inspector, "shutdown"):
                app.inspector.shutdown()

    asyncio.run(_run())


def test_rsk_viewer_subprocess_lifecycle_and_clean_exit():
    """Verify that the real RSK viewer subprocess exits cleanly with returncode 0 without GLXBadDrawable."""
    import os
    if not _rsk_scene_available():
        pytest.skip("RSK MuJoCo scene is not present in local cache.")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.path.exists("/tmp/.X11-unix/X1")):
        pytest.skip("No graphical display detected for GUI test.")
    pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")

    import subprocess, sys
    runner_path = Path(__file__).resolve().parents[2] / "src" / "robosahaya" / "simulators" / "mujoco_runner.py"
    rsk_path = Path.home() / ".cache/robosahaya/robot_models/rsk/v1.0.0/model/robot_soccer_kit.xml"
    if not rsk_path.exists():
        rsk_path = Path.home() / ".cache/robot_descriptions/mujoco_menagerie/robot_soccer_kit/robot_soccer_kit.xml"

    env = os.environ.copy()
    if not env.get("DISPLAY") and os.path.exists("/tmp/.X11-unix/X1"):
        env["DISPLAY"] = ":1"

    proc = subprocess.run(
        [
            sys.executable,
            str(runner_path),
            str(rsk_path),
            "5.44",
            "wheel1_speed",
            "0.3",
            "0.6",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert proc.returncode == 0, f"Viewer failed with code {proc.returncode}, stderr: {proc.stderr}"
    assert "GLXBadDrawable" not in proc.stderr
    assert "GLXBadContext" not in proc.stderr
    assert "motion_verified=True" in proc.stdout


def test_e2e_rsk_drive_forward_pipeline(orchestrator: TaskOrchestrator):
    """Verifies end-to-end drive forward instruction execution through orchestrator and real MuJoCo."""
    if not _rsk_scene_available():
        pytest.skip("RSK MuJoCo scene is not present in local cache.")
    pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")

    async def _run():
        task_id = "test_rsk_drive_01"
        query = "Drive the soccer robot forward at 0.5 m/s for 2 seconds"
        events = []

        plan = orchestrator.formulate_plan(task_id, query)
        assert plan.is_supported is True
        assert plan.robot_template == "rsk"
        assert plan.simulator_adapter == "mujoco"
        assert len(plan.action_steps) == 1
        assert plan.action_steps[0]["skill_id"] == "drive_forward"

        res = await orchestrator.execute_studio_pipeline(
            task_id=task_id,
            query=query,
            plan=plan,
            on_event_cb=lambda e: events.append(e),
            headless=True,
        )

        assert res["stage"] == TaskStage.COMPLETED
        assert res["test_result"].passed is True
        meas = res["test_result"].measured_values
        assert meas["steps_completed"] == 1
        assert meas["measured_displacement_m"] > 0.005
        assert any(e.stage == TaskStage.COMPLETED for e in events)

    asyncio.run(_run())


def test_e2e_rsk_turn_in_place_pipeline(orchestrator: TaskOrchestrator):
    """Verifies end-to-end turn in place instruction execution through orchestrator and real MuJoCo."""
    if not _rsk_scene_available():
        pytest.skip("RSK MuJoCo scene is not present in local cache.")
    pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")

    async def _run():
        task_id = "test_rsk_turn_01"
        query = "Turn the soccer robot in place by 90 degrees"
        events = []

        plan = orchestrator.formulate_plan(task_id, query)
        assert plan.is_supported is True
        assert plan.robot_template == "rsk"
        assert plan.simulator_adapter == "mujoco"
        assert len(plan.action_steps) == 1
        assert plan.action_steps[0]["skill_id"] == "turn_in_place"

        res = await orchestrator.execute_studio_pipeline(
            task_id=task_id,
            query=query,
            plan=plan,
            on_event_cb=lambda e: events.append(e),
            headless=True,
        )

        assert res["stage"] == TaskStage.COMPLETED
        assert res["test_result"].passed is True
        meas = res["test_result"].measured_values
        assert meas["steps_completed"] == 1
        assert meas["measured_yaw_change_rad"] > 0.02
        assert any(e.stage == TaskStage.COMPLETED for e in events)

    asyncio.run(_run())
