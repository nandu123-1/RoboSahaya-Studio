"""End-to-end integration tests for Robot Soccer Kit (RSK) and MuJoCo runtime."""
import asyncio
from pathlib import Path
import pytest

from robosahaya.evidence_models import TaskStage
from robosahaya.orchestrator import TaskOrchestrator
from robosahaya.task_store import TaskStore
from robosahaya.workspace_manager import WorkspaceManager
from robosahaya.simulators.gazebo_adapter import GazeboAdapter
from robosahaya.simulators.mujoco_adapter import MujocoAdapter


@pytest.fixture
def orchestrator(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    templates_dir = repo_root / "simulation" / "templates"
    store = TaskStore(tmp_path / "tasks")
    ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)
    return TaskOrchestrator(task_store=store, workspace_manager=ws_mgr, mode="mock")


def test_rsk_plan_formulation(orchestrator):
    plan = orchestrator.formulate_plan("task_rsk_01", "Please build the rsk robot in simulation")
    assert plan.is_supported is True
    assert plan.robot_template == "rsk"
    assert plan.world_template == "soccer_pitch"
    assert plan.simulator_adapter == "mujoco"
    assert plan.model_format == "MJCF"
    assert "test_rsk_physics_stepping" in plan.verification_tests
    assert "Omnidirectional Wheel Encoders" in plan.sensors_selected
    assert plan.plan_digest != ""


def test_rsk_execution_milestones_e2e(orchestrator):
    async def _run():
        events = []
        def on_event(evt):
            events.append(evt)

        plan = orchestrator.formulate_plan("task_rsk_exec", "Build the RSK robot")
        assert plan.is_supported is True

        res = await orchestrator.execute_studio_pipeline(
            task_id="task_rsk_exec",
            query="Build the RSK robot",
            plan=plan,
            on_event_cb=on_event,
            headless=True
        )

        assert res["stage"] == TaskStage.COMPLETED
        assert res["simulator"] == "mujoco"
        assert res["model_id"] == "rsk"
        assert res["test_result"].passed is True
        assert res["test_result"].test_id == "test_rsk_physics_stepping"

        milestones = res["milestones"]
        # Milestone A: Model structure & assets
        assert milestones["milestone_a"]["passed"] is True
        assert milestones["milestone_a"]["actuators"] == 4
        assert milestones["milestone_a"]["bodies"] >= 60

        # Milestone B: Physics Stepping
        assert milestones["milestone_b"]["passed"] is True
        assert milestones["milestone_b"]["steps_executed"] == 100
        assert milestones["milestone_b"]["sim_time_sec"] >= 0.1

        # Milestone C: Control verification
        assert milestones["milestone_c"]["passed"] is True
        assert milestones["milestone_c"]["actuator_count"] == 4

        # Milestone D: ROS 2 Truthful inspection
        assert milestones["milestone_d"]["passed"] is True
        assert milestones["milestone_d"]["ros2_bridged"] is False

        # Milestone E: Overall task verification
        assert milestones["milestone_e"]["passed"] is True
        assert milestones["milestone_e"]["status"] == "simulation_ready"

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


def test_gazebo_adapter_rejects_rsk_mjcf():
    gazebo = GazeboAdapter()
    rsk_model = get_rsk_model()
    compat = gazebo.inspect_compatibility(rsk_model)
    assert compat["compatible"] is False
    assert "not supported by Gazebo adapter" in compat["reason"] or "does not declare Gazebo" in compat["reason"]


def test_mujoco_adapter_rejects_urdf_diff_drive():
    mujoco = MujocoAdapter()
    from robosahaya.robot_registry import get_robot_registry
    diff_model = get_robot_registry().get_model("differential_drive")
    compat = mujoco.inspect_compatibility(diff_model)
    assert compat["compatible"] is False
    assert "not supported by MuJoCo adapter" in compat["reason"] or "does not declare MuJoCo" in compat["reason"]


def get_rsk_model():
    from robosahaya.robot_registry import get_robot_registry
    return get_robot_registry().get_model("rsk")
