import asyncio
import pytest
from pathlib import Path

from robosahaya.evidence_models import TaskStage
from robosahaya.orchestrator import TaskOrchestrator
from robosahaya.task_store import TaskStore
from robosahaya.workspace_manager import WorkspaceManager


@pytest.fixture
def orchestrator(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    templates_dir = repo_root / "simulation" / "templates"
    store = TaskStore(tmp_path / "tasks")
    ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)
    return TaskOrchestrator(task_store=store, workspace_manager=ws_mgr, mode="mock")


def test_automatic_selection_inference_diff_drive(orchestrator):
    infer = orchestrator.infer_robot_and_world("Build a wheeled differential drive robot with LiDAR and camera.")
    assert infer["auto_robot"] == "differential_drive"
    assert infer["auto_world"] == "obstacle_world"
    assert "Inferred differential-drive mobile chassis" in infer["robot_reason"]
    assert "Inferred obstacle_world" in infer["world_reason"]
    assert "2D Planar LiDAR" in infer["sensors"]
    assert "/scan" in infer["topics"]
    assert infer["selection_source"] in ("rule_based", "evon")


def test_formulate_plan_auto_selected_indicator(orchestrator):
    plan = orchestrator.formulate_plan("task_auto_01", "Create a differential drive mobile robot in obstacle course.")
    assert plan.is_supported is True
    assert plan.robot_template == "differential_drive"
    assert plan.world_template == "obstacle_world"
    assert plan.is_auto_selected is True
    assert plan.is_manual_override is False
    assert plan.plan_digest != ""
    assert plan.robot_selection_reason != ""
    assert plan.world_selection_reason != ""


def test_manual_override_tracking(orchestrator):
    # User overrides robot template manually
    manual_overrides = {"robot_template": "differential_drive", "world_template": "obstacle_world"}
    plan = orchestrator.formulate_plan(
        "task_override_01",
        "Create a mobile robot.",
        manual_overrides=manual_overrides
    )
    assert plan.is_manual_override is True
    assert plan.is_auto_selected is False
    assert plan.manual_overrides == manual_overrides


def test_reset_to_ai_selected_clears_overrides(orchestrator):
    # Simulating a user resetting manual overrides
    overridden_plan = orchestrator.formulate_plan(
        "task_reset_01",
        "Create a differential drive robot.",
        manual_overrides={"robot_template": "custom_chassis"}
    )
    assert overridden_plan.is_manual_override is True

    # Clearing overrides restores planner's automatic selection
    reset_plan = orchestrator.formulate_plan(
        "task_reset_01",
        "Create a differential drive robot.",
        manual_overrides={}
    )
    assert reset_plan.is_manual_override is False
    assert reset_plan.is_auto_selected is True
    assert reset_plan.robot_template == "differential_drive"
    assert reset_plan.is_supported is True


def test_supported_robot_arm_selection(orchestrator):
    plan = orchestrator.formulate_plan("task_arm_01", "Create a 6-DOF robotic arm with gripper to pick up blocks.")
    assert plan.is_supported is True
    assert plan.robot_template == "robotic_arm_6dof"
    assert plan.world_template == "tabletop_world"
    assert "Joint Position Encoders" in plan.sensors_selected
    assert "/joint_states" in plan.ros_topics


def test_unsupported_drone_rejection(orchestrator):
    plan = orchestrator.formulate_plan("task_drone_01", "Deploy an aerial drone quadcopter to fly over obstacles.")
    assert plan.is_supported is False
    assert "aerial_drone" in plan.unsupported_capabilities
    assert "Unsupported Capability" in plan.rejection_reason
    assert "Aerial drone" in plan.rejection_reason


def test_stale_plan_execution_prevention(orchestrator):
    # Plan formulated for one prompt
    plan = orchestrator.formulate_plan("task_stale_01", "Create a differential drive robot with LiDAR.")
    assert plan.is_supported is True

    # Attempt to execute with modified query without re-formulating plan
    async def _run():
        res = await orchestrator.execute_studio_pipeline(
            task_id="task_stale_01",
            query="Create a camera robot with no LiDAR.",  # Changed query
            plan=plan
        )
        assert res["stage"] == TaskStage.BLOCKED
        assert "Stale plan detected" in res["error"]

    asyncio.run(_run())


def test_explicit_stale_flag_execution_prevention(orchestrator):
    plan = orchestrator.formulate_plan("task_stale_02", "Create a differential drive robot with LiDAR.")
    plan.is_stale = True

    async def _run():
        res = await orchestrator.execute_studio_pipeline(
            task_id="task_stale_02",
            query="Create a differential drive robot with LiDAR.",
            plan=plan
        )
        assert res["stage"] == TaskStage.BLOCKED
        assert "Stale plan detected" in res["error"]

    asyncio.run(_run())
