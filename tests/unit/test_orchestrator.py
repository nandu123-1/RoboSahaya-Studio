import pytest
from pathlib import Path
from robosahaya.orchestrator import TaskOrchestrator
from robosahaya.task_store import TaskStore
from robosahaya.workspace_manager import WorkspaceManager
from robosahaya.evidence_models import TaskStage

import asyncio

def test_orchestrator_full_repair_loop(tmp_path):
    async def _run():
        repo_root = Path(__file__).resolve().parent.parent.parent
        templates_dir = repo_root / "simulation" / "templates"
        store = TaskStore(tmp_path / "tasks")
        ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)

        orch = TaskOrchestrator(task_store=store, workspace_manager=ws_mgr, mode="mock")
        task_id = "test_studio_run"

        events = []
        def on_event(e):
            events.append(e)

        # 1. Execute initial pipeline (with injected fault)
        res = await orch.execute_studio_pipeline(
            task_id=task_id,
            query="Create a small differential-drive robot with LiDAR in Gazebo, launch it, test sensors, and fix any issue.",
            inject_fault=True,
            on_event_cb=on_event
        )

        assert res["stage"] == TaskStage.AWAITING_APPROVAL
        assert "patch_proposal" in res
        assert res["patch_proposal"].patch_id == "patch_enable_laser_bridge"

        # Verify event order
        stages = [e.stage for e in events]
        assert TaskStage.RECEIVED in stages
        assert TaskStage.PLANNING in stages
        assert TaskStage.PREPARING in stages
        assert TaskStage.BUILDING in stages
        assert TaskStage.LAUNCHING in stages
        assert TaskStage.TESTING in stages
        assert TaskStage.DIAGNOSING in stages
        assert TaskStage.PATCH_PROPOSED in stages
        assert TaskStage.AWAITING_APPROVAL in stages

        # 2. Approve and retest
        proposal = res["patch_proposal"]
        token = res["approval_token"]
        ws_dir = res["workspace_dir"]

        retest_res = await orch.approve_and_retest(
            task_id=task_id,
            proposal=proposal,
            approval_token=token,
            workspace_dir=ws_dir,
            on_event_cb=on_event
        )

        assert retest_res["stage"] == TaskStage.COMPLETED
        assert retest_res["retest_result"].passed is True
        assert (Path(ws_dir) / "bridge.yaml.bak").exists()

        # Final event is COMPLETED
        assert events[-1].stage == TaskStage.COMPLETED

    asyncio.run(_run())


def test_orchestrator_rejects_unsupported_drone_capability(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    templates_dir = repo_root / "simulation" / "templates"
    store = TaskStore(tmp_path / "tasks")
    ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)
    orch = TaskOrchestrator(task_store=store, workspace_manager=ws_mgr, mode="mock")

    # Drone must be rejected rather than silently using wheeled robot
    plan = orch.formulate_plan("task_drone_01", "Create an aerial drone quadcopter to fly over obstacles.")
    assert plan.is_supported is False
    assert "aerial_drone" in plan.unsupported_capabilities
    assert "Unsupported Capability" in plan.rejection_reason

    # Running pipeline with unsupported capability must return BLOCKED stage
    async def _run():
        res = await orch.execute_studio_pipeline(
            task_id="task_drone_pipeline",
            query="Create an aerial drone quadcopter to fly over obstacles."
        )
        assert res["stage"] == TaskStage.BLOCKED
        assert "Unsupported Capability" in res["error"]
    asyncio.run(_run())


def test_orchestrator_supports_robotic_arm_simulation(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    templates_dir = repo_root / "simulation" / "templates"
    store = TaskStore(tmp_path / "tasks")
    ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)
    orch = TaskOrchestrator(task_store=store, workspace_manager=ws_mgr, mode="mock")

    plan = orch.formulate_plan("task_arm_01", "Build a 6-DOF robotic arm with a gripper camera to pick and place objects.")
    assert plan.is_supported is True
    assert plan.robot_template == "robotic_arm_6dof"
    assert plan.world_template == "tabletop_world"

    async def _run():
        res = await orch.execute_studio_pipeline(
            task_id="task_arm_pipeline",
            query="Build a 6-DOF robotic arm with a gripper camera to pick and place objects.",
            inject_fault=False,
            plan=plan
        )
        assert res["stage"] == TaskStage.COMPLETED
        assert res["test_result"].passed is True
        assert res["test_result"].test_id == "test_joint_states_payload"
    asyncio.run(_run())



def test_orchestrator_formulates_valid_diff_drive_plan(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    templates_dir = repo_root / "simulation" / "templates"
    store = TaskStore(tmp_path / "tasks")
    ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)
    orch = TaskOrchestrator(task_store=store, workspace_manager=ws_mgr, mode="mock")

    plan = orch.formulate_plan("task_valid_01", "Create a small differential-drive robot with LiDAR in obstacle world.")
    assert plan.is_supported is True
    assert plan.robot_template == "differential_drive"
    assert "2d_lidar_scan" in plan.requested_capabilities
    assert "differential_drive_chassis" in plan.requested_capabilities
    assert "test_laser_scan_payload" in plan.verification_tests

