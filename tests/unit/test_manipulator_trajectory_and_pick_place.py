"""Unit tests for Franka Panda, UR5e, and SO-ARM100 trajectory, reachability, and pick-and-place execution.

Verifies:
1. Home pose configuration validation and execution.
2. Joint limit enforcement and rejection of out-of-boundary commands.
3. Unknown joint rejection.
4. Kinematic reachability envelope limits (0.95m max radius).
5. Multi-phase collision-aware pick-and-place execution and simulator state verification.
"""
import asyncio
import pytest
from robosahaya.tasks.contracts import (
    TaskSpecification,
    TaskType,
    ActionStep,
    TaskExecutionState,
)
from robosahaya.simulators.task_execution import SimulatorTaskExecutionAdapter


def test_panda_home_pose_execution():
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        spec = TaskSpecification(
            task_id="test_panda_home",
            original_instruction="Return Franka Panda to home pose",
            task_type=TaskType.MANIPULATION_TRAJECTORY,
            canonical_robot_id="franka_panda",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            world_id="tabletop_world",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="move_joints",
                    description="Return to home pose",
                    parameters={"pose": "home", "duration_sec": 0.5},
                )
            ],
            is_supported=True,
        )
        res = await adapter.execute_task(spec, context={"headless": True})
        assert res["success"] is True
        assert res["state"] == "completed"
        step_res = res["action_results"][0]
        assert step_res.success is True
        assert step_res.measured_data["motion_verified"] is True
        assert step_res.measured_data["max_joint_error_rad"] < 0.20

    asyncio.run(_run())


def test_joint_limits_and_unknown_joint_rejection():
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        # 1. Out-of-bounds joint command (panda_joint1 limit is approx [-2.89, 2.89])
        spec_out_of_bounds = TaskSpecification(
            task_id="test_oob_joint",
            original_instruction="Move joint 1 to 5.0 rad",
            task_type=TaskType.MANIPULATION_TRAJECTORY,
            canonical_robot_id="franka_panda",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="move_joints",
                    description="Move joint 1 out of bounds",
                    parameters={"joint_positions": {"panda_joint1": 5.0}, "duration_sec": 0.2},
                )
            ],
            is_supported=True,
        )
        res_oob = await adapter.execute_task(spec_out_of_bounds, context={"headless": True})
        assert res_oob["success"] is False
        assert "limits" in res_oob["error"].lower()

        # 2. Unknown joint name rejection
        spec_unknown = TaskSpecification(
            task_id="test_unknown_joint",
            original_instruction="Move non_existent_joint to 0.5 rad",
            task_type=TaskType.MANIPULATION_TRAJECTORY,
            canonical_robot_id="franka_panda",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="move_joints",
                    description="Move non-existent joint",
                    parameters={"joint_positions": {"phantom_joint_99": 0.5}, "duration_sec": 0.2},
                )
            ],
            is_supported=True,
        )
        res_unknown = await adapter.execute_task(spec_unknown, context={"headless": True})
        assert res_unknown["success"] is False
        assert "unknown joint" in res_unknown["error"].lower()

    asyncio.run(_run())


def test_reachability_envelope_boundaries():
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        # Command a point beyond reachable radius (> 0.95m)
        spec = TaskSpecification(
            task_id="test_unreachable_point",
            original_instruction="Reach target pose [2.5, 2.5, 1.0]",
            task_type=TaskType.MANIPULATION_TRAJECTORY,
            canonical_robot_id="franka_panda",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="reach_target_pose",
                    description="Reach point far outside reachability envelope",
                    parameters={"target_xyz": [2.5, 2.5, 1.0], "duration_sec": 0.3},
                )
            ],
            is_supported=True,
        )
        res = await adapter.execute_task(spec, context={"headless": True})
        assert res["success"] is False
        assert "reachable" in res["error"].lower() or "envelope" in res["error"].lower()

    asyncio.run(_run())


def test_pick_and_place_pipeline_physics_verification():
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        spec = TaskSpecification(
            task_id="test_panda_pick_place",
            original_instruction="Pick target_cube_red and place in target_bin_green",
            task_type=TaskType.MANIPULATION_PICK_PLACE,
            canonical_robot_id="franka_panda",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            world_id="tabletop_world",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="pick_and_place",
                    description="Execute collision-free pick-and-place",
                    parameters={
                        "target_object_id": "target_cube_red",
                        "destination_id": "target_bin_green",
                        "duration_sec": 2.0,
                    },
                )
            ],
            is_supported=True,
        )
        res = await adapter.execute_task(spec, context={"headless": True})
        assert res["success"] is True
        assert res["state"] == "completed"
        step_res = res["action_results"][0]
        assert step_res.success is True
        data = step_res.measured_data
        assert data["lift_displacement_m"] > 0.02
        assert data["placement_error_m"] <= 0.12
        assert data["picked_and_placed"] is True
        assert data["predicate_satisfied"] is True

    asyncio.run(_run())
