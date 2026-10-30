"""Unit tests for multi-robot manipulator and quadruped posture skills in MuJoCo.

Verifies:
1. ReachTargetPoseSkill parameter validation
2. Real MuJoCo execution of move_joints on Franka Panda, UR5e, and SO-100
3. Real MuJoCo execution of gripper_control on Franka Panda
4. Real MuJoCo execution of stand_sit_posture on Unitree Go2
5. Honest gating of pick-and-place and dynamic quadruped trot
"""
import asyncio
import pytest
from robosahaya.tasks.contracts import (
    ActionStep,
    SkillStatus,
    TaskSpecification,
    TaskType,
)
from robosahaya.tasks.skills import (
    ReachTargetPoseSkill,
    get_skill_registry,
)
from robosahaya.simulators.task_execution import SimulatorTaskExecutionAdapter


def test_reach_target_pose_parameter_validation():
    skill = ReachTargetPoseSkill()
    skill.validate_parameters({"target_xyz": [0.4, 0.0, 0.4], "tolerance_m": 0.05})

    with pytest.raises(ValueError, match="3-element list or tuple"):
        skill.validate_parameters({"target_xyz": [0.4, 0.0]})

    with pytest.raises(ValueError, match="positive number"):
        skill.validate_parameters({"target_xyz": [0.4, 0.0, 0.4], "tolerance_m": -0.1})


def test_real_mujoco_panda_move_joints_and_gripper():
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        spec = TaskSpecification(
            task_id="test_panda_joints",
            original_instruction="Move Franka Panda joint 1 to 0.4 rad",
            task_type=TaskType.MANIPULATION_TRAJECTORY,
            canonical_robot_id="franka_panda",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            world_id="tabletop_world",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="move_joints",
                    description="Set joint 1",
                    parameters={"joint_positions": {"panda_joint1": 0.4}, "duration_sec": 0.5},
                ),
                ActionStep(
                    step_index=2,
                    skill_id="gripper_control",
                    description="Open gripper",
                    parameters={"action": "open", "duration_sec": 0.3},
                ),
            ],
            is_supported=True,
        )
        res = await adapter.execute_task(spec, context={"headless": True})
        assert res["success"] is True
        assert res["state"] == "completed"
        assert len(res["action_results"]) == 2
        assert res["action_results"][0].success is True
        assert res["action_results"][1].success is True
        assert res["action_results"][0].measured_data["max_joint_error_rad"] < 0.15

    asyncio.run(_run())


def test_real_mujoco_ur5e_move_joints():
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        spec = TaskSpecification(
            task_id="test_ur5e_joints",
            original_instruction="Move UR5e joint 1 to 0.2 rad",
            task_type=TaskType.MANIPULATION_TRAJECTORY,
            canonical_robot_id="ur5e",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            world_id="tabletop_world",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="move_joints",
                    description="Set joint 1",
                    parameters={"joint_positions": {"1": 0.2}, "duration_sec": 0.5},
                )
            ],
            is_supported=True,
        )
        res = await adapter.execute_task(spec, context={"headless": True})
        assert res["success"] is True
        assert res["state"] == "completed"
        assert res["action_results"][0].measured_data["max_joint_error_rad"] < 0.20

    asyncio.run(_run())


def test_real_mujoco_so100_move_joints():
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        spec = TaskSpecification(
            task_id="test_so100_joints",
            original_instruction="Move SO-100 joint 1 to 0.3 rad",
            task_type=TaskType.MANIPULATION_TRAJECTORY,
            canonical_robot_id="so_arm100",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            world_id="tabletop_world",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="move_joints",
                    description="Set joint 1",
                    parameters={"joint_positions": {"1": 0.3}, "duration_sec": 0.5},
                )
            ],
            is_supported=True,
        )
        res = await adapter.execute_task(spec, context={"headless": True})
        assert res["success"] is True
        assert res["state"] == "completed"
        assert res["action_results"][0].measured_data["max_joint_error_rad"] < 0.20

    asyncio.run(_run())


def test_real_mujoco_go2_stand_posture():
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        spec = TaskSpecification(
            task_id="test_go2_posture",
            original_instruction="Set Unitree Go2 to stand posture",
            task_type=TaskType.LEGGED_LOCOMOTION,
            canonical_robot_id="unitree_go2",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            world_id="obstacle_world",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="stand_sit_posture",
                    description="Stand posture",
                    parameters={"posture": "stand", "duration_sec": 0.5},
                )
            ],
            is_supported=True,
        )
        res = await adapter.execute_task(spec, context={"headless": True})
        assert res["success"] is True
        assert res["state"] == "completed"
        assert res["action_results"][0].measured_data["stable"] is True

    asyncio.run(_run())


def test_mujoco_honest_gating_quadruped_gait_and_drone():
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        spec_gait = TaskSpecification(
            task_id="test_gait_blocked",
            original_instruction="Trot forward",
            task_type=TaskType.LEGGED_LOCOMOTION,
            canonical_robot_id="unitree_go2",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            world_id="obstacle_world",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="legged_gait",
                    description="Trotting gait",
                    parameters={"gait": "trot"},
                )
            ],
            is_supported=True,
        )
        res_gait = await adapter.execute_task(spec_gait, context={"headless": True})
        assert res_gait["success"] is False
        assert res_gait["state"] == "blocked"
        assert "whole-body controller" in res_gait["error"].lower()

        spec_drone = TaskSpecification(
            task_id="test_drone_blocked",
            original_instruction="Fly drone",
            task_type=TaskType.DRONE_FLIGHT,
            canonical_robot_id="skydio_x2",
            robot_model_version="v1.0.0",
            simulator_adapter="mujoco",
            world_id="obstacle_world",
            action_sequence=[
                ActionStep(
                    step_index=1,
                    skill_id="drone_flight",
                    description="Takeoff",
                    parameters={"altitude_m": 1.5},
                )
            ],
            is_supported=True,
        )
        res_drone = await adapter.execute_task(spec_drone, context={"headless": True})
        assert res_drone["success"] is False
        assert res_drone["state"] == "blocked"
        assert "aerodynamic" in res_drone["error"].lower() or "flight controller" in res_drone["error"].lower()

    asyncio.run(_run())
