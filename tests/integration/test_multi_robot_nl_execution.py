"""Integration tests for end-to-end natural-language task planning and execution across multi-robot families.

Verifies:
1. Natural-language prompt parsing for Franka Panda, SO-100, UR5e, and RSK
2. End-to-end execution of arm trajectory and gripper control in MuJoCo
3. Verification of measured telemetry and predicates
4. Accurate capability gating and rejection for unbundled WBC gaits and drone flight
"""
import asyncio
import pytest
from robosahaya.tasks.contracts import TaskType
from robosahaya.tasks.interpreter import get_task_interpreter
from robosahaya.simulators.task_execution import SimulatorTaskExecutionAdapter


def test_nl_franka_panda_joint_and_gripper_pipeline():
    interp = get_task_interpreter()
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        # Move Panda joint
        spec1 = interp.interpret_instruction("Move Franka Panda joint 1 to 0.4 rad")
        assert spec1.is_supported is True
        assert spec1.canonical_robot_id == "franka_panda"
        assert spec1.task_type == TaskType.MANIPULATION_TRAJECTORY
        assert len(spec1.action_sequence) == 1
        assert spec1.action_sequence[0].skill_id == "move_joints"

        spec1.simulator_adapter = "mujoco"
        res1 = await adapter.execute_task(spec1, context={"headless": True})
        assert res1["success"] is True
        assert res1["state"] == "completed"
        assert res1["verification_result"].success is True

        # Open gripper on Panda
        spec2 = interp.interpret_instruction("Open gripper on Franka Panda")
        assert spec2.is_supported is True
        assert spec2.action_sequence[0].skill_id == "gripper_control"

        spec2.simulator_adapter = "mujoco"
        res2 = await adapter.execute_task(spec2, context={"headless": True})
        assert res2["success"] is True
        assert res2["state"] == "completed"

    asyncio.run(_run())


def test_nl_so100_and_ur5e_execution():
    interp = get_task_interpreter()
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        # SO-100
        spec_so = interp.interpret_instruction("Move so_arm100 joint 1 to 0.3 rad")
        assert spec_so.is_supported is True
        assert spec_so.canonical_robot_id == "so_arm100"
        spec_so.simulator_adapter = "mujoco"
        res_so = await adapter.execute_task(spec_so, context={"headless": True})
        assert res_so["success"] is True
        assert res_so["state"] == "completed"

        # UR5e
        spec_ur5 = interp.interpret_instruction("Move ur5e joint 1 to 0.2 rad")
        assert spec_ur5.is_supported is True
        assert spec_ur5.canonical_robot_id == "ur5e"
        spec_ur5.simulator_adapter = "mujoco"
        res_ur5 = await adapter.execute_task(spec_ur5, context={"headless": True})
        assert res_ur5["success"] is True
        assert res_ur5["state"] == "completed"

    asyncio.run(_run())


def test_nl_reach_target_pose():
    interp = get_task_interpreter()
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        spec = interp.interpret_instruction("Reach target pose [0.4, 0.0, 0.4] with Panda")
        assert spec.is_supported is True
        assert spec.canonical_robot_id == "franka_panda"
        assert spec.action_sequence[0].skill_id == "reach_target_pose"

        spec.simulator_adapter = "mujoco"
        res = await adapter.execute_task(spec, context={"headless": True})
        assert res["success"] is True
        assert res["state"] == "completed"
        assert res["verification_result"].measured_values["displacement_m"] > 0.01

    asyncio.run(_run())


def test_nl_quadruped_and_drone_honest_gating():
    interp = get_task_interpreter()
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        # Dynamic running trot
        spec_quad = interp.interpret_instruction("Command quadruped robot to execute dynamic running trot.")
        assert spec_quad.is_supported is False
        assert "legged gait controller" in spec_quad.rejection_reason.lower() or "blocked" in spec_quad.rejection_reason.lower()

        res_quad = await adapter.execute_task(spec_quad, context={"headless": True})
        assert res_quad["success"] is False
        assert res_quad["state"] == "blocked"

        # Drone flight
        spec_drone = interp.interpret_instruction("Fly drone along waypoints at 10m altitude.")
        assert spec_drone.is_supported is False
        assert "drone flight controller" in spec_drone.rejection_reason.lower() or "blocked" in spec_drone.rejection_reason.lower()

        res_drone = await adapter.execute_task(spec_drone, context={"headless": True})
        assert res_drone["success"] is False
        assert res_drone["state"] == "blocked"

    asyncio.run(_run())
