"""Integration tests for multi-step compound mobile base tasks in RoboSahaya Studio.

Verifies:
1. Natural language compound instruction parsing ("Drive forward 0.5 meters, turn 90 degrees, and stop").
2. Dynamic synthesis of 3-step action sequence: drive_forward -> turn_in_place -> stop_mobile_base.
3. Physics simulation execution in MuJoCo.
4. Physical state verification: measured forward displacement > 0, yaw angle change > 0, and final velocity settled.
"""
import asyncio
import pytest
from robosahaya.tasks.contracts import TaskType
from robosahaya.tasks.interpreter import get_task_interpreter
from robosahaya.simulators.task_execution import SimulatorTaskExecutionAdapter


def test_compound_mobile_instruction_interpretation():
    interp = get_task_interpreter()
    spec = interp.interpret_instruction("Drive forward 0.5 meters, turn 90 degrees, and stop")

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MOBILE_WHEEL_CONTROL
    assert len(spec.action_sequence) == 3

    assert spec.action_sequence[0].skill_id == "drive_forward"
    assert spec.action_sequence[0].parameters.get("distance_m") == 0.5

    assert spec.action_sequence[1].skill_id == "turn_in_place"
    assert spec.action_sequence[1].parameters.get("angle_deg") == 90.0

    assert spec.action_sequence[2].skill_id == "stop_mobile_base"


def test_compound_mobile_rsk_execution():
    interp = get_task_interpreter()
    adapter = SimulatorTaskExecutionAdapter()

    async def _run():
        spec = interp.interpret_instruction("Drive forward 0.5 meters, turn 90 degrees, and stop on RSK")
        assert spec.canonical_robot_id == "rsk"
        assert spec.is_supported is True
        spec.simulator_adapter = "mujoco"

        res = await adapter.execute_task(spec, context={"headless": True})
        assert res["success"] is True
        assert res["state"] == "completed"
        assert len(res["action_results"]) == 3

        # Step 1: Forward displacement
        step1 = res["action_results"][0]
        assert step1.success is True
        assert step1.measured_data["measured_displacement_m"] > 0.005

        # Step 2: Turn in place
        step2 = res["action_results"][1]
        assert step2.success is True
        assert step2.measured_data["measured_yaw_change_rad"] > 0.01

        # Step 3: Stop
        step3 = res["action_results"][2]
        assert step3.success is True
        assert step3.measured_data["is_stopped"] is True

        # Overall verification conditions
        assert res["verification_result"].success is True

    asyncio.run(_run())
