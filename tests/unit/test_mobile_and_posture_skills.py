"""Tests for mobile robot locomotion, posture skills, and natural-language planning.

Verifies:
1. Registration and parameter validation for DriveForwardSkill, TurnInPlaceSkill, StopMobileBaseSkill, StandSitPostureSkill
2. Honest gating of quadruped posture skills when controllers are missing
3. Natural-language planning of forward driving, turning in place, and halting
4. Real physics execution and telemetry measurement on RSK in MuJoCo
"""
import pytest
from robosahaya.tasks.contracts import (
    SkillStatus,
    TaskType,
)
from robosahaya.tasks.interpreter import get_task_interpreter
from robosahaya.tasks.skills import (
    DriveForwardSkill,
    StandSitPostureSkill,
    StopMobileBaseSkill,
    TurnInPlaceSkill,
    get_skill_registry,
)


def test_skill_registry_contains_new_skills():
    registry = get_skill_registry()
    assert registry.get_skill("drive_forward") is not None
    assert registry.get_skill("turn_in_place") is not None
    assert registry.get_skill("stop_mobile_base") is not None
    assert registry.get_skill("stand_sit_posture") is not None


def test_drive_forward_parameter_validation():
    skill = DriveForwardSkill()
    skill.validate_parameters({"speed_m_s": 0.5, "duration_sec": 2.0})
    skill.validate_parameters({"velocity": 1.0})

    with pytest.raises(ValueError, match="exceeds safe limits"):
        skill.validate_parameters({"speed_m_s": 10.0})

    with pytest.raises(ValueError, match="must be between"):
        skill.validate_parameters({"duration_sec": 0.0})


def test_turn_in_place_parameter_validation():
    skill = TurnInPlaceSkill()
    skill.validate_parameters({"angular_velocity": 1.5, "angle_deg": 90.0})

    with pytest.raises(ValueError, match="exceeds safe limits"):
        skill.validate_parameters({"angular_velocity": 25.0})


def test_stand_sit_posture_honest_gating():
    skill = StandSitPostureSkill()
    skill.validate_parameters({"posture": "stand"})
    skill.validate_parameters({"posture": "sit"})

    with pytest.raises(ValueError, match="is invalid"):
        skill.validate_parameters({"posture": "backflip"})

    # Honest gating when quadruped controller is missing
    res = skill.execute({"posture": "stand"}, context={"has_quadruped_controller": False})
    assert res.status == SkillStatus.BLOCKED
    assert res.success is False
    assert "Missing quadruped joint controller" in res.error_reason


def test_interpreter_parses_drive_forward_query():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction("Drive the soccer robot forward at 0.5 m/s for 2 seconds")

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MOBILE_WHEEL_CONTROL
    assert spec.canonical_robot_id == "rsk"
    assert len(spec.action_sequence) == 1
    action = spec.action_sequence[0]
    assert action.skill_id == "drive_forward"
    assert action.parameters["speed_m_s"] == 0.5
    assert action.parameters["duration_sec"] == 2.0
    assert len(spec.verification_conditions) >= 1
    assert spec.verification_conditions[0].metric == "measured_displacement_m"


def test_interpreter_parses_turn_in_place_query():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction("Turn the soccer robot in place by 90 degrees")

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MOBILE_WHEEL_CONTROL
    assert spec.canonical_robot_id == "rsk"
    assert len(spec.action_sequence) == 1
    action = spec.action_sequence[0]
    assert action.skill_id == "turn_in_place"
    assert action.parameters["angle_deg"] == 90.0
    assert len(spec.verification_conditions) >= 1
    assert spec.verification_conditions[0].metric == "measured_yaw_change_rad"


def test_interpreter_parses_stop_mobile_base_query():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction("Stop the soccer robot")

    assert spec.is_supported is True
    assert spec.canonical_robot_id == "rsk"
    assert len(spec.action_sequence) == 1
    action = spec.action_sequence[0]
    assert action.skill_id == "stop_mobile_base"
    assert len(spec.verification_conditions) >= 1
    assert spec.verification_conditions[0].metric == "is_stopped"


def test_real_mujoco_drive_forward_and_turn_execution():
    import asyncio
    from robosahaya.simulators.task_execution import SimulatorTaskExecutionAdapter

    adapter = SimulatorTaskExecutionAdapter()
    interpreter = get_task_interpreter()

    async def _run():
        # Drive forward execution
        spec_drive = interpreter.interpret_instruction("Drive the soccer robot forward at 0.5 m/s for 1.0 second")
        res_drive = await adapter.execute_task(spec_drive, context={"headless": True})
        assert res_drive["success"] is True
        assert res_drive["state"] == "completed"
        assert res_drive["verification_result"].measured_values["measured_displacement_m"] > 0.005

        # Turn in place execution
        spec_turn = interpreter.interpret_instruction("Turn the soccer robot in place by 90 degrees")
        res_turn = await adapter.execute_task(spec_turn, context={"headless": True})
        assert res_turn["success"] is True
        assert res_turn["state"] == "completed"
        assert res_turn["verification_result"].measured_values["measured_yaw_change_rad"] > 0.02

        # Stop execution
        spec_stop = interpreter.interpret_instruction("Stop the soccer robot")
        res_stop = await adapter.execute_task(spec_stop, context={"headless": True})
        assert res_stop["success"] is True
        assert res_stop["state"] == "completed"
        assert res_stop["verification_result"].measured_values["is_stopped"] is True

    asyncio.run(_run())
