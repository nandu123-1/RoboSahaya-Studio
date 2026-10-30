"""Unit tests for Shared Task-Execution Contracts, Task Interpretation,
Capability-Based Skill Selection, Plan Invalidation, and UI Telemetry.

Tests:
1. Shared task-execution contracts (TaskSpecification, SkillAction, ActionResult, ObservationRecord, VerificationResult).
2. SimulatorTaskController contract conformance on SimulatorAdapter.
3. Ambiguous prompt detection across all robot domains.
4. Natural language task interpretation and entity grounding.
5. Capability-based skill selection and rejection of mismatched capabilities.
6. Deterministic plan fingerprinting and stale plan invalidation.
7. Honest readiness gating and limitation reporting.
"""
import pytest
from typing import Dict, Any

from robosahaya.evidence_models import TaskStage
from robosahaya.robot_registry import get_robot_registry
from robosahaya.simulators.registry import get_simulator_registry
from robosahaya.simulators.base import SimulatorAdapter
from robosahaya.tasks.capabilities import CapabilityType, RobotCapabilityResolver
from robosahaya.tasks.contracts import (
    ActionResult,
    ActionStep,
    ExecutionObservation,
    ObservationRecord,
    PlacementPredicate,
    SkillAction,
    SkillResult,
    SkillStatus,
    SimulatorTaskController,
    TaskExecutionState,
    TaskSpecification,
    TaskType,
    VerificationCondition,
    VerificationResult,
    VerificationVerdict,
    compute_task_fingerprint,
)
from robosahaya.tasks.interpreter import TaskInterpreter, get_task_interpreter
from robosahaya.tasks.skills import get_skill_registry


# =========================================================================
# 1. SHARED TASK-EXECUTION CONTRACTS TESTS
# =========================================================================

def test_task_specification_contract_creation():
    spec = TaskSpecification(
        task_id="task_test_001",
        original_instruction="Pick up the red cube and place it on the plate.",
        task_type=TaskType.MANIPULATION_PICK_PLACE,
        canonical_robot_id="franka_panda",
        simulator_adapter="gazebo_harmonic",
        world_id="tabletop_world",
        target_objects=["target_cube_red"],
        destination="target_bin_green",
        required_capabilities=["manipulation_arm", "manipulation_gripper"],
        action_sequence=[
            SkillAction(
                step_index=1,
                skill_id="move_joints",
                description="Approach target",
                parameters={"pose": "pre_grasp"},
                can_retry=True,
                max_retries=2,
            )
        ],
    )

    assert spec.task_id == "task_test_001"
    assert spec.task_type == TaskType.MANIPULATION_PICK_PLACE
    assert spec.canonical_robot_id == "franka_panda"
    assert len(spec.action_sequence) == 1
    assert spec.action_sequence[0].can_retry is True
    assert spec.action_sequence[0].max_retries == 2
    assert spec.is_supported is True
    assert spec.plan_status == "draft"

    # Serialization / Deserialization
    dump = spec.model_dump()
    assert dump["task_id"] == "task_test_001"
    reloaded = TaskSpecification.model_validate(dump)
    assert reloaded.task_id == spec.task_id


def test_action_result_and_observation_record_contracts():
    act_res = ActionResult(
        skill_id="move_joints",
        status=SkillStatus.SUCCEEDED,
        success=True,
        step_index=1,
        measured_data={"joint_state": "reached", "elapsed": 1.2},
        details="Arm reached target pose.",
        duration_sec=1.2,
        retry_count=0,
    )
    assert act_res.success is True
    assert act_res.status == SkillStatus.SUCCEEDED
    assert act_res.step_index == 1

    obs = ObservationRecord(
        step_index=1,
        skill_id="move_joints",
        timestamp=1234567.89,
        preconditions_met=True,
        commanded_parameters={"pose": "pre_grasp"},
        observed_state=act_res.measured_data,
        step_success=True,
        recovery_attempted=False,
        details=act_res.details,
    )
    assert obs.step_index == 1
    assert obs.step_success is True
    assert obs.observed_state["joint_state"] == "reached"


def test_verification_result_contract():
    verif = VerificationResult(
        task_id="task_test_002",
        verdict=VerificationVerdict.VERIFIED_SUCCESS,
        success=True,
        placement_satisfied=True,
        placement_details="Object in tolerance.",
        condition_results=[{"name": "Placement", "passed": True}],
        measured_values={"steps_completed": 10},
        summary="Task completed successfully.",
    )
    assert verif.verdict == VerificationVerdict.VERIFIED_SUCCESS
    assert verif.success is True
    assert verif.placement_satisfied is True


def test_simulator_adapter_implements_simulator_task_controller():
    sim_registry = get_simulator_registry()
    adapters = sim_registry.list_adapters()
    assert len(adapters) >= 2

    for adapter in adapters:
        assert isinstance(adapter, SimulatorTaskController)
        # Verify contract methods exist and are callable
        res = adapter.execute_skill("dummy_skill", {}, {})
        assert isinstance(res, ActionResult)
        obs = adapter.observe_state({}, {})
        assert isinstance(obs, dict)
        passed, msg = adapter.verify_predicate("dummy", {}, {})
        assert isinstance(passed, bool)
        assert isinstance(msg, str)


# =========================================================================
# 2. AMBIGUOUS PROMPTS DETECTION TESTS
# =========================================================================

def test_interpreter_detects_ambiguous_single_words():
    interpreter = get_task_interpreter()

    for vague in ("move", "drive", "pick", "fly", "walk", "run", "do task", "run fast"):
        spec = interpreter.interpret_instruction(vague)
        assert spec.is_supported is False, f"Expected '{vague}' to be rejected as ambiguous"
        assert "Ambiguous" in spec.rejection_reason or "too brief" in spec.rejection_reason
        assert spec.plan_status == "rejected"


def test_interpreter_detects_ambiguous_manipulation():
    interpreter = get_task_interpreter()

    vague_manip = [
        "pick something",
        "pick up something",
        "put it somewhere",
        "pick and place",
    ]
    for prompt in vague_manip:
        spec = interpreter.interpret_instruction(prompt)
        assert spec.is_supported is False
        assert "Ambiguous" in spec.rejection_reason
        assert spec.task_type == TaskType.MANIPULATION_PICK_PLACE


def test_interpreter_detects_ambiguous_wheel_motion():
    interpreter = get_task_interpreter()

    spec = interpreter.interpret_instruction("move forward")
    assert spec.is_supported is False
    assert "Ambiguous" in spec.rejection_reason
    assert "speed" in spec.rejection_reason.lower() or "velocity" in spec.rejection_reason.lower()


# =========================================================================
# 3. TASK PARSING ACROSS ROBOT FAMILIES
# =========================================================================

def test_interpreter_parses_rsk_wheel_velocity():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction("build rsk and it has move forward at 5.44")

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MOBILE_WHEEL_CONTROL
    assert spec.canonical_robot_id == "rsk"
    assert spec.simulator_adapter == "mujoco"
    assert len(spec.action_sequence) == 1
    action = spec.action_sequence[0]
    assert action.skill_id == "set_wheel_velocity"
    assert action.parameters.get("velocity") == 5.44
    assert len(spec.verification_conditions) == 1
    assert spec.verification_conditions[0].expected_value == 5.44


def test_interpreter_parses_so_101_pick_and_place():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction(
        "build so 101 arm and pick up a real simulated object and place it on a plate, verifying the final object position."
    )

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MANIPULATION_PICK_PLACE
    assert spec.canonical_robot_id == "so_arm101"
    assert spec.simulator_adapter == "gazebo_harmonic"
    assert spec.world_id == "tabletop_world"
    assert "target_cube_red" in spec.target_objects
    assert spec.destination == "target_bin_green"
    assert len(spec.action_sequence) == 10
    assert spec.placement_predicate is not None
    assert spec.placement_predicate.allowed_radius <= 0.10


def test_interpreter_parses_differential_drive_navigation():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction(
        "Move to the red box and stop 0.5 metres away"
    )

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MOBILE_NAVIGATION
    assert spec.canonical_robot_id == "differential_drive"
    assert spec.simulator_adapter == "gazebo_harmonic"
    assert len(spec.action_sequence) == 1
    assert spec.action_sequence[0].skill_id == "drive_to_pose"
    assert spec.action_sequence[0].parameters.get("stop_distance_m") == 0.5


# =========================================================================
# 4. CAPABILITY GATING & UNSUPPORTED ACTION REJECTION
# =========================================================================

def test_unsupported_action_diff_drive_cannot_pick_place():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction(
        instruction="Pick up the red cube and place it on the plate",
        robot_id_override="differential_drive",
    )

    assert spec.is_supported is False
    assert spec.canonical_robot_id == "differential_drive"
    assert "cannot execute this task" in spec.rejection_reason or "has no manipulator arm" in spec.rejection_reason


def test_unsupported_action_arm_cannot_drive_wheels():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction(
        instruction="Drive forward at 5.44 rad/s",
        robot_id_override="franka_panda",
    )

    assert spec.is_supported is False
    assert spec.canonical_robot_id == "franka_panda"
    assert "cannot execute this task" in spec.rejection_reason or "has no drive wheels" in spec.rejection_reason


def test_unsupported_action_drone_cannot_pick_place():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction(
        instruction="Pick up the red cube and place it on the plate",
        robot_id_override="skydio_x2",
    )

    assert spec.is_supported is False
    assert spec.canonical_robot_id == "skydio_x2"
    assert "cannot execute this task" in spec.rejection_reason


# =========================================================================
# 5. CAPABILITY-BASED SKILL SELECTION
# =========================================================================

def test_skill_registry_compatible_skills_for_arm():
    reg = get_robot_registry()
    skill_reg = get_skill_registry()

    panda = reg.get_model("franka_panda")
    assert panda is not None
    compat = skill_reg.get_compatible_skills(panda, "gazebo_harmonic")
    skill_ids = {s.skill_id for s in compat}

    assert "move_joints" in skill_ids
    assert "gripper_control" in skill_ids
    assert "pick_and_place" in skill_ids
    assert "set_wheel_velocity" not in skill_ids
    assert "drive_to_pose" not in skill_ids


def test_skill_registry_compatible_skills_for_rsk():
    reg = get_robot_registry()
    skill_reg = get_skill_registry()

    rsk = reg.get_model("rsk")
    assert rsk is not None
    compat = skill_reg.get_compatible_skills(rsk, "mujoco")
    skill_ids = {s.skill_id for s in compat}

    assert "set_wheel_velocity" in skill_ids
    assert "pick_and_place" not in skill_ids
    assert "move_joints" not in skill_ids


# =========================================================================
# 6. PLAN FINGERPRINTING & INVALIDATION
# =========================================================================

def test_plan_fingerprint_determinism_and_invalidation():
    interpreter = get_task_interpreter()
    spec1 = interpreter.interpret_instruction("build rsk and it has move forward at 5.44")
    spec2 = interpreter.interpret_instruction("build rsk and it has move forward at 5.44")

    assert spec1.plan_fingerprint != ""
    assert spec1.plan_fingerprint == spec2.plan_fingerprint

    # Drift checks
    assert spec1.is_invalidated_by(
        new_instruction="build rsk and it has move forward at 2.0",
        new_robot_id="rsk",
        new_world_id="soccer_pitch",
    ) is True

    assert spec1.is_invalidated_by(
        new_instruction="build rsk and it has move forward at 5.44",
        new_robot_id="differential_drive",
        new_world_id="soccer_pitch",
    ) is True

    assert spec1.is_invalidated_by(
        new_instruction="build rsk and it has move forward at 5.44",
        new_robot_id="rsk",
        new_world_id="tabletop_world",
    ) is True

    assert spec1.is_invalidated_by(
        new_instruction="build rsk and it has move forward at 5.44",
        new_robot_id="rsk",
        new_world_id="soccer_pitch",
    ) is False


# =========================================================================
# 7. HONEST READINESS & LIMITATION REPORTING
# =========================================================================

def test_unintegrated_robots_report_honest_limitations():
    interpreter = get_task_interpreter()

    # Skydio X2 drone flight
    drone_spec = interpreter.interpret_instruction("Fly drone along waypoints at 10m altitude.")
    assert drone_spec.is_supported is False
    assert len(drone_spec.known_limitations) > 0
    assert any("not integrated" in lim.lower() or "blocked" in lim.lower() for lim in drone_spec.known_limitations)

    # Unitree Go2 legged gait
    quad_spec = interpreter.interpret_instruction("Command quadruped robot to execute dynamic running trot.")
    assert quad_spec.is_supported is False
    assert len(quad_spec.known_limitations) > 0
    assert any("not integrated" in lim.lower() or "blocked" in lim.lower() for lim in quad_spec.known_limitations)
