"""Unit and integration tests for the Universal Robot Task Execution Engine.

Tests:
1. Structured task interpretation and grounding across robot families
2. Placement predicate evaluation with spatial bounds
3. Robot capability resolution and rejection of mismatched capabilities
4. Skill registry parameter validation and execution
5. Closed-loop state machine execution with observations and retries
6. Real MuJoCo simulation execution of RSK wheel control (wheel1_speed = 5.44)
7. Honest rejection of unintegrated capabilities (drones, legged gaits)
"""
import asyncio
from pathlib import Path
import pytest

from robosahaya.evidence_models import TaskStage
from robosahaya.orchestrator import TaskOrchestrator
from robosahaya.robot_registry import get_robot_registry
from robosahaya.task_store import TaskStore
from robosahaya.tasks.capabilities import CapabilityType, RobotCapabilityResolver
from robosahaya.tasks.executor import ClosedLoopExecutor, TaskExecutionState
from robosahaya.tasks.interpreter import TaskInterpreter, get_task_interpreter
from robosahaya.tasks.scene import SceneEntity, WorldScene, get_world_scene
from robosahaya.tasks.skills import (
    GripperControlSkill,
    MoveJointsSkill,
    PickAndPlaceSkill,
    SetWheelVelocitySkill,
    SkillStatus,
    get_skill_registry,
)
from robosahaya.tasks.specification import (
    ActionStep,
    PlacementPredicate,
    TaskSpecification,
    TaskType,
    VerificationCondition,
)
from robosahaya.workspace_manager import WorkspaceManager


# =========================================================================
# 1. TASK INTERPRETER & GROUNDING TESTS
# =========================================================================

def test_interpreter_pick_and_place_tabletop():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction("Pick up the red cube in front of the arm and place it on the plate.")

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MANIPULATION_PICK_PLACE
    assert spec.canonical_robot_id in ("franka_panda", "robotic_arm_6dof")
    assert spec.world_id == "tabletop_world"
    assert "target_cube_red" in spec.target_objects
    assert spec.destination == "target_bin_green"
    assert len(spec.action_sequence) >= 6
    assert spec.placement_predicate is not None
    assert spec.placement_predicate.target_entity_id == "target_cube_red"
    assert spec.placement_predicate.destination_entity_id == "target_bin_green"


def test_interpreter_rsk_wheel_velocity():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction("Set wheel1_speed to 5.44 rad/s on the RSK soccer robot.")

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MOBILE_WHEEL_CONTROL
    assert spec.canonical_robot_id == "rsk"
    assert spec.simulator_adapter == "mujoco"
    assert len(spec.action_sequence) == 1
    step = spec.action_sequence[0]
    assert step.skill_id == "set_wheel_velocity"
    assert step.parameters.get("wheel_name") == "wheel1_speed"
    assert step.parameters.get("velocity") == 5.44


def test_user_query_rsk_move_forward_at_5_44():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction("build rsk and it has move forward at 5.44")

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MOBILE_WHEEL_CONTROL
    assert spec.canonical_robot_id == "rsk"
    assert spec.simulator_adapter == "mujoco"
    assert len(spec.action_sequence) == 1
    step = spec.action_sequence[0]
    assert step.skill_id == "set_wheel_velocity"
    assert step.parameters.get("velocity") == 5.44


def test_user_query_so101_pick_and_place_on_plate():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction(
        "build so 101 arm and pick up a real simulated object and place it on a plate, verifying the final object position."
    )

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MANIPULATION_PICK_PLACE
    assert spec.canonical_robot_id == "so_arm101"
    assert spec.world_id == "tabletop_world"
    assert spec.target_objects == ["target_cube_red"]
    assert spec.destination == "target_bin_green"
    assert len(spec.action_sequence) == 10
    assert spec.placement_predicate is not None
    assert spec.placement_predicate.target_entity_id == "target_cube_red"
    assert spec.placement_predicate.destination_entity_id == "target_bin_green"


def test_interpreter_mobile_navigation():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction("Drive to the red box and stop 0.5 metres away.")

    assert spec.is_supported is True
    assert spec.task_type == TaskType.MOBILE_NAVIGATION
    assert spec.canonical_robot_id == "differential_drive"
    assert spec.world_id == "obstacle_world"
    assert len(spec.action_sequence) >= 1
    assert spec.action_sequence[0].skill_id == "drive_to_pose"


def test_interpreter_honest_rejection_drone_flight():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction("Fly drone quadcopter along waypoints at 10m altitude.")

    assert spec.is_supported is False
    assert spec.task_type == TaskType.DRONE_FLIGHT
    assert "cannot execute this task" in spec.rejection_reason or "No compatible robot" in spec.rejection_reason
    assert any("flight" in lim.lower() or "px4" in lim.lower() or "ground vehicle" in lim.lower() or "drone" in lim.lower() for lim in spec.known_limitations)


def test_interpreter_honest_rejection_legged_locomotion():
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction("Command quadruped robot to execute dynamic running trot.")

    assert spec.is_supported is False
    assert spec.task_type == TaskType.LEGGED_LOCOMOTION
    assert "cannot execute this task" in spec.rejection_reason or "No compatible robot" in spec.rejection_reason
    assert any("legged" in lim.lower() or "wbc" in lim.lower() for lim in spec.known_limitations)


# =========================================================================
# 2. PLACEMENT PREDICATE TESTS
# =========================================================================

def test_placement_predicate_success():
    predicate = PlacementPredicate(
        target_entity_id="target_cube_red",
        destination_entity_id="target_bin_green",
        target_center_xyz=(0.25, 0.30, 0.805),
        allowed_radius=0.08,
        min_height=0.78,
        max_height=0.90,
    )
    # Target placed 2cm from center at height 0.825m
    passed, msg = predicate.evaluate((0.26, 0.31, 0.825))
    assert passed is True
    assert "within" in msg.lower()


def test_placement_predicate_fail_radial_distance():
    predicate = PlacementPredicate(
        target_entity_id="target_cube_red",
        destination_entity_id="target_bin_green",
        target_center_xyz=(0.25, 0.30, 0.805),
        allowed_radius=0.08,
        min_height=0.78,
        max_height=0.90,
    )
    # Placed at pick pose (0.45, 0.15, 0.825), >0.20m away from plate
    passed, msg = predicate.evaluate((0.45, 0.15, 0.825))
    assert passed is False
    assert "exceeds" in msg.lower()


def test_placement_predicate_fail_height_bounds():
    predicate = PlacementPredicate(
        target_entity_id="target_cube_red",
        destination_entity_id="target_bin_green",
        target_center_xyz=(0.25, 0.30, 0.805),
        allowed_radius=0.08,
        min_height=0.78,
        max_height=0.90,
    )
    # Fallen on the floor (z = 0.05m)
    passed, msg = predicate.evaluate((0.25, 0.30, 0.05))
    assert passed is False
    assert "outside" in msg.lower() or "height" in msg.lower() or "range" in msg.lower()


# =========================================================================
# 3. ROBOT CAPABILITY RESOLVER TESTS
# =========================================================================

def test_capability_resolver_arms():
    resolver = RobotCapabilityResolver()
    caps = resolver.get_capabilities("franka_panda")

    assert caps[CapabilityType.MANIPULATION_ARM].supported is True
    assert caps[CapabilityType.MANIPULATION_GRIPPER].supported is True
    assert caps[CapabilityType.MOBILE_WHEEL_VELOCITY].supported is False
    assert caps[CapabilityType.DRONE_FLIGHT].supported is False

    can_do, sat, missing = resolver.check_capabilities(
        "franka_panda",
        [CapabilityType.MANIPULATION_ARM, CapabilityType.MANIPULATION_GRIPPER]
    )
    assert can_do is True
    assert len(missing) == 0


def test_capability_resolver_rsk():
    resolver = RobotCapabilityResolver()
    caps = resolver.get_capabilities("rsk")

    assert caps[CapabilityType.MOBILE_WHEEL_VELOCITY].supported is True
    assert caps[CapabilityType.MANIPULATION_ARM].supported is False
    assert caps[CapabilityType.MANIPULATION_GRIPPER].supported is False

    can_do, sat, missing = resolver.check_capabilities(
        "rsk",
        [CapabilityType.MANIPULATION_ARM]
    )
    assert can_do is False
    assert len(missing) == 1
    assert "manipulator arm" in missing[0].lower() or "manipulation_arm" in missing[0].lower()


# =========================================================================
# 4. SKILL REGISTRY & PARAMETER VALIDATION TESTS
# =========================================================================

def test_skill_registry_registration():
    reg = get_skill_registry()
    assert reg.get_skill("set_wheel_velocity") is not None
    assert reg.get_skill("pick_and_place") is not None
    assert reg.get_skill("move_joints") is not None
    assert reg.get_skill("gripper_control") is not None
    assert reg.get_skill("drive_to_pose") is not None


def test_wheel_velocity_parameter_validation():
    skill = SetWheelVelocitySkill()

    # Valid parameter
    skill.validate_parameters({"velocity": 5.44, "duration_sec": 0.5})

    # Missing velocity
    with pytest.raises(ValueError, match="Parameter 'velocity' is required"):
        skill.validate_parameters({})

    # Velocity exceeding safe bounds
    with pytest.raises(ValueError, match="exceeds safety limits"):
        skill.validate_parameters({"velocity": 50.0})


def test_gripper_control_parameter_validation():
    skill = GripperControlSkill()

    # Valid
    skill.validate_parameters({"action": "open"})
    skill.validate_parameters({"action": "close"})

    # Invalid
    with pytest.raises(ValueError, match="must be 'open', 'close'"):
        skill.validate_parameters({"action": "twist"})


# =========================================================================
# 5. CLOSED-LOOP EXECUTOR & RETRY LOGIC TESTS
# =========================================================================

def test_closed_loop_executor_step_by_step():
    executor = ClosedLoopExecutor(max_retries=2)
    spec = TaskSpecification(
        task_id="test_exec_01",
        original_instruction="Test pick and place steps",
        task_type=TaskType.MANIPULATION_PICK_PLACE,
        canonical_robot_id="franka_panda",
        action_sequence=[
            ActionStep(
                step_index=1,
                skill_id="move_joints",
                description="Move arm to pre-grasp pose",
                parameters={"pose": "pre_grasp"},
                is_critical=False,
            ),
            ActionStep(
                step_index=2,
                skill_id="gripper_control",
                description="Open gripper",
                parameters={"action": "open"},
                is_critical=True,
            ),
        ],
    )

    events = []

    async def _run():
        return await executor.execute_task(
            spec,
            context={"task_id": "test_exec_01"},
            on_event_cb=lambda evt: events.append(evt)
        )

    res = asyncio.run(_run())

    assert res["success"] is True
    assert res["state"] == TaskExecutionState.COMPLETED.value
    assert len(res["observations"]) == 2
    assert all(o.step_success for o in res["observations"])
    assert len(events) >= 2


# =========================================================================
# 6. REAL MUJOCO SIMULATION EXECUTION (RSK wheel1_speed = 5.44)
# =========================================================================

def test_real_mujoco_rsk_wheel_velocity_execution():
    """Verify that SetWheelVelocitySkill actually executes against the real MuJoCo physics engine."""
    import mujoco

    # Look for cached RSK model
    rsk_paths = [
        Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie" / "robot_soccer_kit" / "robot_soccer_kit.xml",
        Path.home() / ".cache" / "robosahaya" / "robot_models" / "rsk" / "v1.0.0" / "model" / "robot_soccer_kit.xml",
    ]
    model_path = next((p for p in rsk_paths if p.exists()), None)
    if not model_path:
        pytest.skip("RSK MuJoCo model not cached locally.")

    # Load model and initialize simulation data
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)

    skill = SetWheelVelocitySkill()
    res = skill.execute(
        params={"wheel_name": "wheel1_speed", "velocity": 5.44, "duration_sec": 0.2},
        context={"mujoco_model": m, "mujoco_data": d, "model_path": str(model_path)},
    )

    assert res.success is True
    assert res.status == SkillStatus.SUCCEEDED
    assert res.measured_data["actuator_name"] == "wheel1_speed"
    assert res.measured_data["commanded_rad_s"] == 5.44
    assert res.measured_data["steps_simulated"] > 0
    # Verified physical motion in MuJoCo
    assert res.measured_data["motion_verified"] is True
    assert abs(res.measured_data["final_actuator_ctrl"] - 5.44) < 1e-4


# =========================================================================
# 7. ORCHESTRATOR INTEGRATION WITH TASK SPECIFICATION
# =========================================================================

def test_orchestrator_formulates_structured_task_plan(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    templates_dir = repo_root / "simulation" / "templates"
    store = TaskStore(tmp_path / "tasks")
    ws_mgr = WorkspaceManager(base_workspace_root=tmp_path / "ws", templates_dir=templates_dir)
    orch = TaskOrchestrator(task_store=store, workspace_manager=ws_mgr, mode="mock")

    plan = orch.formulate_plan(
        "task_arm_plan",
        "Pick up the red cube and place it on the plate."
    )

    assert plan.is_supported is True
    assert plan.robot_template in ("robotic_arm_6dof", "franka_panda")
    assert plan.world_template == "tabletop_world"
    assert plan.task_spec is not None
    assert plan.task_spec["task_type"] == "manipulation_pick_place"
    assert "target_cube_red" in plan.task_spec["target_objects"]
    assert plan.task_spec["destination"] == "target_bin_green"
    assert len(plan.action_steps) >= 6
