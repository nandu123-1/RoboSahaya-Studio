"""Phase 5 Integration Tests: Physics-Faithful Controllers, Sensor-Driven Autonomy,
Cross-Simulator Robotics, and Reliable End-to-End Task Verification.
"""
import asyncio
import shutil
import pytest
from robosahaya.tasks.contracts import (
    TaskType,
    TaskExecutionState,
    TaskSpecification,
    SkillAction,
)
from robosahaya.tasks.interpreter import TaskInterpreter, get_task_interpreter
from robosahaya.simulators.task_execution import (
    SimulatorTaskExecutionAdapter,
    GazeboTaskExecutionAdapter,
    MujocoTaskExecutionAdapter,
)
from robosahaya.simulators.scenarios import (
    get_scenario,
    list_scenarios,
    ScenarioRunResult,
    evaluate_statistical_runs,
    ScenarioCategory,
)
from robosahaya.execution_matrix import (
    RobotExecutionMatrix,
    get_execution_matrix,
    SkillReadinessState,
    ExecutionReadiness,
)


def test_versioned_scenarios_and_evidence_invalidation():
    """Verify reusable versioned scenarios across 4 families and targeted evidence invalidation."""
    # 1. Scenarios registered
    scenarios = list_scenarios()
    assert len(scenarios) >= 4
    categories = {s.category for s in scenarios}
    assert ScenarioCategory.MOBILE_BASE in categories
    assert ScenarioCategory.QUADRUPED in categories
    assert ScenarioCategory.AERIAL_DRONE in categories
    assert ScenarioCategory.MANIPULATOR in categories

    go1_scen = get_scenario("go1_trot_stability_v1")
    assert go1_scen is not None
    assert go1_scen.robot_id == "unitree_go1"
    assert "min_displacement_m" in go1_scen.success_criteria

    x2_scen = get_scenario("skydio_x2_flight_v1")
    assert x2_scen is not None
    assert x2_scen.controller_settings["propulsion_model"] == "rotor_aerodynamics_mixer"

    # 2. Targeted evidence invalidation
    matrix = RobotExecutionMatrix()
    rec = matrix.build_record_for_robot("franka_panda")
    matrix.record_skill_verification("franka_panda", "move_joints", simulator="mujoco", success=True)
    assert "move_joints" in rec.verified_skills

    # Invalidate ONLY skills dimension
    invalidated = matrix.invalidate_evidence("franka_panda", dimensions=["skills"], reason="Controller refactored")
    assert len(invalidated.verified_skills) == 0
    # Model load and physics evidence remain intact
    assert invalidated.model_load_status != "untested"
    assert any("Evidence invalidated (skills)" in b for b in invalidated.exact_blockers)


def test_quadruped_closed_loop_trot_stability_and_transitions():
    """Verify Go1 closed-loop trot with roll/pitch regulation and stopping transition."""
    spec = TaskSpecification(
        task_id="go1_closed_loop_01",
        original_instruction="Make Unitree Go1 walk forward 1.0m with closed-loop stabilization",
        task_type=TaskType.LEGGED_LOCOMOTION,
        canonical_robot_id="unitree_go1",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="legged_gait",
                description="Closed-loop trot locomotion with roll/pitch regulation",
                parameters={
                    "gait": "trot",
                    "duration_sec": 2.0,
                    "speed_m_s": 0.5,
                    "frequency_hz": 1.5,
                    "mode": "walking",
                },
            )
        ],
    )
    adapter = SimulatorTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))

    assert res["success"] is True
    assert res["state"] == TaskExecutionState.COMPLETED.value
    meas = res["action_results"][0].measured_data
    assert meas["displacement_m"] >= 0.35
    assert meas["closed_loop_stabilization"] is True
    assert meas["max_abs_roll_rad"] < 0.35
    assert meas["max_abs_pitch_rad"] < 0.35
    assert meas["foot_contact_ratio"] > 0.0
    assert meas["control_effort"] > 0.0


def test_quadruped_turning_and_standing_modes():
    """Verify Go1 standing posture and yaw turning locomotion modes."""
    adapter = SimulatorTaskExecutionAdapter()

    # 1. Standing mode
    spec_stand = TaskSpecification(
        task_id="go1_stand_01",
        original_instruction="Unitree Go1 hold nominal standing posture",
        task_type=TaskType.LEGGED_LOCOMOTION,
        canonical_robot_id="unitree_go1",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="legged_gait",
                description="Hold standing posture",
                parameters={"mode": "standing", "duration_sec": 1.0, "speed_m_s": 0.0},
            )
        ],
    )
    res_stand = asyncio.run(adapter.execute_task(spec_stand, context={"headless": True}))
    assert res_stand["success"] is True
    assert res_stand["action_results"][0].measured_data["displacement_m"] == 0.0

    # 2. Turning mode
    spec_turn = TaskSpecification(
        task_id="go1_turn_01",
        original_instruction="Unitree Go1 trot and turn at 0.4 rad/s",
        task_type=TaskType.LEGGED_LOCOMOTION,
        canonical_robot_id="unitree_go1",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="legged_gait",
                description="Turn while trotting",
                parameters={
                    "mode": "walking",
                    "duration_sec": 1.5,
                    "speed_m_s": 0.3,
                    "turn_rate_rad_s": 0.4,
                },
            )
        ],
    )
    res_turn = asyncio.run(adapter.execute_task(spec_turn, context={"headless": True}))
    assert res_turn["success"] is True
    assert res_turn["action_results"][0].measured_data["turning_verified"] is True


def test_quadruped_dynamic_disturbance_recovery():
    """Verify Go1 recovers stability after a transient external perturbation force."""
    spec = TaskSpecification(
        task_id="go1_disturb_01",
        original_instruction="Unitree Go1 recover from lateral disturbance",
        task_type=TaskType.LEGGED_LOCOMOTION,
        canonical_robot_id="unitree_go1",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="legged_gait",
                description="Recover from lateral disturbance",
                parameters={
                    "duration_sec": 2.5,
                    "speed_m_s": 0.4,
                    "apply_disturbance": True,
                    "disturbance_force_n": 15.0,
                },
            )
        ],
    )
    adapter = SimulatorTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))
    assert res["success"] is True
    meas = res["action_results"][0].measured_data
    assert meas["disturbance_applied"] is True
    assert meas["disturbance_recovered"] is True
    assert meas["recovery_time_sec"] <= 0.8


def test_quadruped_go2_honest_gating_preserved():
    """Verify Unitree Go2 remains gated for locomotion when whole-body controller is unconfigured."""
    spec = TaskSpecification(
        task_id="go2_gate_01",
        original_instruction="Command Unitree Go2 to execute dynamic running trot.",
        task_type=TaskType.LEGGED_LOCOMOTION,
        canonical_robot_id="unitree_go2",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="legged_gait",
                description="Attempt Go2 legged trot gait",
                parameters={},
            )
        ],
    )
    adapter = SimulatorTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))
    assert res["success"] is False
    assert res["state"] == TaskExecutionState.BLOCKED.value
    assert "whole-body controller" in res["error"].lower() or "wbc" in res["error"].lower()


def test_drone_flight_physical_rotor_mixer():
    """Verify Skydio X2 flight dynamics using genuine physical 4-rotor aerodynamics mixer."""
    spec = TaskSpecification(
        task_id="x2_mixer_01",
        original_instruction="Fly Skydio X2 drone with flight controller to 1.5m altitude",
        task_type=TaskType.DRONE_FLIGHT,
        canonical_robot_id="skydio_x2",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="drone_flight",
                description="Fly Skydio X2 with 4-rotor aerodynamics mixer",
                parameters={
                    "altitude_m": 1.5,
                    "duration_sec": 3.0,
                    "propulsion_model": "rotor_aerodynamics_mixer",
                    "waypoint": [0.4, 0.0],
                },
            )
        ],
    )
    adapter = SimulatorTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True, "has_flight_controller": True}))
    assert res["success"] is True
    meas = res["action_results"][0].measured_data
    assert meas["propulsion_model"] == "rotor_aerodynamics_mixer"
    assert meas["peak_altitude_m"] >= 1.2
    assert meas["hover_stable"] is True
    assert len(meas["rotor_thrusts_n"]) == 4
    assert all(t > 0.5 for t in meas["rotor_thrusts_n"])


def test_drone_flight_motor_saturation_detection():
    """Verify motor saturation detection under extreme flight climb commands."""
    spec = TaskSpecification(
        task_id="x2_sat_01",
        original_instruction="Climb aggressively to 10m in 1s",
        task_type=TaskType.DRONE_FLIGHT,
        canonical_robot_id="skydio_x2",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="drone_flight",
                description="Fly aggressive climb to test saturation",
                parameters={
                    "altitude_m": 10.0,
                    "duration_sec": 1.0,
                    "propulsion_model": "rotor_aerodynamics_mixer",
                },
            )
        ],
    )
    adapter = SimulatorTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True, "has_flight_controller": True}))
    meas = res["action_results"][0].measured_data
    assert meas["motor_saturation_detected"] is True
    assert meas["finite_state"] is True


def test_gazebo_harmonic_so_arm101_manipulator_execution():
    """Verify real Gazebo Harmonic JointPositionController execution for SO-101 arm."""
    gz = shutil.which("gz")
    if gz is None:
        pytest.skip("Gazebo Sim executable 'gz' is not installed.")

    spec = TaskSpecification(
        task_id="gz_so101_01",
        original_instruction="Move so_arm101 joint 1 to 0.4 rad in Gazebo",
        task_type=TaskType.MANIPULATION_TRAJECTORY,
        canonical_robot_id="so_arm101",
        robot_model_version="v1.0.0",
        simulator_adapter="gazebo_harmonic",
        world_id="tabletop_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="move_joints",
                description="Move SO-101 arm joints in Gazebo",
                parameters={"joint_positions": {"joint_1": 0.4, "joint_2": 0.2}, "duration_sec": 1.5},
            )
        ],
    )
    adapter = GazeboTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))

    assert res["success"] is True
    assert res["state"] == TaskExecutionState.COMPLETED.value
    meas = res["action_results"][0].measured_data
    assert meas["motion_verified"] is True
    assert meas["controller_type"] == "gz::sim::systems::JointPositionController"
    assert meas["simulated_iterations"] >= 100


def test_perception_failure_safety_gate():
    """Verify natural-language interpreter stops safely before blind execution when target is ungrounded."""
    interpreter = get_task_interpreter()
    spec = interpreter.interpret_instruction(
        "Pick up the purple pyramid from tabletop",
        robot_id_override="franka_panda",
        world_id_override="tabletop_world",
    )
    # The purple pyramid does not exist in tabletop_world
    assert spec.is_supported is False
    assert "could not be found" in spec.rejection_reason or "target object" in spec.rejection_reason.lower()


def test_hardened_pick_and_place_transparent_grasp_metrics():
    """Verify pick-and-place reports transparent kinematic grasp abstraction and collision clearance metrics."""
    spec = TaskSpecification(
        task_id="pick_place_metrics_01",
        original_instruction="Pick up red cube and place it into target bin",
        task_type=TaskType.MANIPULATION_PICK_PLACE,
        canonical_robot_id="franka_panda",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="tabletop_world",
        target_objects=["target_cube_red"],
        destination="target_bin_green",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="pick_and_place",
                description="Pick and place red cube with transparent grasp metrics",
                parameters={
                    "target_object_id": "target_cube_red",
                    "destination_id": "target_bin_green",
                    "cube_initial_xyz": [0.45, -0.05, 0.035],
                    "destination_xyz": [0.30, -0.20, 0.035],
                },
            )
        ],
    )
    adapter = SimulatorTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))

    assert res["success"] is True
    meas = res["action_results"][0].measured_data
    assert meas["grasp_model"] == "kinematic_grasp_constraint"
    assert meas["min_collision_clearance_m"] > 0.05
    assert meas["lift_displacement_m"] > 0.02
    assert meas["placement_error_m"] <= 0.12
    assert meas["release_settled"] is True
    assert meas["slip_detected"] is False


def test_statistical_scenario_evaluation():
    """Verify statistical scenario multi-run evaluation and tolerance rejection."""
    results = [
        ScenarioRunResult(
            run_id=f"run_{i}",
            scenario_id="test_scen",
            scenario_version="1.0.0",
            passed=(i != 0),
            initial_state_actual={"x": 0.0},
            measured_metrics={"displacement_m": 0.95 + 0.02 * i, "roll_rad": 0.02},
            violations=["lateral deviation"] if i == 0 else [],
            trajectory_samples_count=50,
            duration_simulated_sec=2.0,
            evidence_recorded={},
        )
        for i in range(5)
    ]
    # 4 passed out of 5 -> 80% success rate
    stat = evaluate_statistical_runs("test_scen", results, min_required_success_rate=0.75)
    assert stat.total_runs == 5
    assert stat.successful_runs == 4
    assert stat.success_rate == 0.8
    assert stat.all_runs_passed is True
    assert "displacement_m" in stat.mean_metrics
    assert stat.mean_metrics["displacement_m"] > 0.9
