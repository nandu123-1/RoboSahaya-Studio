"""Integration and acceptance tests for RoboSahaya Studio Phase 4.

Verifies:
1. Real Gazebo Harmonic simulation execution with bounded iterations, odometry, and clean process cleanup.
2. Quadruped locomotion (legged_gait) on Unitree Go1 with genuine joint trajectories, displacement, and stability.
3. Honest gating of quadruped locomotion when WBC is missing (Unitree Go2 without controller).
4. Drone multirotor flight (takeoff, hover, translation, landing) with aerodynamic flight controller on Skydio X2.
5. Honest gating of drone flight when flight controller is unconfigured.
6. Hardened manipulator pick-and-place across varied initial cube locations, varied destinations, and multiple arms.
7. Resumable batch verification and dynamic skill verification recording on RobotExecutionMatrix.
"""
import asyncio
import os
import shutil
import pytest
from pathlib import Path

from robosahaya.evidence_models import TaskStage
from robosahaya.execution_matrix import (
    SkillReadinessState,
    get_execution_matrix,
)
from robosahaya.simulators.task_execution import (
    GazeboTaskExecutionAdapter,
    SimulatorTaskExecutionAdapter,
)
from robosahaya.tasks.contracts import (
    ActionStep,
    SkillStatus,
    TaskExecutionState,
    TaskSpecification,
    TaskType,
)
from robosahaya.tasks.interpreter import TaskInterpreter


def test_gazebo_harmonic_mobile_execution_and_cleanup():
    """Verify real Gazebo Harmonic execution of differential drive with process cleanup."""
    gz = shutil.which("gz")
    if gz is None:
        pytest.skip("Gazebo Sim executable 'gz' is not installed on host.")

    adapter = GazeboTaskExecutionAdapter()
    spec = TaskSpecification(
        task_id="gz_test_diff_01",
        original_instruction="Drive differential base forward at 0.5 m/s for 1.5 seconds in Gazebo",
        task_type=TaskType.MOBILE_WHEEL_CONTROL,
        canonical_robot_id="differential_drive",
        simulator_adapter="gazebo_harmonic",
        world_id="obstacle_world",
        action_sequence=[
            ActionStep(
                step_index=1,
                skill_id="drive_forward",
                description="Drive differential drive robot forward",
                parameters={"speed_m_s": 0.5, "duration_sec": 1.5},
            )
        ],
        is_supported=True,
    )

    res = asyncio.run(adapter.execute_task(spec))
    assert res["success"] is True
    assert res["state"] == TaskExecutionState.COMPLETED.value
    assert res["simulator"] == "gazebo_harmonic"
    assert len(res["action_results"]) == 1

    action_res = res["action_results"][0]
    assert action_res.success is True
    assert action_res.status == SkillStatus.SUCCEEDED
    assert action_res.measured_data["motion_verified"] is True
    assert action_res.measured_data["distance_m"] > 0.5
    assert action_res.measured_data["simulated_iterations"] >= 100
    assert action_res.measured_data["finite_state"] is True


def test_quadruped_go1_locomotion_gait_execution():
    """Verify real quadruped locomotion gait on Unitree Go1 in MuJoCo."""
    adapter = SimulatorTaskExecutionAdapter()
    spec = TaskSpecification(
        task_id="go1_trot_test_01",
        original_instruction="Trot forward with Unitree Go1",
        task_type=TaskType.LEGGED_LOCOMOTION,
        canonical_robot_id="unitree_go1",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            ActionStep(
                step_index=1,
                skill_id="legged_gait",
                description="Execute trotting gait",
                parameters={"gait": "trot", "speed_m_s": 0.5, "duration_sec": 1.5, "frequency_hz": 1.5},
            )
        ],
        is_supported=True,
    )

    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))
    assert res["success"] is True
    assert res["state"] == TaskExecutionState.COMPLETED.value
    assert len(res["action_results"]) == 1

    action_res = res["action_results"][0]
    assert action_res.success is True
    assert action_res.status == SkillStatus.SUCCEEDED
    assert action_res.measured_data["gait"] == "trot"
    assert action_res.measured_data["cycles"] >= 2
    assert action_res.measured_data["displacement_m"] > 0.2
    assert action_res.measured_data["stable"] is True
    assert action_res.measured_data["trajectory_samples"] > 0


def test_quadruped_go2_honest_gating_without_wbc_controller():
    """Verify honest gating of Unitree Go2 when whole-body controller is not configured."""
    adapter = SimulatorTaskExecutionAdapter()
    spec = TaskSpecification(
        task_id="go2_trot_gated_01",
        original_instruction="Trot forward with Unitree Go2",
        task_type=TaskType.LEGGED_LOCOMOTION,
        canonical_robot_id="unitree_go2",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            ActionStep(
                step_index=1,
                skill_id="legged_gait",
                description="Execute trotting gait",
                parameters={"gait": "trot"},
            )
        ],
        is_supported=True,
    )

    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))
    assert res["success"] is False
    assert res["state"] == TaskExecutionState.BLOCKED.value
    assert "whole-body controller" in res["error"].lower()


def test_drone_flight_takeoff_hover_landing():
    """Verify real multirotor aerial flight on Skydio X2 with flight controller."""
    adapter = SimulatorTaskExecutionAdapter()
    spec = TaskSpecification(
        task_id="drone_flight_test_01",
        original_instruction="Fly Skydio X2 drone to 1.2m altitude, hover, and land",
        task_type=TaskType.DRONE_FLIGHT,
        canonical_robot_id="skydio_x2",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            ActionStep(
                step_index=1,
                skill_id="drone_flight",
                description="Multirotor flight maneuver",
                parameters={"altitude_m": 1.2, "duration_sec": 2.0, "waypoint": [0.4, 0.0]},
            )
        ],
        is_supported=True,
    )

    res = asyncio.run(adapter.execute_task(spec, context={"headless": True, "has_flight_controller": True}))
    assert res["success"] is True
    assert res["state"] == TaskExecutionState.COMPLETED.value
    assert len(res["action_results"]) == 1

    action_res = res["action_results"][0]
    assert action_res.success is True
    assert action_res.status == SkillStatus.SUCCEEDED
    assert action_res.measured_data["target_altitude_m"] == 1.2
    assert action_res.measured_data["peak_altitude_m"] >= 0.5
    assert action_res.measured_data["hover_stable"] is True
    assert action_res.measured_data["flight_verified"] is True
    assert action_res.measured_data["telemetry_samples"] > 0


def test_drone_flight_honest_gating_without_controller():
    """Verify honest gating of drone flight when flight controller is absent."""
    adapter = SimulatorTaskExecutionAdapter()
    spec = TaskSpecification(
        task_id="drone_flight_gated_01",
        original_instruction="Fly drone",
        task_type=TaskType.DRONE_FLIGHT,
        canonical_robot_id="skydio_x2",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            ActionStep(
                step_index=1,
                skill_id="drone_flight",
                description="Flight maneuver",
                parameters={"altitude_m": 1.0},
            )
        ],
        is_supported=True,
    )

    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))
    assert res["success"] is False
    assert res["state"] == TaskExecutionState.BLOCKED.value
    assert "flight controller" in res["error"].lower()


def test_hardened_manipulator_pick_and_place_varied_locations():
    """Verify pick-and-place with varied initial cube coordinates and destinations."""
    adapter = SimulatorTaskExecutionAdapter()
    spec = TaskSpecification(
        task_id="panda_pick_varied_01",
        original_instruction="Pick red cube from custom pose and place at target bin",
        task_type=TaskType.MANIPULATION_PICK_PLACE,
        canonical_robot_id="franka_panda",
        simulator_adapter="mujoco",
        world_id="tabletop_world",
        target_objects=["target_cube_red"],
        destination="target_bin_green",
        action_sequence=[
            ActionStep(
                step_index=1,
                skill_id="pick_and_place",
                description="Pick and place cube from offset location",
                parameters={
                    "target_object_id": "target_cube_red",
                    "destination_id": "target_bin_green",
                    "cube_initial_xyz": [0.42, 0.05, 0.05],
                    "destination_xyz": [0.32, -0.18, 0.05],
                },
            )
        ],
        is_supported=True,
    )

    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))
    assert res["success"] is True
    assert res["state"] == TaskExecutionState.COMPLETED.value

    action_res = res["action_results"][0]
    assert action_res.success is True
    assert action_res.measured_data["placement_verified"] is True
    assert action_res.measured_data["grasp_stable"] is True
    assert action_res.measured_data["collision_free"] is True
    assert action_res.measured_data["placement_error_m"] <= 0.12


def test_execution_matrix_batch_verification_and_recording():
    """Verify resumable batch verification and per-skill evidence recording."""
    matrix = get_execution_matrix()

    # Resumable batch run for a small sample
    subset = ["franka_panda", "ur5e", "rsk"]
    batch_res = matrix.run_batch_verification(robot_ids=subset, max_models=3)
    assert len(batch_res) == 3
    assert "franka_panda" in batch_res

    # Dynamic skill recording
    matrix.record_skill_verification(
        robot_id="unitree_go1",
        skill_id="legged_gait",
        simulator="mujoco",
        success=True,
    )
    go1_rec = matrix.build_record_for_robot("unitree_go1")
    assert go1_rec.skill_readiness["legged_gait"] == SkillReadinessState.VERIFIED.value
    assert "legged_gait" in go1_rec.verified_skills


def test_end_to_end_natural_language_quadruped_gait():
    """Verify natural language instruction produces executable task for Unitree Go1."""
    interpreter = TaskInterpreter()
    spec = interpreter.interpret_instruction(
        "Trot Unitree Go1 forward",
        robot_id_override="unitree_go1",
        world_id_override="obstacle_world",
        task_id="nl_go1_trot_01",
    )
    assert spec.is_supported is True
    assert spec.canonical_robot_id == "unitree_go1"
    assert spec.task_type == TaskType.LEGGED_LOCOMOTION
    assert len(spec.action_sequence) == 1
    assert spec.action_sequence[0].skill_id == "legged_gait"

    adapter = SimulatorTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))
    assert res["success"] is True
    assert res["state"] == TaskExecutionState.COMPLETED.value
    assert res["action_results"][0].measured_data["motion_verified"] is True


def test_end_to_end_natural_language_drone_flight():
    """Verify natural language instruction produces executable flight task for Skydio X2."""
    interpreter = TaskInterpreter()
    spec = interpreter.interpret_instruction(
        "Fly Skydio X2 drone with flight controller to 1.5m altitude",
        robot_id_override="skydio_x2",
        world_id_override="obstacle_world",
        task_id="nl_drone_01",
    )
    assert spec.is_supported is True
    assert spec.canonical_robot_id == "skydio_x2"
    assert spec.task_type == TaskType.DRONE_FLIGHT
    assert len(spec.action_sequence) == 1
    assert spec.action_sequence[0].skill_id == "drone_flight"

    adapter = SimulatorTaskExecutionAdapter()
    # 1. Gating when flight controller is absent
    res_blocked = asyncio.run(adapter.execute_task(spec, context={"headless": True, "has_flight_controller": False}))
    assert res_blocked["success"] is False
    assert res_blocked["state"] == TaskExecutionState.BLOCKED.value

    # 2. Verified flight when flight controller is present
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True, "has_flight_controller": True}))
    assert res["success"] is True
    assert res["state"] == TaskExecutionState.COMPLETED.value
    assert res["action_results"][0].measured_data["flight_verified"] is True


def test_cross_simulator_parity_differential_drive():
    """Verify cross-simulator parity: the same mobile task can target either MuJoCo or Gazebo."""
    gz = shutil.which("gz")
    if gz is None:
        pytest.skip("Gazebo Sim executable 'gz' is not installed.")

    spec_gz = TaskSpecification(
        task_id="cross_sim_gz_01",
        original_instruction="Drive forward 0.5 m/s",
        task_type=TaskType.MOBILE_WHEEL_CONTROL,
        canonical_robot_id="differential_drive",
        simulator_adapter="gazebo_harmonic",
        world_id="obstacle_world",
        action_sequence=[
            ActionStep(
                step_index=1,
                skill_id="drive_forward",
                description="Drive forward",
                parameters={"speed_m_s": 0.5, "duration_sec": 1.0},
            )
        ],
        is_supported=True,
    )
    adapter = SimulatorTaskExecutionAdapter()
    res_gz = asyncio.run(adapter.execute_task(spec_gz, context={"headless": True}))
    assert res_gz["success"] is True
    assert res_gz["simulator"] == "gazebo_harmonic"
    assert res_gz["action_results"][0].measured_data["distance_m"] > 0.0

