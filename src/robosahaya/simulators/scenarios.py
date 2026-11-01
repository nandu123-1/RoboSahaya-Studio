"""Reproducible Simulation Scenarios and Validation Engine for RoboSahaya Studio.

Defines versioned, reusable simulation test scenarios across robot families:
- Mobile base (differential drive, RSK omnidirectional)
- Manipulator (Franka Panda, SO-101)
- Quadruped (Unitree Go1)
- Aerial drone (Skydio X2)

Supports initial state variations, statistical multi-run evaluations,
failure boundary checks, and evidence invalidation.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ScenarioCategory(str, Enum):
    MOBILE_BASE = "mobile_base"
    MANIPULATOR = "manipulator"
    QUADRUPED = "quadruped"
    AERIAL_DRONE = "aerial_drone"
    PERCEPTION = "perception"


class FailureConditionType(str, Enum):
    ATTITUDE_INSTABILITY = "attitude_instability"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    COLLISION_VIOLATION = "collision_violation"
    MOTOR_SATURATION = "motor_saturation"
    TIMEOUT = "timeout"
    DIVERGENT_STATE = "divergent_state"
    PERCEPTION_LOST = "perception_lost"


@dataclass
class ScenarioDefinition:
    """Versioned, reproducible simulation scenario definition."""
    scenario_id: str
    version: str
    category: ScenarioCategory
    robot_id: str
    robot_version: str
    simulator: str
    world_id: str
    initial_state: Dict[str, Any]
    controller_settings: Dict[str, Any]
    task_command: str
    duration_sec: float
    state_variables_to_record: List[str]
    success_criteria: Dict[str, Any]
    failure_conditions: List[str]
    required_evidence: List[str]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class ScenarioRunResult:
    """Result from executing one instance of a ScenarioDefinition."""
    run_id: str
    scenario_id: str
    scenario_version: str
    passed: bool
    initial_state_actual: Dict[str, Any]
    measured_metrics: Dict[str, Any]
    violations: List[str]
    trajectory_samples_count: int
    duration_simulated_sec: float
    evidence_recorded: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StatisticalScenarioResult:
    """Aggregated statistical result across repeated runs with variations/perturbations."""
    scenario_id: str
    total_runs: int
    successful_runs: int
    success_rate: float
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    all_runs_passed: bool
    rejection_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Authoritative versioned scenario registry
SCENARIO_REGISTRY: Dict[str, ScenarioDefinition] = {
    # 1. Mobile Base Scenarios
    "diff_drive_navigation_v1": ScenarioDefinition(
        scenario_id="diff_drive_navigation_v1",
        version="1.0.0",
        category=ScenarioCategory.MOBILE_BASE,
        robot_id="differential_drive",
        robot_version="v1.0.0",
        simulator="gazebo_harmonic",
        world_id="obstacle_world",
        initial_state={"x": 0.0, "y": 0.0, "yaw": 0.0},
        controller_settings={"speed_m_s": 0.5, "turn_speed_rad_s": 0.5},
        task_command="Drive forward 1.0m in obstacle course",
        duration_sec=2.0,
        state_variables_to_record=["x", "y", "yaw", "vx", "wz"],
        success_criteria={"min_displacement_m": 0.8, "max_lateral_deviation_m": 0.25},
        failure_conditions=[FailureConditionType.COLLISION_VIOLATION.value, FailureConditionType.TIMEOUT.value],
        required_evidence=["distance_m", "simulated_iterations", "motion_verified"],
        description="Deterministic differential drive straight translation under Gazebo Harmonic.",
    ),
    "rsk_omni_sprint_v1": ScenarioDefinition(
        scenario_id="rsk_omni_sprint_v1",
        version="1.0.0",
        category=ScenarioCategory.MOBILE_BASE,
        robot_id="rsk",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="soccer_pitch",
        initial_state={"x": 0.0, "y": 0.0, "yaw": 0.0},
        controller_settings={"wheel_velocities": [5.44, -2.72, -2.72]},
        task_command="Drive RSK robot forward at 5.44 rad/s for 0.5s",
        duration_sec=0.5,
        state_variables_to_record=["xpos", "qpos", "qvel"],
        success_criteria={"min_displacement_m": 0.1, "max_tilt_rad": 0.15},
        failure_conditions=[FailureConditionType.ATTITUDE_INSTABILITY.value, FailureConditionType.DIVERGENT_STATE.value],
        required_evidence=["displacement_m", "velocity_norm", "wheel_velocities"],
        description="Omnidirectional 3-wheel kinematic velocity control on soccer pitch.",
    ),

    # 2. Quadruped Scenarios
    "go1_trot_stability_v1": ScenarioDefinition(
        scenario_id="go1_trot_stability_v1",
        version="1.0.0",
        category=ScenarioCategory.QUADRUPED,
        robot_id="unitree_go1",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="obstacle_world",
        initial_state={"trunk_height_m": 0.35, "roll_rad": 0.0, "pitch_rad": 0.0},
        controller_settings={"gait": "trot", "frequency_hz": 1.5, "speed_m_s": 0.5, "closed_loop": True},
        task_command="Make Unitree Go1 walk forward 1.0m",
        duration_sec=2.5,
        state_variables_to_record=["trunk_height", "roll", "pitch", "displacement", "foot_contacts"],
        success_criteria={"min_displacement_m": 0.3, "max_abs_roll_rad": 0.25, "max_abs_pitch_rad": 0.25},
        failure_conditions=[FailureConditionType.ATTITUDE_INSTABILITY.value, FailureConditionType.DIVERGENT_STATE.value],
        required_evidence=["displacement_m", "cycles", "trunk_height_m", "stable"],
        description="Unitree Go1 alternating trotting gait with closed-loop roll/pitch stabilization.",
    ),

    # 3. Aerial Drone Scenarios
    "skydio_x2_flight_v1": ScenarioDefinition(
        scenario_id="skydio_x2_flight_v1",
        version="1.0.0",
        category=ScenarioCategory.AERIAL_DRONE,
        robot_id="skydio_x2",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="obstacle_world",
        initial_state={"altitude_m": 0.0, "roll_rad": 0.0, "pitch_rad": 0.0},
        controller_settings={"target_altitude_m": 1.5, "waypoint": [0.5, 0.0], "propulsion_model": "rotor_aerodynamics_mixer"},
        task_command="Fly Skydio X2 drone with flight controller to 1.5m altitude",
        duration_sec=3.0,
        state_variables_to_record=["altitude", "pos_x", "pos_y", "roll", "pitch", "rotor_speeds"],
        success_criteria={"target_altitude_reached": True, "max_abs_roll_rad": 0.15, "max_abs_pitch_rad": 0.15},
        failure_conditions=[FailureConditionType.ATTITUDE_INSTABILITY.value, FailureConditionType.MOTOR_SATURATION.value],
        required_evidence=["peak_altitude_m", "final_altitude_m", "hover_stable", "flight_verified"],
        description="Physical 4-rotor aerodynamics mixer with altitude hold and soft landing.",
    ),

    # 4. Manipulator Scenarios
    "panda_pick_place_v1": ScenarioDefinition(
        scenario_id="panda_pick_place_v1",
        version="1.0.0",
        category=ScenarioCategory.MANIPULATOR,
        robot_id="franka_panda",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="tabletop_world",
        initial_state={"cube_initial_xyz": [0.45, 0.0, 0.035], "destination_xyz": [0.3, -0.2, 0.035]},
        controller_settings={"kinematic_constraint_grasp": True, "clearance_height_m": 0.15},
        task_command="Pick up red block and place it in the target bin",
        duration_sec=4.0,
        state_variables_to_record=["ee_pos", "cube_pos", "joint_angles", "grasp_state"],
        success_criteria={"max_placement_error_m": 0.10, "lift_verified": True, "collision_free": True},
        failure_conditions=[FailureConditionType.COLLISION_VIOLATION.value, FailureConditionType.JOINT_LIMIT_VIOLATION.value],
        required_evidence=["placement_error_m", "lift_displacement_m", "collision_free", "grasp_stable"],
        description="Collision-aware kinematic constraint pick and place onto destination table/bin.",
    ),
    "so101_trajectory_v1": ScenarioDefinition(
        scenario_id="so101_trajectory_v1",
        version="1.0.0",
        category=ScenarioCategory.MANIPULATOR,
        robot_id="so_arm101",
        robot_version="v1.0.0",
        simulator="gazebo_harmonic",
        world_id="tabletop_world",
        initial_state={"joint_angles": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        controller_settings={"target_joint_angles": [0.3, 0.2, 0.1, 0.0, 0.0, 0.0]},
        task_command="Move so_arm101 joints to target configuration in Gazebo",
        duration_sec=2.0,
        state_variables_to_record=["joint_states", "joint_errors"],
        success_criteria={"max_joint_error_rad": 0.05, "motion_verified": True},
        failure_conditions=[FailureConditionType.TIMEOUT.value, FailureConditionType.JOINT_LIMIT_VIOLATION.value],
        required_evidence=["simulated_iterations", "joint_displacement_rad", "motion_verified"],
        description="Gazebo Harmonic JointPositionController execution on SO-101 arm.",
    ),

    # 5. Advanced Quadruped Scenarios (Phase 6 statistical testing)
    "go1_multispeed_trot_v1": ScenarioDefinition(
        scenario_id="go1_multispeed_trot_v1",
        version="1.0.0",
        category=ScenarioCategory.QUADRUPED,
        robot_id="unitree_go1",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="obstacle_world",
        initial_state={"trunk_height_m": 0.35, "roll_rad": 0.0, "pitch_rad": 0.0},
        controller_settings={"gait": "trot", "speeds_m_s": [0.2, 0.4, 0.6], "closed_loop": True},
        task_command="Test Go1 trotting across variable forward speeds and measure residual stopping drift",
        duration_sec=3.0,
        state_variables_to_record=["displacement_m", "speed_m_s", "max_abs_roll_rad", "max_abs_pitch_rad", "residual_drift_m"],
        success_criteria={"min_displacement_m": 0.4, "max_abs_roll_rad": 0.30, "max_abs_pitch_rad": 0.30, "max_residual_drift_m": 0.08},
        failure_conditions=[FailureConditionType.ATTITUDE_INSTABILITY.value, FailureConditionType.DIVERGENT_STATE.value],
        required_evidence=["displacement_m", "speed_m_s", "closed_loop_stabilization", "motion_verified"],
        description="Multi-speed Go1 forward trotting and controlled deceleration with minimal residual drift.",
    ),
    "go1_bidirectional_turning_v1": ScenarioDefinition(
        scenario_id="go1_bidirectional_turning_v1",
        version="1.0.0",
        category=ScenarioCategory.QUADRUPED,
        robot_id="unitree_go1",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="obstacle_world",
        initial_state={"trunk_height_m": 0.35, "roll_rad": 0.0, "pitch_rad": 0.0},
        controller_settings={"gait": "trot", "turn_rates_rad_s": [0.4, -0.4], "closed_loop": True},
        task_command="Test Go1 bidirectional yaw turning during active trot",
        duration_sec=2.5,
        state_variables_to_record=["turn_rate_rad_s", "turning_verified", "max_abs_roll_rad", "max_abs_pitch_rad"],
        success_criteria={"turning_verified": True, "max_abs_roll_rad": 0.30, "max_abs_pitch_rad": 0.30},
        failure_conditions=[FailureConditionType.ATTITUDE_INSTABILITY.value],
        required_evidence=["turning_verified", "closed_loop_stabilization"],
        description="Go1 trotting yaw turn in positive (+left) and negative (-right) directions.",
    ),
    "go1_disturbance_recovery_v1": ScenarioDefinition(
        scenario_id="go1_disturbance_recovery_v1",
        version="1.0.0",
        category=ScenarioCategory.QUADRUPED,
        robot_id="unitree_go1",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="obstacle_world",
        initial_state={"trunk_height_m": 0.35, "roll_rad": 0.0, "pitch_rad": 0.0},
        controller_settings={"apply_disturbance": True, "disturbance_force_n": 15.0, "speed_m_s": 0.4},
        task_command="Test Go1 recovery under dynamic lateral impulse perturbation",
        duration_sec=2.5,
        state_variables_to_record=["disturbance_applied", "disturbance_recovered", "recovery_time_sec", "max_abs_roll_rad"],
        success_criteria={"disturbance_recovered": True, "max_recovery_time_sec": 0.85},
        failure_conditions=[FailureConditionType.ATTITUDE_INSTABILITY.value],
        required_evidence=["disturbance_applied", "disturbance_recovered", "recovery_time_sec"],
        description="Dynamic disturbance recovery testing posture restoration under external impulse force.",
    ),

    # 6. Advanced Drone Scenarios (Phase 6 rotor mixer & motor state validation)
    "skydio_x2_hover_landing_v1": ScenarioDefinition(
        scenario_id="skydio_x2_hover_landing_v1",
        version="1.0.0",
        category=ScenarioCategory.AERIAL_DRONE,
        robot_id="skydio_x2",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="obstacle_world",
        initial_state={"altitude_m": 0.0, "roll_rad": 0.0, "pitch_rad": 0.0},
        controller_settings={"target_altitude_m": 1.2, "disarm_on_land": True, "propulsion_model": "rotor_aerodynamics_mixer"},
        task_command="Take off, hover at 1.2m, and descend to soft landing with motor disarm",
        duration_sec=3.5,
        state_variables_to_record=["peak_altitude_m", "final_altitude_m", "motor_state", "hover_stable"],
        success_criteria={"peak_altitude_m": 1.0, "final_altitude_max_m": 0.15, "disarmed_on_stop": True},
        failure_conditions=[FailureConditionType.ATTITUDE_INSTABILITY.value, FailureConditionType.DIVERGENT_STATE.value],
        required_evidence=["peak_altitude_m", "final_altitude_m", "hover_stable", "disarmed_on_stop"],
        description="Skydio X2 rotor-mixer flight with controlled landing and motor disarm verification.",
    ),
    "skydio_x2_motor_saturation_v1": ScenarioDefinition(
        scenario_id="skydio_x2_motor_saturation_v1",
        version="1.0.0",
        category=ScenarioCategory.AERIAL_DRONE,
        robot_id="skydio_x2",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="obstacle_world",
        initial_state={"altitude_m": 0.0},
        controller_settings={"altitude_m": 10.0, "duration_sec": 1.0, "propulsion_model": "rotor_aerodynamics_mixer"},
        task_command="Command extreme climb rate to evaluate rotor mixer actuator limits",
        duration_sec=1.0,
        state_variables_to_record=["motor_saturation_detected", "rotor_speeds_rad_s", "collective_thrust_n"],
        success_criteria={"motor_saturation_detected": True, "finite_state": True},
        failure_conditions=[FailureConditionType.DIVERGENT_STATE.value],
        required_evidence=["motor_saturation_detected", "finite_state"],
        description="Evaluation of rotor speed clamping and motor saturation detection under extreme commands.",
    ),

    # 7. Perception-to-Action Scenarios (Phase 6 sensor-grounded perception)
    "tabletop_perception_rgb_v1": ScenarioDefinition(
        scenario_id="tabletop_perception_rgb_v1",
        version="1.0.0",
        category=ScenarioCategory.PERCEPTION,
        robot_id="franka_panda",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="tabletop_world",
        initial_state={"target_cube_red": [0.45, 0.15, 0.825]},
        controller_settings={"camera": "tabletop_overhead_rgbd", "observation_source": "camera_rgb"},
        task_command="Detect red cube using camera RGB perspective projection",
        duration_sec=1.0,
        state_variables_to_record=["target_detected", "confidence", "bbox_2d", "observation_source"],
        success_criteria={"target_detected": True, "min_confidence": 0.90, "observation_source": "camera_rgb"},
        failure_conditions=[FailureConditionType.PERCEPTION_LOST.value],
        required_evidence=["target_detected", "confidence", "observation_source"],
        description="Overhead camera RGB observation with perspective projection, FOV validation, and 2D bounding box.",
    ),
    "tabletop_occlusion_rejection_v1": ScenarioDefinition(
        scenario_id="tabletop_occlusion_rejection_v1",
        version="1.0.0",
        category=ScenarioCategory.PERCEPTION,
        robot_id="franka_panda",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="tabletop_world",
        initial_state={"target_cube_red": [0.45, 0.15, 0.825]},
        controller_settings={"camera": "tabletop_overhead_rgbd", "observation_source": "camera_rgb", "force_occluded": True},
        task_command="Detect target with occlusion and verify safe task halt before blind execution",
        duration_sec=1.0,
        state_variables_to_record=["target_detected", "is_occluded", "error_reason"],
        success_criteria={"target_detected": False, "is_occluded": True},
        failure_conditions=[],
        required_evidence=["is_occluded", "error_reason"],
        description="Perception failure safety gate verifying system stops safely when target is occluded.",
    ),
    "rsk_omni_waypoint_v1": ScenarioDefinition(
        scenario_id="rsk_omni_waypoint_v1",
        version="1.0.0",
        category=ScenarioCategory.MOBILE_BASE,
        robot_id="rsk",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="soccer_pitch",
        initial_state={"x": 0.0, "y": 0.0, "yaw": 0.0},
        controller_settings={"speed_m_s": 0.3, "duration_sec": 1.0},
        task_command="Drive RSK mobile base forward with heading monitoring",
        duration_sec=1.0,
        state_variables_to_record=["measured_displacement_m", "measured_heading_error_rad", "heading_stable"],
        success_criteria={"min_displacement_m": 0.05, "heading_stable": True},
        failure_conditions=[FailureConditionType.ATTITUDE_INSTABILITY.value],
        required_evidence=["measured_displacement_m", "measured_heading_error_rad", "heading_stable"],
        description="RSK 3-wheel omni motion verifying linear displacement and heading stability.",
    ),
    "rsk_closed_loop_heading_v1": ScenarioDefinition(
        scenario_id="rsk_closed_loop_heading_v1",
        version="1.0.0",
        category=ScenarioCategory.MOBILE_BASE,
        robot_id="rsk",
        robot_version="v1.0.0",
        simulator="mujoco",
        world_id="soccer_pitch",
        initial_state={"x": 0.0, "y": 0.0, "yaw": 0.0},
        controller_settings={"speed_m_s": 0.3, "heading_kp": -10.0, "duration_sec": 1.0},
        task_command="Drive forward at 0.3 m/s with closed loop heading correction",
        duration_sec=1.0,
        state_variables_to_record=["xpos", "yaw", "measured_heading_error_rad", "measured_displacement_m"],
        success_criteria={"max_heading_error_rad": 0.07, "min_displacement_m": 0.05},
        failure_conditions=[FailureConditionType.ATTITUDE_INSTABILITY.value, FailureConditionType.DIVERGENT_STATE.value],
        required_evidence=["measured_heading_error_rad", "measured_displacement_m", "heading_stable"],
        description="Statistical closed-loop heading correction (<0.07 rad) under multi-seed perturbations.",
    ),
    "lidar_planar_scan_v1": ScenarioDefinition(
        scenario_id="lidar_planar_scan_v1",
        version="1.0.0",
        category=ScenarioCategory.PERCEPTION,
        robot_id="differential_drive",
        robot_version="v1.0.0",
        simulator="gazebo_harmonic",
        world_id="obstacle_world",
        initial_state={"x": 0.0, "y": 0.0, "yaw": 0.0},
        controller_settings={"sensor": "planar_2d_lidar", "observation_source": "lidar_pointcloud"},
        task_command="Scan obstacle course with planar 2D LiDAR",
        duration_sec=1.0,
        state_variables_to_record=["measured_range_m", "bearing_rad", "num_samples", "scan_verified"],
        success_criteria={"scan_verified": True, "num_samples": 360},
        failure_conditions=[FailureConditionType.PERCEPTION_LOST.value],
        required_evidence=["measured_range_m", "bearing_rad", "scan_verified"],
        description="Planar 2D LiDAR ray distance calculation with obstacle occlusion verification.",
    ),
}


def get_scenario(scenario_id: str) -> Optional[ScenarioDefinition]:
    """Retrieve versioned scenario definition by ID."""
    return SCENARIO_REGISTRY.get(scenario_id)


def list_scenarios(category: Optional[ScenarioCategory] = None) -> List[ScenarioDefinition]:
    """List registered scenarios, optionally filtered by category."""
    if category is None:
        return list(SCENARIO_REGISTRY.values())
    return [s for s in SCENARIO_REGISTRY.values() if s.category == category]


def evaluate_statistical_runs(
    scenario_id: str,
    run_results: List[ScenarioRunResult],
    min_required_success_rate: float = 0.8,
) -> StatisticalScenarioResult:
    """Evaluate statistical performance across multiple perturbed runs of a scenario."""
    if not run_results:
        return StatisticalScenarioResult(
            scenario_id=scenario_id,
            total_runs=0,
            successful_runs=0,
            success_rate=0.0,
            mean_metrics={},
            std_metrics={},
            all_runs_passed=False,
            rejection_reasons=["No run results supplied for evaluation."],
        )

    total = len(run_results)
    passed = sum(1 for r in run_results if r.passed)
    rate = passed / total
    rejections: List[str] = []

    # Compute mean and standard deviation for common numeric metrics
    metric_keys = set()
    for r in run_results:
        metric_keys.update(k for k, v in r.measured_metrics.items() if isinstance(v, (int, float)))

    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}

    for k in sorted(metric_keys):
        vals = [float(r.measured_metrics[k]) for r in run_results if k in r.measured_metrics and isinstance(r.measured_metrics[k], (int, float))]
        if vals:
            mean_val = sum(vals) / len(vals)
            var = sum((x - mean_val) ** 2 for x in vals) / max(1, len(vals) - 1)
            mean_metrics[k] = round(mean_val, 4)
            std_metrics[k] = round(math.sqrt(var), 4)

    all_passed = (rate >= min_required_success_rate)
    if not all_passed:
        rejections.append(
            f"Scenario {scenario_id} achieved {rate*100:.1f}% success rate, below required {min_required_success_rate*100:.1f}%."
        )

    for r in run_results:
        if not r.passed and r.violations:
            rejections.extend(r.violations)

    return StatisticalScenarioResult(
        scenario_id=scenario_id,
        total_runs=total,
        successful_runs=passed,
        success_rate=round(rate, 4),
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        all_runs_passed=all_passed,
        rejection_reasons=rejections[:5],
    )
