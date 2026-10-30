"""Phase 6 Integration Tests: Sensor-Driven Autonomy, Closed-Loop Task Execution,
Multi-Robot Reliability, and Catalogue-Wide Acceptance.
"""
import asyncio
import math
import shutil
import pytest

from robosahaya.tasks.contracts import (
    TaskType,
    TaskExecutionState,
    TaskSpecification,
    SkillAction,
    SkillStatus,
    ObservationRecord,
)
from robosahaya.tasks.scene import (
    ObservationSource,
    WorldScene,
    get_world_scene,
    normalize_observation_source,
)
from robosahaya.tasks.perception import (
    CameraSensorModel,
    PerceptionPipeline,
    get_perception_pipeline,
)
from robosahaya.tasks.interpreter import get_task_interpreter
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


def test_perception_sources_distinction_and_pipeline():
    """Verify distinct observation sources (GROUND_TRUTH_SIM, CAMERA_RGB, DEPTH_CAMERA, LIDAR_POINTCLOUD)
    with perspective projection, FOV validation, and strict occlusion failure without silent fallbacks."""
    scene = get_world_scene("tabletop_world")
    pipeline = get_perception_pipeline()

    # 1. GROUND_TRUTH_SIM: exact simulator coordinates
    gt_obs = pipeline.detect_target(scene, "target_cube_red", observation_source=ObservationSource.GROUND_TRUTH_SIM)
    assert gt_obs.success is True
    assert gt_obs.observation_source == ObservationSource.GROUND_TRUTH_SIM
    assert gt_obs.confidence == 1.0
    assert gt_obs.sensor_telemetry["mode"] == "ground_truth_sim"

    # 2. CAMERA_RGB: perspective projection, FOV check, bounding box
    rgb_obs = pipeline.detect_target(scene, "target_cube_red", observation_source=ObservationSource.CAMERA_RGB)
    assert rgb_obs.success is True
    assert rgb_obs.observation_source == ObservationSource.CAMERA_RGB
    assert rgb_obs.bbox_2d is not None
    assert len(rgb_obs.bbox_2d) == 4
    assert rgb_obs.confidence >= 0.90
    assert "pixel_centroid" in rgb_obs.sensor_telemetry

    # 3. Occlusion failure: must NOT silently fall back to ground truth
    occluded_obs = pipeline.detect_target(
        scene, "target_cube_red", observation_source=ObservationSource.CAMERA_RGB, force_occluded=True
    )
    assert occluded_obs.success is False
    assert occluded_obs.is_occluded is True
    assert occluded_obs.observation_source == ObservationSource.CAMERA_RGB
    assert "occluded" in occluded_obs.error_reason.lower()

    # 4. Out of FOV check
    narrow_cam = CameraSensorModel(position=(10.0, 10.0, 10.0), look_at=(15.0, 15.0, 10.0))
    fov_obs = pipeline.detect_target(
        scene, "target_cube_red", observation_source=ObservationSource.CAMERA_RGB, camera=narrow_cam
    )
    assert fov_obs.success is False
    assert fov_obs.in_fov is False
    assert "field of view" in fov_obs.error_reason.lower()

    # 5. DEPTH_CAMERA and LIDAR_POINTCLOUD
    depth_obs = pipeline.detect_target(scene, "target_cube_red", observation_source=ObservationSource.DEPTH_CAMERA)
    assert depth_obs.success is True
    assert depth_obs.depth_m is not None
    assert depth_obs.observation_source == ObservationSource.DEPTH_CAMERA

    lidar_obs = pipeline.detect_target(scene, "target_cube_red", observation_source=ObservationSource.LIDAR_POINTCLOUD)
    assert lidar_obs.success is True
    assert lidar_obs.observation_source == ObservationSource.LIDAR_POINTCLOUD


def test_closed_loop_pick_and_place_with_camera_rgb():
    """Verify 13-stage pick-and-place workflow driven by CAMERA_RGB observation source."""
    spec = TaskSpecification(
        task_id="pnp_camera_rgb_01",
        original_instruction="Pick up red cube detected by camera and place into target bin",
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
                description="Pick and place red cube using camera RGB perception",
                parameters={
                    "target_object_id": "target_cube_red",
                    "destination_id": "target_bin_green",
                    "observation_source": "camera_rgb",
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
    assert meas["observation_source"] == "camera_rgb"
    assert meas["target_detected"] is True
    assert meas["perception_confidence"] >= 0.90
    assert meas["grasp_secured"] is True
    assert meas["lift_verified"] is True
    assert meas["lift_displacement_m"] > 0.02
    assert meas["placement_executed"] is True
    assert meas["placement_error_m"] <= 0.12
    assert meas["grasp_model"] == "kinematic_grasp_constraint"
    assert meas["min_collision_clearance_m"] > 0.05


def test_pick_and_place_stops_safely_on_perception_failure():
    """Verify pick-and-place stops safely before blind motion if target is unobservable/occluded."""
    spec = TaskSpecification(
        task_id="pnp_occlusion_abort_01",
        original_instruction="Pick up occluded red cube",
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
                description="Pick and place with forced occlusion",
                parameters={
                    "target_object_id": "target_cube_red",
                    "destination_id": "target_bin_green",
                    "observation_source": "camera_rgb",
                    "force_occluded": True,
                },
            )
        ],
    )
    adapter = SimulatorTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))

    assert res["success"] is False
    assert res["state"] == TaskExecutionState.FAILED.value
    assert "perception failure" in res["error"].lower() or "occluded" in res["error"].lower()


def test_closed_loop_mobile_navigation_heading_tracking():
    """Verify mobile drive forward tracks measured linear displacement and heading stability."""
    spec = TaskSpecification(
        task_id="rsk_closed_loop_drive_01",
        original_instruction="Drive RSK soccer robot forward at 0.3 m/s with heading tracking",
        task_type=TaskType.MOBILE_WHEEL_CONTROL,
        canonical_robot_id="rsk",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="soccer_pitch",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="drive_forward",
                description="Drive forward with heading monitoring",
                parameters={"speed_m_s": 0.3, "duration_sec": 1.0},
            )
        ],
    )
    adapter = SimulatorTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))

    assert res["success"] is True
    meas = res["action_results"][0].measured_data
    assert meas["measured_displacement_m"] > 0.05
    assert "measured_heading_error_rad" in meas
    assert meas["measured_heading_error_rad"] < 0.25
    assert meas["heading_stable"] is True


def test_drone_flight_motor_states_and_disarm_on_land():
    """Verify drone flight handles DISARMED, IDLE, and ACTIVE motor states, ensuring disarmed state has 0 thrust."""
    adapter = SimulatorTaskExecutionAdapter()

    # 1. Unarmed drone retains 0 thrust and 0 RPM
    spec_disarmed = TaskSpecification(
        task_id="x2_disarmed_01",
        original_instruction="Check Skydio X2 disarmed motor state",
        task_type=TaskType.DRONE_FLIGHT,
        canonical_robot_id="skydio_x2",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="drone_flight",
                description="Check disarmed motor state",
                parameters={"altitude_m": 0.0, "duration_sec": 0.5, "armed": False},
            )
        ],
    )
    res_disarmed = asyncio.run(adapter.execute_task(spec_disarmed, context={"headless": True, "has_flight_controller": True}))
    assert res_disarmed["success"] is True
    meas_dis = res_disarmed["action_results"][0].measured_data
    assert meas_dis["motor_state"] == "disarmed"
    assert meas_dis["collective_thrust_n"] == 0.0
    assert all(s == 0.0 for s in meas_dis["rotor_thrusts_n"])

    # 2. Active flight and soft landing with motor disarm
    spec_active = TaskSpecification(
        task_id="x2_active_land_01",
        original_instruction="Fly to 1.2m and land softly with motor disarm",
        task_type=TaskType.DRONE_FLIGHT,
        canonical_robot_id="skydio_x2",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="obstacle_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="drone_flight",
                description="Active flight and landing",
                parameters={"altitude_m": 1.2, "duration_sec": 3.0, "armed": True, "disarm_on_land": True},
            )
        ],
    )
    res_active = asyncio.run(adapter.execute_task(spec_active, context={"headless": True, "has_flight_controller": True}))
    assert res_active["success"] is True
    meas_act = res_active["action_results"][0].measured_data
    assert meas_act["peak_altitude_m"] >= 1.0
    assert meas_act["disarmed_on_stop"] is True


def test_statistical_quadruped_locomotion_multi_speed_and_drift():
    """Verify statistical scenario evaluation for Unitree Go1 across variable speeds with residual drift bounds."""
    runs = []
    speeds = [0.2, 0.4, 0.5, 0.6, 0.4]
    for i, spd in enumerate(speeds):
        runs.append(
            ScenarioRunResult(
                run_id=f"go1_run_{i}",
                scenario_id="go1_multispeed_trot_v1",
                scenario_version="1.0.0",
                passed=True,
                initial_state_actual={"trunk_height_m": 0.35, "roll_rad": 0.0},
                measured_metrics={
                    "displacement_m": spd * 2.0 * 0.95,
                    "max_abs_roll_rad": 0.04 + 0.01 * i,
                    "max_abs_pitch_rad": 0.05 + 0.01 * i,
                    "residual_drift_m": 0.02 + 0.005 * i,
                },
                violations=[],
                trajectory_samples_count=100,
                duration_simulated_sec=2.0,
                evidence_recorded={},
            )
        )

    stat = evaluate_statistical_runs("go1_multispeed_trot_v1", runs, min_required_success_rate=0.90)
    assert stat.total_runs == 5
    assert stat.success_rate == 1.0
    assert stat.all_runs_passed is True
    assert stat.mean_metrics["residual_drift_m"] < 0.08


def test_gazebo_harmonic_so_arm101_gripper_and_joint_execution():
    """Verify Gazebo Harmonic JointPositionController execution for so_arm101 gripper and joint trajectory."""
    gz = shutil.which("gz")
    if gz is None:
        pytest.skip("Gazebo Sim executable 'gz' is not installed.")

    spec = TaskSpecification(
        task_id="gz_so101_gripper_01",
        original_instruction="Open and close so_arm101 gripper in Gazebo",
        task_type=TaskType.MANIPULATION_TRAJECTORY,
        canonical_robot_id="so_arm101",
        robot_model_version="v1.0.0",
        simulator_adapter="gazebo_harmonic",
        world_id="tabletop_world",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="gripper_control",
                description="Open gripper in Gazebo",
                parameters={"action": "open", "aperture_m": 0.08, "duration_sec": 1.0},
            )
        ],
    )
    adapter = GazeboTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))

    assert res["success"] is True
    meas = res["action_results"][0].measured_data
    assert meas["controller_type"] == "gz::sim::systems::JointPositionController"
    assert meas["gripper_state"] == "opened"


def test_evidence_based_verifier_payload():
    """Verify execution matrix stores structured ExecutionEvidencePayload with scenario, observation source, and thresholds."""
    matrix = RobotExecutionMatrix()
    rec = matrix.build_record_for_robot("franka_panda")

    matrix.record_skill_verification(
        robot_id="franka_panda",
        skill_id="pick_and_place",
        simulator="mujoco",
        success=True,
        details={
            "scenario_id": "panda_pick_place_v1",
            "observation_source": "camera_rgb",
            "command_and_target": {"target": "target_cube_red", "dest": "target_bin_green"},
            "acceptance_thresholds": {"max_placement_error_m": 0.12},
            "test_runs_count": 5,
            "measured_result": {"placement_error_m": 0.045, "lift_displacement_m": 0.12},
        },
    )

    ev = matrix.get_skill_evidence("franka_panda", "pick_and_place")
    assert ev is not None
    assert ev["scenario_id"] == "panda_pick_place_v1"
    assert ev["observation_source"] == "camera_rgb"
    assert ev["passed"] is True
    assert ev["acceptance_thresholds"]["max_placement_error_m"] == 0.12
    assert ev["test_runs_count"] == 5


def test_closed_loop_recovery_and_partial_result_preservation():
    """Verify multi-step task preserves partial results when an intermediate step fails."""
    spec = TaskSpecification(
        task_id="multistep_partial_01",
        original_instruction="Drive RSK forward then attempt unsupported action",
        task_type=TaskType.MOBILE_WHEEL_CONTROL,
        canonical_robot_id="rsk",
        robot_model_version="v1.0.0",
        simulator_adapter="mujoco",
        world_id="soccer_pitch",
        action_sequence=[
            SkillAction(
                step_index=0,
                skill_id="drive_forward",
                description="Step 1: Drive forward",
                parameters={"speed_m_s": 0.2, "duration_sec": 0.5},
            ),
            SkillAction(
                step_index=1,
                skill_id="unsupported_skill_xyz",
                description="Step 2: Non-existent skill",
                parameters={},
            ),
        ],
    )
    adapter = SimulatorTaskExecutionAdapter()
    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))

    assert res["success"] is False
    assert res["state"] == TaskExecutionState.BLOCKED.value
    # Step 0 succeeded and was preserved
    assert len(res["action_results"]) == 2
    assert res["action_results"][0].success is True
    assert res["action_results"][1].success is False


def test_nl_interpreter_scene_grounding_marker_and_obstacle():
    """Verify natural language interpreter grounds visible marker and obstacles into task plan."""
    interpreter = get_task_interpreter()

    spec_marker = interpreter.interpret_instruction(
        "Drive toward the visible marker in obstacle course",
        robot_id_override="differential_drive",
        world_id_override="obstacle_world",
    )
    assert spec_marker.is_supported is True
    assert spec_marker.destination == "goal_marker"

    spec_find = interpreter.interpret_instruction(
        "Find the red cube on tabletop",
        robot_id_override="franka_panda",
        world_id_override="tabletop_world",
    )
    assert spec_find.is_supported is True
    assert "target_cube_red" in spec_find.target_objects
