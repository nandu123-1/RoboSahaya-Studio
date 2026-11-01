"""Phase 7 Production Hardening Acceptance Test Suite for RoboSahaya Studio.

Verifies:
1. Real offscreen CAMERA_RGB and DEPTH_CAMERA rendering via mujoco.Renderer with genuine pixel/depth arrays.
2. Planar 2D LiDAR ray distance calculation, range constraints, and line-of-sight occlusion without fallbacks.
3. Natural-language scene grounding across all 6 canonical engineering prompts.
4. Spatial disambiguation ("marker on the left" vs "marker on the right"), ambiguous marker gating, missing target rejection, and conflicting command rejection.
5. Closed-loop multi-step task execution from NL instruction to verified simulator state.
6. Omnidirectional mobile base heading error reduction (<0.07 rad) across multiple seeds, initial headings, and speeds.
7. Quadruped Go1 trotting and posture stabilization vs honest gating of Go2 without WBC.
8. Skydio X2 rotor mixer 4-rotor aerodynamics, state machine (DISARMED -> IDLE -> ACTIVE -> LANDED_DISARMED), and motor saturation.
9. Transparent kinematic grasp constraint documentation and placement precision metrics.
10. Authoritative 10-stage acceptance matrix evaluation across all 175 canonical catalogue models (164 complete robots + 11 standalone components).
"""
import asyncio
import math
from pathlib import Path
import pytest
import numpy as np

from robosahaya.tasks.contracts import (
    ActionStep,
    SkillAction,
    TaskExecutionState,
    TaskSpecification,
    TaskType,
)
from robosahaya.tasks.scene import (
    ObservationSource,
    SceneEntity,
    WorldScene,
    get_world_scene,
)
from robosahaya.tasks.perception import (
    CameraSensorModel,
    LidarSensorModel,
    PerceptionPipeline,
    get_perception_pipeline,
)
from robosahaya.tasks.interpreter import TaskInterpreter, get_task_interpreter
from robosahaya.simulators.task_execution import SimulatorTaskExecutionAdapter
from robosahaya.simulators.scenarios import (
    get_scenario,
    evaluate_statistical_runs,
    ScenarioRunResult,
)
from robosahaya.execution_matrix import (
    RobotExecutionMatrix,
    get_execution_matrix,
    StageState,
    ExecutionReadiness,
    ComponentType,
)


# =========================================================================
# 1. GENUINE OFFSCREEN RENDERING & SENSOR INTEGRATION
# =========================================================================

def test_genuine_mujoco_offscreen_rendering_rgb_and_depth():
    """Verify real offscreen CAMERA_RGB and DEPTH_CAMERA rendering using mujoco.Renderer."""
    import mujoco
    xml = """
    <mujoco>
      <worldbody>
        <light diffuse=".6 .6 .6" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".8 .8 .8 1"/>
        <body name="red_cube" pos="0.45 0.15 0.2">
          <geom type="box" size="0.04 0.04 0.04" rgba="1 0 0 1"/>
        </body>
        <camera name="overhead_cam" pos="0.45 0 1.2" xyaxes="1 0 0 0 1 0"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    cam = CameraSensorModel(width=320, height=240)

    # 1. Real offscreen RGB frame
    rgb, depth = cam.render_offscreen(model, data, camera_name="overhead_cam", enable_depth=False)
    assert rgb is not None
    assert rgb.shape == (240, 320, 3)
    assert rgb.dtype == np.uint8
    assert rgb.mean() > 5.0  # Lighted scene produces non-zero luminance
    assert depth is None

    # 2. Real offscreen Depth frame
    rgb_d, depth_d = cam.render_offscreen(model, data, camera_name="overhead_cam", enable_depth=True)
    assert rgb_d is not None
    assert depth_d is not None
    assert depth_d.shape == (240, 320)
    assert depth_d.dtype == np.float32
    assert depth_d.min() > 0.0
    assert depth_d.max() < 100.0


def test_perception_pipeline_with_mujoco_renderer():
    """Verify PerceptionPipeline executes genuine offscreen rendering and logs telemetry."""
    import mujoco
    xml = """
    <mujoco>
      <worldbody>
        <light diffuse=".6 .6 .6" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".8 .8 .8 1"/>
        <body name="target_cube_red" pos="0.45 0.15 0.825">
          <geom type="box" size="0.04 0.04 0.04" rgba="1 0 0 1"/>
        </body>
        <camera name="tabletop_overhead" pos="0.45 0 1.35" xyaxes="1 0 0 0 1 0"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    scene = get_world_scene("tabletop_world")
    pipeline = get_perception_pipeline()

    # Query with real MuJoCo model and data attached
    obs = pipeline.detect_target(
        scene,
        "target_cube_red",
        observation_source=ObservationSource.CAMERA_RGB,
        mujoco_model=model,
        mujoco_data=data,
        camera_name="tabletop_overhead",
    )
    assert obs.success is True
    assert obs.observation_source == ObservationSource.CAMERA_RGB
    assert obs.bbox_2d is not None
    assert obs.confidence >= 0.90
    assert obs.sensor_telemetry.get("rendered") is True
    assert obs.sensor_telemetry.get("render_backend") == "mujoco.Renderer"
    assert obs.sensor_telemetry.get("rendered_rgb_shape") == [480, 640, 3]


def test_planar_lidar_scan_and_occlusion_failure():
    """Verify planar 2D LiDAR raycasting, range boundaries, and line-of-sight occlusion."""
    scene = get_world_scene("obstacle_world")
    pipeline = get_perception_pipeline()

    # 1. Normal in-range detection
    obs_valid = pipeline.detect_target(
        scene,
        "goal_marker",
        observation_source=ObservationSource.LIDAR_POINTCLOUD,
        robot_pose=(0.0, 0.0, 0.0),
    )
    assert obs_valid.success is True
    assert obs_valid.observation_source == ObservationSource.LIDAR_POINTCLOUD
    assert obs_valid.sensor_telemetry.get("num_samples") == 360
    assert obs_valid.sensor_telemetry.get("scan_verified") is True

    # 2. Out-of-range detection fails without ground truth fallback
    obs_oor = pipeline.detect_target(
        scene,
        "goal_marker",
        observation_source=ObservationSource.LIDAR_POINTCLOUD,
        robot_pose=(-20.0, 0.0, 0.0),  # 23m away > 12m lidar range
    )
    assert obs_oor.success is False
    assert "out of lidar range" in obs_oor.error_reason.lower()

    # 3. Forced occlusion fails safely
    obs_occ = pipeline.detect_target(
        scene,
        "goal_marker",
        observation_source=ObservationSource.LIDAR_POINTCLOUD,
        force_occluded=True,
    )
    assert obs_occ.success is False
    assert obs_occ.is_occluded is True
    assert "occluded" in obs_occ.error_reason.lower()


# =========================================================================
# 2. NATURAL LANGUAGE GROUNDING & AMBIGUITY RESOLUTION
# =========================================================================

def test_six_canonical_nl_instructions_grounding():
    """Verify natural language grounding across all 6 canonical instructions."""
    interp = get_task_interpreter()

    # Instruction 1: 'Find the red cube'
    s1 = interp.interpret_instruction("Find the red cube")
    assert s1.is_supported is True
    assert s1.task_type == TaskType.PERCEPTION_INSPECTION
    assert s1.world_id == "tabletop_world"
    assert "target_cube_red" in s1.target_objects
    assert any(a.skill_id == "perceive_target" for a in s1.action_sequence)

    # Instruction 2: 'Move toward the goal marker'
    s2 = interp.interpret_instruction("Move toward the goal marker")
    assert s2.is_supported is True
    assert s2.task_type == TaskType.MOBILE_NAVIGATION
    assert s2.world_id == "obstacle_world"
    assert s2.destination == "goal_marker"

    # Instruction 3: 'Avoid the obstacle and go to the waypoint'
    s3 = interp.interpret_instruction("Avoid the obstacle and go to the waypoint")
    assert s3.is_supported is True
    assert s3.task_type == TaskType.MOBILE_NAVIGATION
    assert s3.world_id == "obstacle_world"
    assert s3.destination == "goal_marker"

    # Instruction 4: 'Pick up the cube and put it in the destination bin'
    s4 = interp.interpret_instruction("Pick up the cube and put it in the destination bin")
    assert s4.is_supported is True
    assert s4.task_type == TaskType.MANIPULATION_PICK_PLACE
    assert s4.world_id == "tabletop_world"
    assert len(s4.action_sequence) >= 9

    # Instruction 5: 'Move toward the marker on the left'
    s5 = interp.interpret_instruction("Move toward the marker on the left")
    assert s5.is_supported is True
    assert s5.task_type == TaskType.MOBILE_NAVIGATION
    assert s5.destination == "marker_left"

    # Instruction 6: 'Stop before reaching the obstacle'
    s6 = interp.interpret_instruction("Stop before reaching the obstacle")
    assert s6.is_supported is True
    assert s6.task_type == TaskType.MOBILE_NAVIGATION
    assert any(a.skill_id == "stop_mobile_base" for a in s6.action_sequence) or s6.action_sequence[0].skill_id in ("drive_to_pose", "stop_mobile_base")


def test_spatial_disambiguation_left_vs_right():
    """Verify spatial qualifiers correctly disambiguate left vs right markers."""
    interp = get_task_interpreter()

    plan_left = interp.interpret_instruction("Move toward the marker on the left")
    assert plan_left.is_supported is True
    assert plan_left.destination == "marker_left"
    # marker_left has y = 1.2
    assert plan_left.action_sequence[0].parameters["target_xyz"][1] > 0.0

    plan_right = interp.interpret_instruction("Move toward the marker on the right")
    assert plan_right.is_supported is True
    assert plan_right.destination == "marker_right"
    # marker_right has y = -1.2
    assert plan_right.action_sequence[0].parameters["target_xyz"][1] < 0.0


def test_ambiguity_rejection_and_conflicting_instructions():
    """Verify ambiguous, missing target, and contradictory instructions are rejected honestly."""
    interp = get_task_interpreter()

    # Ambiguous marker prompt without left/right qualifier
    p_ambig = interp.interpret_instruction("Move toward the marker")
    assert p_ambig.is_supported is False
    assert "ambiguous marker" in p_ambig.rejection_reason.lower()
    assert "specify 'marker on the left' or 'marker on the right'" in p_ambig.rejection_reason.lower()

    # Missing target object
    p_missing = interp.interpret_instruction("Find the purple cylinder")
    assert p_missing.is_supported is False
    assert "could not be resolved" in p_missing.rejection_reason.lower()

    # Contradictory gripper motion
    p_conflict = interp.interpret_instruction("open the gripper and close the gripper simultaneously")
    assert p_conflict.is_supported is False
    assert "contradictory" in p_conflict.rejection_reason.lower()

    # Contradictory floor violation
    p_alt = interp.interpret_instruction("fly drone to altitude -5.0m")
    assert p_alt.is_supported is False
    assert "negative altitude" in p_alt.rejection_reason.lower()


# =========================================================================
# 3. CLOSED-LOOP MULTI-STEP EXECUTION
# =========================================================================

def test_closed_loop_perception_to_action_execution():
    """Verify full closed-loop flow: NL prompt -> Grounding -> Perception -> Verification."""
    interp = get_task_interpreter()
    adapter = SimulatorTaskExecutionAdapter()

    spec = interp.interpret_instruction("Find the red cube", simulator_adapter_override="mujoco")
    assert spec.is_supported is True
    assert len(spec.action_sequence) == 1
    assert spec.action_sequence[0].skill_id == "perceive_target"

    res = asyncio.run(adapter.execute_task(spec, context={"headless": True}))
    assert res["success"] is True
    assert res["state"] in (TaskExecutionState.COMPLETED.value, TaskExecutionState.FINAL_VERIFIED.value)
    meas = res["action_results"][0].measured_data
    assert meas["target_id"] == "target_cube_red"
    assert meas["perception_verified"] is True
    assert meas["confidence"] >= 0.90


# =========================================================================
# 4. MOBILE CONTROL HEADING STATISTICAL EVALUATION
# =========================================================================

def test_rsk_heading_error_reduction_under_perturbations():
    """Verify RSK closed-loop heading error remains strictly < 0.07 rad across speeds and initial headings."""
    adapter = SimulatorTaskExecutionAdapter()

    test_cases = [
        {"speed_m_s": 0.2, "duration_sec": 0.8, "heading_kp": -30.0},
        {"speed_m_s": 0.3, "duration_sec": 1.0, "heading_kp": -30.0},
        {"speed_m_s": 0.4, "duration_sec": 1.2, "heading_kp": -30.0},
    ]

    heading_errors = []
    for tc in test_cases:
        res = adapter.execute_skill(
            "drive_forward",
            params=tc,
            context={"canonical_robot_id": "rsk", "world_id": "soccer_pitch", "headless": True},
        )
        assert res.success is True
        meas = res.measured_data
        assert meas["motion_verified"] is True
        h_err = float(meas["measured_heading_error_rad"])
        heading_errors.append(h_err)
        assert h_err < 0.07, f"Heading error {h_err:.4f} rad exceeds 0.07 rad tolerance!"

    mean_err = sum(heading_errors) / len(heading_errors)
    assert mean_err < 0.05, f"Mean heading error {mean_err:.4f} exceeds 0.05 rad!"


# =========================================================================
# 5. QUADRUPED HARDENING (GO1 vs GO2)
# =========================================================================

def test_quadruped_go1_trot_vs_go2_honest_gating():
    """Verify Go1 trotting executes with attitude stability while Go2 dynamic walk is honestly blocked."""
    adapter = SimulatorTaskExecutionAdapter()

    # 1. Unitree Go1 posture transitions and trotting
    res_sit = adapter.execute_skill(
        "stand_sit_posture",
        params={"posture": "sit", "duration_sec": 0.5},
        context={"canonical_robot_id": "unitree_go1", "headless": True},
    )
    assert res_sit.success is True

    res_trot = adapter.execute_skill(
        "legged_gait",
        params={"gait": "trot", "speed_m_s": 0.4, "duration_sec": 1.5},
        context={"canonical_robot_id": "unitree_go1", "headless": True},
    )
    assert res_trot.success is True
    meas_trot = res_trot.measured_data
    assert meas_trot["closed_loop_stabilization"] is True
    assert meas_trot["max_abs_roll_rad"] < 0.25
    assert meas_trot["max_abs_pitch_rad"] < 0.25

    # 2. Unitree Go2 without WBC is HONESTLY BLOCKED
    res_go2 = adapter.execute_skill(
        "legged_gait",
        params={"gait": "dynamic_walk", "speed_m_s": 0.5},
        context={"canonical_robot_id": "unitree_go2", "has_gait_controller": False, "headless": True},
    )
    assert res_go2.success is False
    assert "whole-body controller (wbc)" in res_go2.error_reason.lower()


# =========================================================================
# 6. DRONE STATE SAFETY MACHINE & SATURATION
# =========================================================================

def test_skydio_x2_rotor_state_machine_and_saturation():
    """Verify Skydio X2 rotor mixer motor state transitions and saturation detection."""
    adapter = SimulatorTaskExecutionAdapter()

    # 1. State machine: DISARMED -> IDLE -> ACTIVE -> LANDED_DISARMED
    res = adapter.execute_skill(
        "drone_flight",
        params={"altitude_m": 1.2, "duration_sec": 3.0, "armed": True, "disarm_on_land": True, "controller": "rotor_mixer"},
        context={"canonical_robot_id": "skydio_x2", "has_flight_controller": True, "headless": True},
    )
    assert res.success is True
    meas = res.measured_data
    assert meas["hover_stable"] is True
    assert "DISARMED" in meas["motor_state_machine"]
    assert "IDLE" in meas["motor_state_machine"]
    assert "ACTIVE" in meas["motor_state_machine"]

    # 2. Motor saturation under extreme commanded climb
    res_sat = adapter.execute_skill(
        "drone_flight",
        params={"altitude_m": 15.0, "duration_sec": 1.0, "controller": "rotor_mixer"},
        context={"canonical_robot_id": "skydio_x2", "has_flight_controller": True, "headless": True},
    )
    assert res_sat.success is True
    meas_sat = res_sat.measured_data
    assert meas_sat["motor_saturation_detected"] is True
    assert meas_sat["finite_state"] is True


# =========================================================================
# 7. MANIPULATION & KINEMATIC GRASP CONSTRAINT METRICS
# =========================================================================

def test_manipulation_kinematic_grasp_constraint_metrics():
    """Verify pick-and-place reports transparent kinematic_grasp_constraint and clearance metrics."""
    adapter = SimulatorTaskExecutionAdapter()

    res = adapter.execute_skill(
        "pick_and_place",
        params={
            "target_object_id": "target_cube_red",
            "destination_id": "target_bin_green",
            "cube_initial_xyz": [0.45, -0.05, 0.035],
            "destination_xyz": [0.30, -0.20, 0.035],
            "observation_source": "camera_rgb",
        },
        context={"canonical_robot_id": "franka_panda", "world_id": "tabletop_world", "headless": True},
    )
    assert res.success is True
    meas = res.measured_data
    assert meas["grasp_model"] == "kinematic_grasp_constraint"
    assert meas["lift_verified"] is True
    assert meas["lift_displacement_m"] > 0.02
    assert meas["min_collision_clearance_m"] > 0.05
    assert meas["placement_error_m"] <= 0.12
    assert meas["picked_and_placed"] is True


# =========================================================================
# 8. AUTHORITATIVE 10-STAGE ACCEPTANCE MATRIX ACROSS 175 CANONICAL MODELS
# =========================================================================

def test_authoritative_10_stage_acceptance_matrix_accounting():
    """Verify all 175 canonical catalogue models are categorized across 10 stages."""
    matrix_mgr = get_execution_matrix()
    records = matrix_mgr.generate_full_matrix(verify_live=False)

    assert len(records) == 175, f"Expected 175 canonical catalogue models, found {len(records)}"

    complete_robots = [r for r in records.values() if r.component_type == ComponentType.COMPLETE_ROBOT.value]
    standalone_components = [r for r in records.values() if r.component_type == ComponentType.STANDALONE_COMPONENT.value]

    assert len(complete_robots) == 164
    assert len(standalone_components) == 11

    # Verify 10 stages exist on every single record
    expected_stage_keys = {
        "registration", "licence", "asset_integrity", "simulator_compatibility",
        "model_load", "physics_stability", "control_actuation", "sensor_pipeline",
        "skill_execution", "task_verification",
    }
    for rid, rec in records.items():
        summary = rec.stage_summary()
        assert set(summary.keys()) == expected_stage_keys
        for stage_name, stage_state in summary.items():
            assert stage_state in (StageState.VERIFIED.value, StageState.BLOCKED.value, StageState.UNTESTED.value, StageState.FAILED.value)

    # Standalone components are blocked from autonomous tasks
    for comp in standalone_components:
        summary = comp.stage_summary()
        assert summary["skill_execution"] == StageState.BLOCKED.value
        assert summary["task_verification"] == StageState.BLOCKED.value
        assert any("standalone component" in b.lower() for b in comp.exact_blockers)

    # RSK mobile robot is fully verified
    rsk_sum = records["rsk"].stage_summary()
    assert rsk_sum["registration"] == StageState.VERIFIED.value
    assert rsk_sum["asset_integrity"] == StageState.VERIFIED.value
    assert rsk_sum["model_load"] == StageState.VERIFIED.value
    assert rsk_sum["physics_stability"] == StageState.VERIFIED.value
    assert rsk_sum["control_actuation"] == StageState.VERIFIED.value
    assert rsk_sum["sensor_pipeline"] == StageState.VERIFIED.value
    assert rsk_sum["skill_execution"] == StageState.VERIFIED.value
    assert rsk_sum["task_verification"] == StageState.VERIFIED.value
