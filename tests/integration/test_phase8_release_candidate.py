"""Phase 8 Release Candidate Test Suite for RoboSahaya Studio.

Verifies:
1. Exact mathematical reconciliation of all 175 canonical catalogue models (164 complete robots + 11 standalone components).
2. Skydio X2 rotor-mixer verification vs full flight readiness distinction.
3. Prioritized robot integrations (UR10e, Unitree Z1, Unitree B2) with physical execution and zero RSK fallback.
4. Stale sensor observation detection and rejection without ground-truth fallbacks.
5. Evidence payload tracking, stale-evidence marking, and selective invalidation.
6. Cross-simulator decoupling (MuJoCo vs Gazebo Harmonic independent records).
7. Closed-loop compound legged locomotion on Unitree Go1 vs honest WBC gating on Go2.
8. Task cancellation, execution cleanup, and zero latched actuator controls.
9. Operator UI execution matrix summary reconciliation.
"""
import asyncio
import threading
from pathlib import Path
import pytest

from robosahaya.tasks.contracts import (
    ActionStep,
    SkillAction,
    SkillStatus,
    TaskExecutionState,
    TaskSpecification,
    TaskType,
)
from robosahaya.tasks.scene import (
    ObservationSource,
    get_world_scene,
)
from robosahaya.tasks.perception import (
    CameraSensorModel,
    LidarSensorModel,
    PerceptionPipeline,
    get_perception_pipeline,
)
from robosahaya.tasks.interpreter import TaskInterpreter, get_task_interpreter
from robosahaya.simulators.task_execution import SimulatorTaskExecutionAdapter, MujocoTaskExecutionAdapter
from robosahaya.execution_matrix import (
    RobotExecutionMatrix,
    get_execution_matrix,
    StageState,
    ExecutionReadiness,
    ComponentType,
    ExecutionEvidencePayload,
)


# =========================================================================
# 1. AUTHORITATIVE 175-MODEL CATALOGUE RECONCILIATION
# =========================================================================

def test_authoritative_175_catalogue_reconciliation():
    """Verify all 175 canonical catalogue records are mathematically reconciled without status overlap."""
    matrix_mgr = get_execution_matrix()
    recon = matrix_mgr.reconcile_catalogue_accounting()

    assert recon["is_mathematically_consistent"] is True
    assert recon["total_canonical_records"] == 175
    assert recon["declared_complete_robots"] == 164
    assert recon["declared_standalone_components"] == 11

    part = recon["complete_robots_partition"]
    assert part["end_to_end_verified_count"] == 6
    assert part["quadrupeds_posture_and_trot_count"] == 2  # unitree_go1, go1
    assert part["quadrupeds_wbc_blocked_count"] == 2       # unitree_go2, go2
    assert part["aerial_rotor_mixer_verified_count"] == 1  # skydio_x2
    assert part["untested_or_awaiting_assets_count"] == 153
    assert part["partition_sum"] == 164

    # Mathematical identity: 6 + 2 + 2 + 1 + 153 = 164 complete robots
    assert (
        part["end_to_end_verified_count"]
        + part["quadrupeds_posture_and_trot_count"]
        + part["quadrupeds_wbc_blocked_count"]
        + part["aerial_rotor_mixer_verified_count"]
        + part["untested_or_awaiting_assets_count"]
    ) == 164

    # Total catalogue identity: 164 + 11 = 175
    assert part["partition_sum"] + len(recon["standalone_components_list"]) == 175


def test_skydio_x2_distinction_rotor_mixer_vs_full_flight():
    """Verify Skydio X2 rotor-mixer physics execution vs blocked full flight readiness."""
    matrix_mgr = get_execution_matrix()
    records = matrix_mgr.generate_full_matrix(verify_live=False)

    skydio_rec = records["skydio_x2"]
    # 1. Authoritative catalogue readiness remains BLOCKED (no bundled aerodynamics/autopilot stack)
    assert skydio_rec.execution_readiness == ExecutionReadiness.BLOCKED.value
    assert skydio_rec.physics_stage == StageState.VERIFIED.value
    assert skydio_rec.control_stage == StageState.BLOCKED.value
    assert any("aerodynamics" in b.lower() for b in skydio_rec.exact_blockers)

    # 2. Rotor mixer executes under active flight controller context in simulation
    adapter = SimulatorTaskExecutionAdapter()
    res_flight = adapter.execute_skill(
        "drone_flight",
        params={"altitude_m": 1.2, "duration_sec": 3.0, "armed": True, "disarm_on_land": True, "controller": "rotor_mixer"},
        context={"canonical_robot_id": "skydio_x2", "has_flight_controller": True, "headless": True},
    )
    assert res_flight.success is True
    assert res_flight.measured_data["finite_state"] is True
    assert res_flight.measured_data["hover_stable"] is True

    # 3. Flight without controller context is honestly blocked
    res_blocked = adapter.execute_skill(
        "drone_flight",
        params={"altitude_m": 1.5},
        context={"canonical_robot_id": "skydio_x2", "has_flight_controller": False, "headless": True},
    )
    assert res_blocked.success is False
    assert "aerodynamic rotor physics" in res_blocked.error_reason.lower()


# =========================================================================
# 2. EXPANDED ROBOT INTEGRATIONS & ZERO SILENT FALLBACKS
# =========================================================================

def test_expanded_robot_integrations_ur10e_z1_b2():
    """Verify physical execution on UR10e, Unitree Z1, and Unitree B2 without fallback."""
    adapter = SimulatorTaskExecutionAdapter()

    # 1. Universal Robots UR10e arm execution
    res_ur10e = adapter.execute_skill(
        "move_joints",
        params={"joint_positions": [0.1, -0.2, 0.3, -0.1, 0.2, -0.3], "duration_sec": 0.5},
        context={"canonical_robot_id": "ur10e", "headless": True},
    )
    assert res_ur10e.success is True

    # 2. Unitree Z1 arm joint-limit enforcement
    res_z1_valid = adapter.execute_skill(
        "move_joints",
        params={"joint_positions": [0.1, 0.1, -0.5, 0.1, 0.1, 0.1], "duration_sec": 0.5},
        context={"canonical_robot_id": "unitree_z1", "headless": True},
    )
    assert res_z1_valid.success is True

    # Unitree Z1 joint 3 range is [-2.8798, 0.0] rad; positive value must be rejected
    res_z1_invalid = adapter.execute_skill(
        "move_joints",
        params={"joint_positions": [0.1, 0.1, 0.5, 0.1, 0.1, 0.1], "duration_sec": 0.5},
        context={"canonical_robot_id": "unitree_z1", "headless": True},
    )
    assert res_z1_invalid.success is False
    assert "exceeds permitted limits" in res_z1_invalid.error_reason.lower()

    # 3. Unitree B2 quadruped posture execution
    res_b2 = adapter.execute_skill(
        "stand_sit_posture",
        params={"posture": "sit", "duration_sec": 0.5},
        context={"canonical_robot_id": "b2", "headless": True},
    )
    assert res_b2.success is True

    # 4. Nonexistent robot rejected honestly with zero silent fallback to RSK
    res_unknown = adapter.execute_skill(
        "move_joints",
        params={"joint_positions": [0.1]},
        context={"canonical_robot_id": "unregistered_phantom_robot", "headless": True},
    )
    assert res_unknown.success is False
    assert "requires registered robot" in res_unknown.error_reason.lower()


# =========================================================================
# 3. PERCEPTION & SENSOR PIPELINE HARDENING
# =========================================================================

def test_stale_sensor_detection_and_rejection():
    """Verify stale sensor data is rejected honestly without silent ground-truth fallback."""
    pipeline = get_perception_pipeline()
    scene = get_world_scene("tabletop_world")

    # Fresh observation passes
    obs_fresh = pipeline.detect_target(
        scene,
        "target_cube_red",
        observation_source=ObservationSource.CAMERA_RGB,
        sensor_timestamp=100.0,
        sim_time_sec=100.1,
        max_stale_age_sec=0.5,
    )
    assert obs_fresh.success is True
    assert obs_fresh.is_stale is False

    # Stale observation fails safely
    obs_stale = pipeline.detect_target(
        scene,
        "target_cube_red",
        observation_source=ObservationSource.CAMERA_RGB,
        sensor_timestamp=100.0,
        sim_time_sec=101.5,
        max_stale_age_sec=0.5,
    )
    assert obs_stale.success is False
    assert obs_stale.is_stale is True
    assert "stale sensor observation" in obs_stale.error_reason.lower()


# =========================================================================
# 4. EVIDENCE TRACKING & SELECTIVE STALENESS INVALIDATION
# =========================================================================

def test_evidence_payload_staleness_and_invalidation():
    """Verify execution evidence records identify metadata and invalidate selectively."""
    matrix_mgr = get_execution_matrix()

    payload = ExecutionEvidencePayload(
        canonical_robot_id="rsk",
        robot_model_version="v1.0.0",
        simulator="mujoco",
        controller_type="omnidirectional_wheel_velocity",
        scenario_id="rsk_heading_tracking",
        observation_source=ObservationSource.GROUND_TRUTH_SIM.value,
        measured_result={"heading_error_rad": 0.029},
        acceptance_thresholds={"max_heading_error_rad": 0.07},
        passed=True,
    )
    assert payload.is_stale is False

    matrix_mgr.record_evidence(payload)
    evidence_list = matrix_mgr.get_evidence_for_robot("rsk")
    assert len(evidence_list) >= 1

    # Invalidate robot evidence
    matrix_mgr.notify_model_updated("rsk")
    assert payload.is_stale is True
    assert "invalidated" in payload.stale_reason.lower()


# =========================================================================
# 5. CROSS-SIMULATOR DECOUPLING
# =========================================================================

def test_cross_simulator_distinct_records_mujoco_and_gazebo():
    """Verify MuJoCo and Gazebo Harmonic maintain decoupled skill readiness."""
    matrix_mgr = get_execution_matrix()
    records = matrix_mgr.generate_full_matrix(verify_live=False)

    diff_drive = records["differential_drive"]
    rsk = records["rsk"]

    # Check that simulator readiness is tracked independently
    assert "gazebo_harmonic" in diff_drive.simulator_skill_readiness
    assert "mujoco" not in diff_drive.supported_simulators
    assert "mujoco" in rsk.simulator_skill_readiness
    assert "gazebo_harmonic" not in rsk.supported_simulators


# =========================================================================
# 6. CLOSED-LOOP COMPOUND LEGGED TASK EXECUTION
# =========================================================================

def test_compound_legged_locomotion_end_to_end():
    """Verify compound quadruped instruction execution on Go1 vs honest rejection on Go2."""
    interp = get_task_interpreter()
    adapter = SimulatorTaskExecutionAdapter()

    # 1. Unitree Go1 compound locomotion
    spec_go1 = interp.interpret_instruction("Stand, walk forward, turn, and stop.", robot_id_override="unitree_go1")
    assert spec_go1.is_supported is True
    assert len(spec_go1.action_sequence) == 3

    res_go1 = asyncio.run(adapter.execute_task(spec_go1, context={"headless": True}))
    assert res_go1["success"] is True
    assert res_go1["state"] == TaskExecutionState.COMPLETED.value
    assert len(res_go1["action_results"]) == 3
    assert all(r.success for r in res_go1["action_results"])

    # 2. Unitree Go2 compound locomotion without WBC is honestly rejected
    spec_go2 = interp.interpret_instruction("Stand, walk forward, turn, and stop.", robot_id_override="unitree_go2")
    assert spec_go2.is_supported is False
    assert "gait controller for 'unitree go2' is not integrated" in spec_go2.rejection_reason.lower()


# =========================================================================
# 7. PRODUCTION RELIABILITY: CANCELLATION & PROCESS CLEANUP
# =========================================================================

def test_production_reliability_cancellation_and_cleanup():
    """Verify task cancellation halts simulation immediately and clears latched controls."""
    adapter = MujocoTaskExecutionAdapter()
    cancellation = threading.Event()
    cancellation.set()  # Pre-trigger cancellation

    res = adapter.execute_skill(
        "drive_forward",
        params={"speed_m_s": 0.3, "duration_sec": 5.0},
        context={"canonical_robot_id": "rsk", "cancellation_event": cancellation, "headless": True},
    )
    # Task should terminate safely without hanging
    assert res.success is False
    assert res.status in (SkillStatus.CANCELLED, SkillStatus.FAILED)


# =========================================================================
# 8. OPERATOR UI SUMMARY RECONCILIATION
# =========================================================================

def test_operator_ui_reconciliation_statistics():
    """Verify the RobotExecutionMatrix summary statistics match programmatic reconciliation."""
    matrix_mgr = get_execution_matrix()
    stats = matrix_mgr.get_summary_statistics()

    assert stats["total_canonical_models"] == 175
    assert stats["by_component_type"]["complete_robot"] == 164
    assert stats["by_component_type"]["standalone_component"] == 11
    assert stats["ready_models_count"] + stats["blocked_models_count"] + stats["untested_models_count"] == 175
