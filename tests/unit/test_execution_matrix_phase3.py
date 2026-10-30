"""Unit tests for RoboSahaya Studio Phase 3 execution matrix extensions.

Verifies:
1. Extended SkillReadinessState (7 explicit lifecycle states).
2. Per-skill readiness mapping and simulator readiness tracking across canonical models.
3. Dynamic cache invalidation and notification methods.
4. Traceable ModelFidelityDiagnostics and ModelFidelityStatus tracking.
5. Dynamic filtering by skill and fidelity in matrix table formatting.
"""
import pytest
from robosahaya.execution_matrix import (
    SkillReadinessState,
    ModelFidelityStatus,
    ModelFidelityDiagnostics,
    RobotExecutionMatrixRecord,
    get_execution_matrix,
)


def test_skill_readiness_state_lifecycle_enum():
    expected_states = {
        "not_implemented",
        "not_tested",
        "blocked_by_dependency",
        "blocked_by_controller",
        "ready_for_testing",
        "verification_failed",
        "verified",
    }
    actual_states = {state.value for state in SkillReadinessState}
    assert actual_states == expected_states


def test_execution_matrix_contains_per_skill_readiness():
    matrix = get_execution_matrix()
    records = matrix.generate_full_matrix(verify_live=False)

    assert len(records) == 175
    panda = records.get("franka_panda")
    assert panda is not None
    assert isinstance(panda.skill_readiness, dict)
    assert panda.skill_readiness.get("move_joints") == SkillReadinessState.VERIFIED.value
    assert panda.skill_readiness.get("gripper_control") == SkillReadinessState.VERIFIED.value
    assert panda.skill_readiness.get("pick_and_place") == SkillReadinessState.VERIFIED.value

    # Gated skills for quadrupeds and drones
    go2 = records.get("unitree_go2")
    assert go2 is not None
    assert go2.skill_readiness.get("stand_sit_posture") == SkillReadinessState.VERIFIED.value
    assert go2.skill_readiness.get("legged_gait") in (SkillReadinessState.BLOCKED_BY_CONTROLLER.value, SkillReadinessState.BLOCKED_BY_DEPENDENCY.value)

    skydio = records.get("skydio_x2")
    assert skydio is not None
    assert skydio.skill_readiness.get("drone_flight") in (SkillReadinessState.BLOCKED_BY_CONTROLLER.value, SkillReadinessState.BLOCKED_BY_DEPENDENCY.value)


def test_matrix_cache_invalidation_and_notifications():
    matrix = get_execution_matrix()
    records = matrix.generate_full_matrix(verify_live=False)

    # Invalidate single robot evidence
    matrix.invalidate_robot_evidence("franka_panda")
    assert "franka_panda" not in matrix._record_cache

    # Notify model updated
    matrix.notify_model_updated("franka_panda")
    assert "franka_panda" not in matrix._record_cache

    # Invalidate all evidence
    records_refreshed = matrix.generate_full_matrix(verify_live=False)
    assert len(records_refreshed) == 175
    matrix.invalidate_all_evidence()
    assert len(matrix._record_cache) == 0


def test_model_fidelity_diagnostics_structure():
    diag = ModelFidelityDiagnostics(
        fidelity_status=ModelFidelityStatus.REPAIRED_SYNTAX_ONLY.value,
        synthetic_actuators=["joint1_ctrl", "joint2_ctrl"],
        injected_damping={"joint1": 0.1, "joint2": 0.1},
        repaired_inertias=[],
        syntax_normalizations=["Added dummy root link"],
    )
    d = diag.to_dict()
    assert d["fidelity_status"] == ModelFidelityStatus.REPAIRED_SYNTAX_ONLY.value
    assert len(d["synthetic_actuators"]) == 2
    assert d["injected_damping"]["joint1"] == 0.1


def test_execution_matrix_filters_by_skill_and_fidelity():
    matrix = get_execution_matrix()
    records = matrix.generate_full_matrix(verify_live=False)

    # Filter by verified skill
    rows_reach = matrix.format_matrix_table_rows(
        matrix=records,
        filter_skill="reach_target_pose",
    )
    assert len(rows_reach) > 0
    ids_reach = [r[0] for r in rows_reach]
    assert "franka_panda" in ids_reach

    # Filter by fidelity
    rows_faithful = matrix.format_matrix_table_rows(
        matrix=records,
        filter_fidelity="Faithful",
    )
    assert len(rows_faithful) > 0
