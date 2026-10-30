"""Unit tests for multi-robot runtime verification engine."""
from pathlib import Path
import pytest

from robosahaya.cloud.offline_provider import OfflineCatalogueProvider
from robosahaya.cloud.verify_runtimes import RuntimeVerifier, get_runtime_verifier
from robosahaya.readiness import (
    CatalogueStatus,
    ControllerReadiness,
    ExecutionStatus,
    LocalCacheStatus,
    SimulationReadiness,
    SimulatorCompatibility,
)
from robosahaya.model_cache import ModelCacheManager
from robosahaya.robot_catalog import RobotModelCatalog


def test_verifier_single_robot_rsk_offline(tmp_path):
    offline_cat = OfflineCatalogueProvider()
    isolated_cache = ModelCacheManager(cache_root=tmp_path / "cache")
    verifier = RuntimeVerifier(catalogue_provider=offline_cat, model_cache=isolated_cache)

    report = verifier.verify_robot("rsk", sync_to_cloud=True)
    assert report.robot_id == "rsk"
    assert report.display_name == "Robot Soccer Kit (RSK)"
    assert report.simulator == "mujoco"
    assert report.decision.is_ready is False
    assert report.decision.execution_status == ExecutionStatus.BLOCKED
    assert report.decision.dimensions.local_cache_status == LocalCacheStatus.NOT_CACHED
    local_check = next(c for c in report.decision.evidence_checks if c.check_id == "local_cache")
    assert "integrity manifest" in local_check.error_message
    assert any("sensor" in b for b in report.decision.blockers)
    assert any("end_to_end_task" in b for b in report.decision.blockers)
    assert report.cloud_synced is True

    # Check 9 dimensions
    dims = report.decision.dimensions
    assert dims.catalogue_status == CatalogueStatus.REGISTERED
    assert dims.simulation_readiness == SimulationReadiness.SIMULATION_READY
    assert dims.controller_readiness == ControllerReadiness.CONTROL_READY
    assert dims.sensor_readiness.value == "untested"
    assert dims.task_verification.value == "untested"
    assert dims.simulator_compatibility == SimulatorCompatibility.COMPATIBLE

    actuator_check = next(c for c in report.decision.evidence_checks if c.check_id == "ctrl_readiness")
    assert actuator_check.status == "PASSED"
    assert actuator_check.observed["actuator"] == "wheel1_speed"
    assert actuator_check.observed["command"] == pytest.approx(5.44)
    assert actuator_check.observed["control_range"] == pytest.approx([-16.0, 16.0])
    assert actuator_check.observed["qpos_delta_norm"] > 0

    # Check that offline catalogue recorded the verification run
    assert len(offline_cat._verification_runs) == 1
    run = offline_cat._verification_runs[0]
    assert run.robot_model_version_id == "rsk_v1.0.0"
    assert run.overall_result == "BLOCKED"
    assert run.adapter_id == "mujoco"

    # Check that compatibility was recorded
    assert "rsk_mujoco_v1.0.0" in offline_cat._compatibilities
    compat = offline_cat._compatibilities["rsk_mujoco_v1.0.0"]
    assert compat.compatibility_status == "compatible"


def test_verifier_bundled_arms_offline():
    offline_cat = OfflineCatalogueProvider()
    verifier = RuntimeVerifier(catalogue_provider=offline_cat)

    for arm_id in ("franka_panda", "so_arm101", "differential_drive"):
        report = verifier.verify_robot(arm_id, sync_to_cloud=True)
        assert report.decision.is_ready is False
        assert report.decision.execution_status == ExecutionStatus.BLOCKED
        assert report.cloud_synced is True
        assert report.decision.dimensions.simulation_readiness == SimulationReadiness.UNTESTED
        assert report.decision.dimensions.controller_readiness == ControllerReadiness.UNTESTED


def test_verifier_unready_robots_honest_blockers():
    offline_cat = OfflineCatalogueProvider()
    verifier = RuntimeVerifier(catalogue_provider=offline_cat)

    # 1. SO-ARM100 (Description only, controller not bundled)
    r_so100 = verifier.verify_robot("so_arm100", sync_to_cloud=True)
    assert r_so100.decision.is_ready is False
    assert r_so100.decision.execution_status == ExecutionStatus.BLOCKED
    assert any("simulation load/physics step" in b for b in r_so100.decision.blockers)

    # 2. UR5e (Description only, ros2_control not bundled)
    r_ur5e = verifier.verify_robot("ur5e", sync_to_cloud=True)
    assert r_ur5e.decision.is_ready is False
    assert r_ur5e.decision.execution_status == ExecutionStatus.BLOCKED
    assert any("simulation load/physics step" in b for b in r_ur5e.decision.blockers)

    # 3. Unitree Go2 (Quadruped, WBC gait not integrated, physics stepping passes)
    r_go2 = verifier.verify_robot("unitree_go2", sync_to_cloud=True)
    assert r_go2.decision.is_ready is False
    assert r_go2.decision.execution_status == ExecutionStatus.BLOCKED
    assert r_go2.decision.dimensions.simulation_readiness in (
        SimulationReadiness.SIMULATION_READY,
        SimulationReadiness.UNTESTED,
    )

    # 4. Skydio X2 (Drone, simulator compatible, flight dynamics plugin missing)
    r_x2 = verifier.verify_robot("skydio_x2", sync_to_cloud=True)
    assert r_x2.decision.is_ready is False
    assert r_x2.decision.execution_status == ExecutionStatus.BLOCKED
    assert r_x2.decision.dimensions.simulator_compatibility == SimulatorCompatibility.COMPATIBLE
    assert r_x2.decision.dimensions.simulation_readiness == SimulationReadiness.UNTESTED
    assert r_x2.decision.dimensions.controller_readiness == ControllerReadiness.CONTROLLER_MISSING
    assert any("simulation load/physics step" in b for b in r_x2.decision.blockers)


def test_verifier_verify_all_robots():
    offline_cat = OfflineCatalogueProvider()
    verifier = RuntimeVerifier(catalogue_provider=offline_cat)

    reports = verifier.verify_all_robots(sync_to_cloud=True)
    expected_ids = {m.id for m in verifier.model_catalog.list_models(include_discovered=True)}
    expected_ids.update(m.id for m in verifier.robot_registry.list_models())
    assert {r.robot_id for r in reports} == expected_ids
    ready_ids = {r.robot_id for r in reports if r.decision.is_ready}
    blocked_ids = {r.robot_id for r in reports if not r.decision.is_ready}

    # This verifier does not run an end-to-end task or sensor sampling pass,
    # so it must not report task execution as ready for any discovered model.
    assert ready_ids == set()
    assert blocked_ids == expected_ids


def test_verifier_unknown_robot():
    offline_cat = OfflineCatalogueProvider()
    verifier = RuntimeVerifier(catalogue_provider=offline_cat)

    report = verifier.verify_robot("unknown_nonexistent_bot", sync_to_cloud=False)
    assert report.decision.is_ready is False
    assert report.decision.dimensions.catalogue_status == CatalogueStatus.UNKNOWN
    assert any("not registered in catalogue" in b for b in report.decision.blockers)


def test_robot_catalog_verification_helpers():
    catalog = RobotModelCatalog()
    # Test verify single
    r = catalog.verify_runtime("rsk", sync_to_cloud=False)
    assert r.robot_id == "rsk"
    assert r.decision.is_ready is False

    # Test format rows
    rows = catalog.format_verification_table_rows([r])
    assert len(rows) == 1
    assert rows[0][0] == "Robot Soccer Kit (RSK)"
    assert rows[0][2] == "mujoco"
    assert "Blocked" in rows[0][4]


def test_12_stage_verification_matrix_structure():
    """Verify that every verified robot report contains all 12 stages with valid states."""
    from robosahaya.cloud.verify_runtimes import VerificationStage, VerificationState
    offline_cat = OfflineCatalogueProvider()
    verifier = RuntimeVerifier(catalogue_provider=offline_cat)

    report = verifier.verify_robot("unitree_go2", sync_to_cloud=False)
    assert len(report.stages) == 12

    # Verify all 12 stages are present
    expected_stages = [stage.value for stage in VerificationStage]
    for st_id in expected_stages:
        assert st_id in report.stages
        stage_res = report.stages[st_id]
        assert isinstance(stage_res.state, VerificationState)
        assert stage_res.stage_name
        assert stage_res.message

    # Go2 physics stepping (Stage 8) must pass
    s8 = report.stages[VerificationStage.S8_PHYSICS_STEPPING_STABILITY.value]
    assert s8.state == VerificationState.PASSED

    # Verification state must reflect honest controller gating
    assert report.verification_state == VerificationState.BLOCKED_MISSING_CONTROLLER


def test_batch_verification_execution():
    """Verify bounded batch execution with callback tracking and resumable offset."""
    from robosahaya.cloud.verify_runtimes import VerificationState
    offline_cat = OfflineCatalogueProvider()
    verifier = RuntimeVerifier(catalogue_provider=offline_cat)

    progress_events = []

    def on_progress(evt):
        progress_events.append(evt)

    batch_res = verifier.verify_batch(
        batch_size=5,
        offset=0,
        sync_to_cloud=False,
        progress_callback=on_progress,
    )

    assert batch_res["total"] > 5
    assert batch_res["processed"] == 5
    assert batch_res["offset"] == 0
    assert batch_res["has_more"] is True
    assert batch_res["next_offset"] == 5
    assert len(batch_res["reports"]) == 5
    assert len(progress_events) == 10  # 5 verifying + 5 completed

    for rep in batch_res["reports"]:
        assert len(rep.stages) == 12
        assert isinstance(rep.verification_state, VerificationState)


def test_verification_filter_states():
    """Verify filtering of verification table rows by verification state."""
    catalog = RobotModelCatalog()
    offline_cat = OfflineCatalogueProvider()
    verifier = RuntimeVerifier(catalogue_provider=offline_cat)

    r_go2 = verifier.verify_robot("unitree_go2", sync_to_cloud=False)
    r_so100 = verifier.verify_robot("so_arm100", sync_to_cloud=False)

    reports = [r_go2, r_so100]

    all_rows = catalog.format_verification_table_rows(reports, filter_state="All")
    assert len(all_rows) == 2

    ctrl_rows = catalog.format_verification_table_rows(reports, filter_state="Blocked (Controller)")
    assert len(ctrl_rows) == 1
    assert ctrl_rows[0][0] == "Unitree Go2"

    src_rows = catalog.format_verification_table_rows(reports, filter_state="Blocked (Source / Cache)")
    assert len(src_rows) == 1
    assert "SO-ARM100" in src_rows[0][0]