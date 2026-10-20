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
from robosahaya.robot_catalog import RobotModelCatalog


def test_verifier_single_robot_rsk_offline():
    offline_cat = OfflineCatalogueProvider()
    verifier = RuntimeVerifier(catalogue_provider=offline_cat)

    report = verifier.verify_robot("rsk", sync_to_cloud=True)
    assert report.robot_id == "rsk"
    assert report.display_name == "Robot Soccer Kit (RSK)"
    assert report.simulator == "mujoco"
    assert report.decision.is_ready is True
    assert report.decision.execution_status == ExecutionStatus.EXECUTION_READY
    assert len(report.decision.blockers) == 0
    assert report.cloud_synced is True

    # Check 9 dimensions
    dims = report.decision.dimensions
    assert dims.catalogue_status == CatalogueStatus.REGISTERED
    assert dims.simulation_readiness == SimulationReadiness.SIMULATION_READY
    assert dims.simulator_compatibility == SimulatorCompatibility.COMPATIBLE

    # Check that offline catalogue recorded the verification run
    assert len(offline_cat._verification_runs) == 1
    run = offline_cat._verification_runs[0]
    assert run.robot_model_version_id == "rsk_v1.0.0"
    assert run.overall_result == "PASSED"
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
        assert report.decision.is_ready is True
        assert report.decision.execution_status == ExecutionStatus.EXECUTION_READY
        assert report.cloud_synced is True
        assert len(report.decision.blockers) == 0


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

    # 3. Unitree Go2 (Quadruped, WBC gait not integrated)
    r_go2 = verifier.verify_robot("unitree_go2", sync_to_cloud=True)
    assert r_go2.decision.is_ready is False
    assert r_go2.decision.execution_status == ExecutionStatus.BLOCKED
    assert any("simulation load/physics step" in b for b in r_go2.decision.blockers)

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
    assert len(reports) == 8
    ready_ids = {r.robot_id for r in reports if r.decision.is_ready}
    blocked_ids = {r.robot_id for r in reports if not r.decision.is_ready}

    assert ready_ids == {"differential_drive", "franka_panda", "so_arm101", "rsk"}
    assert blocked_ids == {"so_arm100", "ur5e", "unitree_go2", "skydio_x2"}


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
    assert r.decision.is_ready is True

    # Test format rows
    rows = catalog.format_verification_table_rows([r])
    assert len(rows) == 1
    assert rows[0][0] == "Robot Soccer Kit (RSK)"
    assert rows[0][2] == "mujoco"
    assert "Ready" in rows[0][4]
