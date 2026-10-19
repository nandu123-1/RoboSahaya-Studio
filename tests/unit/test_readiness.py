"""Unit tests for Authoritative Multi-Dimensional Readiness Engine."""
from datetime import datetime, timedelta, timezone
import pytest

from robosahaya.readiness import (
    CatalogueStatus,
    CheckResult,
    CloudAvailability,
    ControllerReadiness,
    DescriptionValidity,
    ExecutionStatus,
    LocalCacheStatus,
    ReadinessDimensions,
    RobotExecutionProfile,
    SensorReadiness,
    SimulationReadiness,
    SimulatorCompatibility,
    TaskVerification,
    evaluate_execution_readiness,
)


def test_cloud_availability_alone_does_not_imply_readiness():
    """Rule 2.4: A model in cloud storage is not automatically execution ready."""
    dims = ReadinessDimensions(
        catalogue_status=CatalogueStatus.CATALOGUED,
        cloud_availability=CloudAvailability.CLOUD_AVAILABLE,
        local_cache_status=LocalCacheStatus.NOT_CACHED,
        description_validity=DescriptionValidity.UNCHECKED,
    )
    profile = RobotExecutionProfile(robot_id="rsk")
    decision = evaluate_execution_readiness(profile, dims)

    assert decision.is_ready is False
    assert decision.execution_status == ExecutionStatus.BLOCKED
    assert any("not cached locally" in b for b in decision.blockers)


def test_download_alone_does_not_imply_simulation_ready():
    """Rule 2.4: Downloaded and cached does not mean simulation-ready."""
    dims = ReadinessDimensions(
        catalogue_status=CatalogueStatus.CATALOGUED,
        cloud_availability=CloudAvailability.CLOUD_AVAILABLE,
        local_cache_status=LocalCacheStatus.CACHE_VERIFIED,
        description_validity=DescriptionValidity.MODEL_VALIDATED,
        simulator_compatibility=SimulatorCompatibility.COMPATIBLE,
        simulation_readiness=SimulationReadiness.UNTESTED,
    )
    profile = RobotExecutionProfile(robot_id="rsk", simulator="mujoco")
    decision = evaluate_execution_readiness(profile, dims)

    assert decision.is_ready is False
    assert any("simulation load/physics step tests" in b for b in decision.blockers)


def test_model_load_does_not_imply_controller_ready():
    """Rule 2.4: A model loading does not mean its controllers work."""
    dims = ReadinessDimensions(
        catalogue_status=CatalogueStatus.REGISTERED,
        local_cache_status=LocalCacheStatus.CACHE_VERIFIED,
        description_validity=DescriptionValidity.MODEL_VALIDATED,
        simulator_compatibility=SimulatorCompatibility.COMPATIBLE,
        simulation_readiness=SimulationReadiness.SIMULATION_READY,
        controller_readiness=ControllerReadiness.UNTESTED,
    )
    profile = RobotExecutionProfile(
        robot_id="differential_drive",
        required_controllers=("diff_drive_controller",)
    )
    decision = evaluate_execution_readiness(profile, dims)

    assert decision.is_ready is False
    assert any("Required controllers" in b for b in decision.blockers)


def test_controller_ready_does_not_imply_sensor_ready():
    """Rule 2.4: Actuator movement does not imply LiDAR or camera streaming."""
    dims = ReadinessDimensions(
        catalogue_status=CatalogueStatus.REGISTERED,
        local_cache_status=LocalCacheStatus.CACHE_VERIFIED,
        description_validity=DescriptionValidity.MODEL_VALIDATED,
        simulator_compatibility=SimulatorCompatibility.COMPATIBLE,
        simulation_readiness=SimulationReadiness.SIMULATION_READY,
        controller_readiness=ControllerReadiness.CONTROL_READY,
        sensor_readiness=SensorReadiness.STREAM_FAILED,
    )
    profile = RobotExecutionProfile(
        robot_id="differential_drive",
        required_controllers=("diff_drive_controller",),
        required_sensors=("2d_lidar",)
    )
    decision = evaluate_execution_readiness(profile, dims)

    assert decision.is_ready is False
    assert any("Sensor telemetry stream failed" in b for b in decision.blockers)


def test_fully_verified_profile_is_execution_ready():
    """All required dimensions verified -> EXECUTION_READY."""
    dims = ReadinessDimensions(
        catalogue_status=CatalogueStatus.REGISTERED,
        cloud_availability=CloudAvailability.LOCAL_ONLY,
        local_cache_status=LocalCacheStatus.CACHE_VERIFIED,
        description_validity=DescriptionValidity.MODEL_VALIDATED,
        simulator_compatibility=SimulatorCompatibility.COMPATIBLE,
        simulation_readiness=SimulationReadiness.SIMULATION_READY,
        controller_readiness=ControllerReadiness.CONTROL_READY,
        sensor_readiness=SensorReadiness.SENSOR_READY,
        task_verification=TaskVerification.TASK_VERIFIED,
    )
    profile = RobotExecutionProfile(
        robot_id="differential_drive",
        required_controllers=("diff_drive_controller",),
        required_sensors=("2d_lidar",),
        required_tests=("test_lidar_streaming",)
    )
    decision = evaluate_execution_readiness(profile, dims)

    assert decision.is_ready is True
    assert decision.execution_status == ExecutionStatus.EXECUTION_READY
    assert len(decision.blockers) == 0
    assert "Execution Ready" in decision.summary_label


def test_stale_evidence_invalidates_readiness():
    """Rule 9.3: Evidence older than max_evidence_age_seconds marks STALE_EVIDENCE."""
    dims = ReadinessDimensions(
        catalogue_status=CatalogueStatus.REGISTERED,
        local_cache_status=LocalCacheStatus.CACHE_VERIFIED,
        description_validity=DescriptionValidity.MODEL_VALIDATED,
        simulator_compatibility=SimulatorCompatibility.COMPATIBLE,
        simulation_readiness=SimulationReadiness.SIMULATION_READY,
        controller_readiness=ControllerReadiness.CONTROL_READY,
        sensor_readiness=SensorReadiness.SENSOR_READY,
        task_verification=TaskVerification.TASK_VERIFIED,
    )
    profile = RobotExecutionProfile(
        robot_id="differential_drive",
        max_evidence_age_seconds=3600.0  # 1 hour
    )
    stale_time = datetime.now(timezone.utc) - timedelta(hours=5)
    decision = evaluate_execution_readiness(profile, dims, evidence_timestamp=stale_time)

    assert decision.is_ready is False
    assert decision.execution_status == ExecutionStatus.STALE_EVIDENCE
    assert any("Evidence is stale" in b for b in decision.blockers)
