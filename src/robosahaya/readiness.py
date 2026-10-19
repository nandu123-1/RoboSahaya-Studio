"""Authoritative Multi-Dimensional Robot Readiness Engine for RoboSahaya Studio.

Separates independent dimensions of robot readiness:
1. Catalogue status
2. Cloud availability
3. Local cache status
4. Description validity
5. Simulator compatibility
6. Simulation readiness
7. Controller readiness
8. Sensor readiness
9. Task verification
10. Execution readiness (task-specific composite decision)

Mandatory rule: Never compress independent dimensions into one misleading status.
A downloaded model is not validated; a validated model is not simulation-ready;
a running model is not controller-ready. Every positive status requires evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CatalogueStatus(str, Enum):
    UNKNOWN = "unknown"
    CATALOGUED = "catalogued"
    REGISTERED = "registered"


class CloudAvailability(str, Enum):
    UNAVAILABLE = "unavailable"
    LOCAL_ONLY = "local_only"
    FETCH_PENDING = "fetch_pending"
    CLOUD_AVAILABLE = "cloud_available"


class LocalCacheStatus(str, Enum):
    NOT_CACHED = "not_cached"
    DOWNLOADING = "downloading"
    CACHE_VERIFIED = "cache_verified"
    CORRUPTED = "corrupted"
    INTEGRITY_FAILED = "integrity_failed"


class DescriptionValidity(str, Enum):
    UNCHECKED = "unchecked"
    MODEL_VALIDATED = "model_validated"
    SYNTAX_ERROR = "syntax_error"
    MISSING_ASSETS = "missing_assets"


class SimulatorCompatibility(str, Enum):
    UNKNOWN = "unknown"
    COMPATIBLE = "compatible"
    SIMULATOR_UNAVAILABLE = "simulator_unavailable"
    INCOMPATIBLE_FORMAT = "incompatible_format"


class SimulationReadiness(str, Enum):
    UNTESTED = "untested"
    SIMULATION_READY = "simulation_ready"
    LOAD_FAILED = "load_failed"
    STEP_FAILED = "step_failed"


class ControllerReadiness(str, Enum):
    UNTESTED = "untested"
    CONTROL_READY = "control_ready"
    CONTROLLER_MISSING = "controller_missing"
    CONTROL_FAILED = "control_failed"


class SensorReadiness(str, Enum):
    UNTESTED = "untested"
    SENSOR_READY = "sensor_ready"
    SENSOR_MISSING = "sensor_missing"
    STREAM_FAILED = "stream_failed"


class TaskVerification(str, Enum):
    UNTESTED = "untested"
    TASK_VERIFIED = "task_verified"
    TASK_FAILED = "task_failed"
    STALE_EVIDENCE = "stale_evidence"


class ExecutionStatus(str, Enum):
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"
    PENDING_APPROVAL = "pending_approval"
    STALE_EVIDENCE = "stale_evidence"
    EXECUTION_READY = "execution_ready"


@dataclass
class CheckResult:
    """Individual verification check result."""
    check_id: str
    name: str
    status: str  # "PASSED", "FAILED", "BLOCKED", "NOT_RUN", "SKIPPED"
    observed: Any = None
    expected: Any = None
    error_message: Optional[str] = None
    is_test_double: bool = False
    evidence_uri: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "status": self.status,
            "observed": self.observed,
            "expected": self.expected,
            "error_message": self.error_message,
            "is_test_double": self.is_test_double,
            "evidence_uri": self.evidence_uri,
            "timestamp": self.timestamp,
        }


@dataclass
class ReadinessDimensions:
    """The 9 independent dimensions of robot readiness."""
    catalogue_status: CatalogueStatus = CatalogueStatus.CATALOGUED
    cloud_availability: CloudAvailability = CloudAvailability.LOCAL_ONLY
    local_cache_status: LocalCacheStatus = LocalCacheStatus.NOT_CACHED
    description_validity: DescriptionValidity = DescriptionValidity.UNCHECKED
    simulator_compatibility: SimulatorCompatibility = SimulatorCompatibility.UNKNOWN
    simulation_readiness: SimulationReadiness = SimulationReadiness.UNTESTED
    controller_readiness: ControllerReadiness = ControllerReadiness.UNTESTED
    sensor_readiness: SensorReadiness = SensorReadiness.UNTESTED
    task_verification: TaskVerification = TaskVerification.UNTESTED

    def to_dict(self) -> Dict[str, str]:
        return {
            "catalogue_status": self.catalogue_status.value,
            "cloud_availability": self.cloud_availability.value,
            "local_cache_status": self.local_cache_status.value,
            "description_validity": self.description_validity.value,
            "simulator_compatibility": self.simulator_compatibility.value,
            "simulation_readiness": self.simulation_readiness.value,
            "controller_readiness": self.controller_readiness.value,
            "sensor_readiness": self.sensor_readiness.value,
            "task_verification": self.task_verification.value,
        }


@dataclass
class RobotExecutionProfile:
    """Task-specific execution profile defining required capabilities."""
    robot_id: str
    model_version: str = "latest"
    simulator: str = "gazebo_harmonic"
    world: str = "obstacle_world"
    required_controllers: Tuple[str, ...] = ()
    required_sensors: Tuple[str, ...] = ()
    required_tests: Tuple[str, ...] = ()
    requires_ros2: bool = False
    max_evidence_age_seconds: float = 86400.0  # 24 hours


@dataclass
class ReadinessDecision:
    """Computed authoritative readiness decision with blockers and evidence."""
    robot_id: str
    model_version: str
    execution_status: ExecutionStatus
    is_ready: bool
    dimensions: ReadinessDimensions
    blockers: List[str] = field(default_factory=list)
    evidence_checks: List[CheckResult] = field(default_factory=list)
    summary_label: str = ""
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "model_version": self.model_version,
            "execution_status": self.execution_status.value,
            "is_ready": self.is_ready,
            "dimensions": self.dimensions.to_dict(),
            "blockers": self.blockers,
            "evidence_checks": [c.to_dict() for c in self.evidence_checks],
            "summary_label": self.summary_label,
            "evaluated_at": self.evaluated_at,
        }


def evaluate_execution_readiness(
    profile: RobotExecutionProfile,
    dimensions: ReadinessDimensions,
    evidence_checks: Optional[List[CheckResult]] = None,
    evidence_timestamp: Optional[datetime] = None,
) -> ReadinessDecision:
    """Compute task-specific Execution Readiness honestly based on verified dimensions."""
    blockers: List[str] = []
    checks = list(evidence_checks or [])

    # 1. Catalogue identity check
    if dimensions.catalogue_status == CatalogueStatus.UNKNOWN:
        blockers.append("Model is not registered in catalogue.")

    # 2. Local cache check
    if dimensions.local_cache_status != LocalCacheStatus.CACHE_VERIFIED:
        if dimensions.local_cache_status == LocalCacheStatus.NOT_CACHED:
            blockers.append("Model assets are not cached locally. Run 'Prepare / Cache Description' first.")
        elif dimensions.local_cache_status == LocalCacheStatus.DOWNLOADING:
            blockers.append("Model assets are currently downloading.")
        elif dimensions.local_cache_status == LocalCacheStatus.INTEGRITY_FAILED:
            blockers.append("Local asset integrity check failed (hash mismatch).")
        elif dimensions.local_cache_status == LocalCacheStatus.CORRUPTED:
            blockers.append("Local cache package is corrupted.")

    # 3. Description validity check
    if dimensions.description_validity != DescriptionValidity.MODEL_VALIDATED:
        if dimensions.description_validity == DescriptionValidity.UNCHECKED:
            blockers.append("Model description has not been structurally validated.")
        elif dimensions.description_validity == DescriptionValidity.SYNTAX_ERROR:
            blockers.append("Model description has XML/Xacro/MJCF syntax errors.")
        elif dimensions.description_validity == DescriptionValidity.MISSING_ASSETS:
            blockers.append("Model references missing mesh or texture assets.")

    # 4. Simulator compatibility check
    if dimensions.simulator_compatibility != SimulatorCompatibility.COMPATIBLE:
        if dimensions.simulator_compatibility == SimulatorCompatibility.SIMULATOR_UNAVAILABLE:
            blockers.append(f"Simulator '{profile.simulator}' is not installed or available in this runtime.")
        elif dimensions.simulator_compatibility == SimulatorCompatibility.INCOMPATIBLE_FORMAT:
            blockers.append(f"Model format is incompatible with simulator '{profile.simulator}'.")
        else:
            blockers.append(f"Simulator compatibility for '{profile.simulator}' is unknown.")

    # 5. Simulation readiness check
    if dimensions.simulation_readiness != SimulationReadiness.SIMULATION_READY:
        if dimensions.simulation_readiness == SimulationReadiness.UNTESTED:
            blockers.append(f"Model has not passed simulation load/physics step tests in '{profile.simulator}'.")
        elif dimensions.simulation_readiness == SimulationReadiness.LOAD_FAILED:
            blockers.append("Simulator failed to load the model.")
        elif dimensions.simulation_readiness == SimulationReadiness.STEP_FAILED:
            blockers.append("Simulator physics stepping failed.")

    # 6. Controller readiness check
    if profile.required_controllers:
        if dimensions.controller_readiness != ControllerReadiness.CONTROL_READY:
            if dimensions.controller_readiness == ControllerReadiness.UNTESTED:
                blockers.append(f"Required controllers {profile.required_controllers} have not been verified.")
            elif dimensions.controller_readiness == ControllerReadiness.CONTROLLER_MISSING:
                blockers.append(f"Required controllers {profile.required_controllers} are missing from configuration.")
            elif dimensions.controller_readiness == ControllerReadiness.CONTROL_FAILED:
                blockers.append(f"Actuator control response test failed.")

    # 7. Sensor readiness check
    if profile.required_sensors:
        if dimensions.sensor_readiness != SensorReadiness.SENSOR_READY:
            if dimensions.sensor_readiness == SensorReadiness.UNTESTED:
                blockers.append(f"Required sensors {profile.required_sensors} have not been verified.")
            elif dimensions.sensor_readiness == SensorReadiness.SENSOR_MISSING:
                blockers.append(f"Required sensors {profile.required_sensors} are missing from model.")
            elif dimensions.sensor_readiness == SensorReadiness.STREAM_FAILED:
                blockers.append(f"Sensor telemetry stream failed.")

    # 8. Task verification check
    if profile.required_tests:
        if dimensions.task_verification != TaskVerification.TASK_VERIFIED:
            if dimensions.task_verification == TaskVerification.UNTESTED:
                blockers.append(f"Task verification tests {profile.required_tests} have not been executed.")
            elif dimensions.task_verification == TaskVerification.TASK_FAILED:
                blockers.append("One or more required verification tests failed.")
            elif dimensions.task_verification == TaskVerification.STALE_EVIDENCE:
                blockers.append("Verification evidence is stale and must be re-run.")

    # 9. Evidence freshness check
    is_stale = False
    if evidence_timestamp is not None:
        now = datetime.now(timezone.utc)
        age = (now - evidence_timestamp).total_seconds()
        if age > profile.max_evidence_age_seconds:
            is_stale = True
            blockers.append(f"Evidence is stale ({int(age // 3600)} hours old; max {int(profile.max_evidence_age_seconds // 3600)}h).")

    # Final readiness synthesis
    if is_stale:
        status = ExecutionStatus.STALE_EVIDENCE
        label = "⚠️ Evidence Stale — Re-verification Required"
        ready = False
    elif not blockers:
        status = ExecutionStatus.EXECUTION_READY
        label = f"🟢 Execution Ready ({profile.simulator})"
        ready = True
    else:
        status = ExecutionStatus.BLOCKED
        label = f"🔴 Blocked ({len(blockers)} requirements unmet)"
        ready = False

    return ReadinessDecision(
        robot_id=profile.robot_id,
        model_version=profile.model_version,
        execution_status=status,
        is_ready=ready,
        dimensions=dimensions,
        blockers=blockers,
        evidence_checks=checks,
        summary_label=label,
    )
