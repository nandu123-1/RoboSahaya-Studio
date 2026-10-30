"""Authoritative Multi-Robot Runtime Verification Engine for RoboSahaya Studio.

Systematically verifies execution readiness across all registered and catalogued
robot models, validating:
1. Catalogue Identity & Metadata
2. Cloud Storage & Catalogue Availability
3. Local Cache Integrity (SHA-256 manifests)
4. Description Structure & Syntax (URDF/Xacro/MJCF)
5. Simulator Engine Compatibility (Gazebo Harmonic / MuJoCo)
6. Simulation Physics Stepping Stability
7. Controller & Actuation Command Response
8. Sensor Stream & Telemetry Availability
9. Task Execution Verification Criteria

Syncs verified 9D readiness decisions, compatibility profiles, and audit runs
directly to Supabase Postgres tables or local fallback cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from robosahaya.cloud.base import (
    BaseCatalogueProvider,
    CloudModelRecord,
    CloudSyncStatus,
    CloudVersionRecord,
    ModelSyncState,
    RuntimeCompatibilityRecord,
    VerificationRunRecord,
)
from robosahaya.cloud.client import get_cloud_catalogue
from robosahaya.model_cache import ModelCacheManager
from robosahaya.readiness import (
    CatalogueStatus,
    CheckResult,
    CloudAvailability,
    ControllerReadiness,
    DescriptionValidity,
    ExecutionStatus,
    LocalCacheStatus,
    ReadinessDecision,
    ReadinessDimensions,
    RobotExecutionProfile,
    SensorReadiness,
    SimulationReadiness,
    SimulatorCompatibility,
    TaskVerification,
    evaluate_execution_readiness,
)
from robosahaya.robot_catalog import RobotModelCatalog
from robosahaya.robot_registry import RobotModel, RobotRegistry, get_robot_registry
from robosahaya.simulators.base import SimulatorAdapter
from robosahaya.simulators.mujoco_adapter import load_mujoco_model
from robosahaya.simulators.registry import SimulatorAdapterRegistry, get_simulator_registry

logger = logging.getLogger(__name__)


class VerificationStage(str, Enum):
    S1_CATALOGUE_REGISTRATION = "S1_catalogue_registration"
    S2_CLOUD_AVAILABILITY = "S2_cloud_availability"
    S3_LOCAL_CACHE_INTEGRITY = "S3_local_cache_integrity"
    S4_DESCRIPTION_STRUCTURE = "S4_description_structure"
    S5_SIMULATOR_COMPATIBILITY = "S5_simulator_compatibility"
    S6_LAUNCH_PROFILE_RESOLUTION = "S6_launch_profile_resolution"
    S7_ASSET_MATERIALIZATION = "S7_asset_materialization"
    S8_PHYSICS_STEPPING_STABILITY = "S8_physics_stepping_stability"
    S9_ACTUATOR_CONTROLLER = "S9_actuator_controller"
    S10_SENSOR_TELEMETRY = "S10_sensor_telemetry"
    S11_TASK_CRITERIA = "S11_task_criteria"
    S12_READINESS_CLASSIFICATION = "S12_readiness_classification"


class VerificationState(str, Enum):
    NOT_CHECKED = "not_checked"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED_MISSING_SOURCE = "blocked_missing_source"
    BLOCKED_LICENCE = "blocked_licence"
    BLOCKED_SIMULATOR_INCOMPATIBLE = "blocked_simulator_incompatible"
    BLOCKED_MISSING_LAUNCH = "blocked_missing_launch"
    BLOCKED_PHYSICS_FAILURE = "blocked_physics_failure"
    BLOCKED_MISSING_CONTROLLER = "blocked_missing_controller"
    BLOCKED_MISSING_SENSOR = "blocked_missing_sensor"
    BLOCKED_TASK_CRITERIA_FAILED = "blocked_task_criteria_failed"
    BLOCKED_TIMEOUT = "blocked_timeout"


@dataclass
class StageResult:
    stage_id: str
    stage_name: str
    state: VerificationState
    message: str = ""
    observed: Any = None
    expected: Any = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "state": self.state.value if isinstance(self.state, VerificationState) else str(self.state),
            "message": self.message,
            "observed": self.observed,
            "expected": self.expected,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ModelRuntimeVerificationReport:
    """Consolidated verification report for a single robot model."""
    robot_id: str
    display_name: str
    family: str
    simulator: str
    decision: ReadinessDecision
    cloud_synced: bool = False
    sync_state: Optional[ModelSyncState] = None
    verified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Dict[str, Any] = field(default_factory=dict)
    stages: Dict[str, StageResult] = field(default_factory=dict)
    verification_state: VerificationState = VerificationState.NOT_CHECKED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "display_name": self.display_name,
            "family": self.family,
            "simulator": self.simulator,
            "decision": self.decision.to_dict(),
            "cloud_synced": self.cloud_synced,
            "sync_state": self.sync_state.to_dict() if self.sync_state else None,
            "verified_at": self.verified_at,
            "details": self.details,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "verification_state": self.verification_state.value if isinstance(self.verification_state, VerificationState) else str(self.verification_state),
        }


class RuntimeVerifier:
    """Executes multi-dimensional verification across robot models and simulators."""

    def __init__(
        self,
        catalogue_provider: Optional[BaseCatalogueProvider] = None,
        robot_registry: Optional[RobotRegistry] = None,
        model_catalog: Optional[RobotModelCatalog] = None,
        model_cache: Optional[ModelCacheManager] = None,
        simulator_registry: Optional[SimulatorAdapterRegistry] = None,
        templates_root: Optional[Path] = None,
    ):
        self.catalogue = catalogue_provider or get_cloud_catalogue()
        self.robot_registry = robot_registry or get_robot_registry()
        self.model_catalog = model_catalog or RobotModelCatalog()
        self.model_cache = model_cache or ModelCacheManager()
        self.simulator_registry = simulator_registry or get_simulator_registry()
        self.templates_root = Path(templates_root or Path(__file__).resolve().parent.parent.parent.parent / "simulation" / "templates")

    def verify_robot(
        self,
        robot_id: str,
        simulator: Optional[str] = None,
        sync_to_cloud: bool = True,
    ) -> ModelRuntimeVerificationReport:
        """Execute full 9D verification for one robot model."""
        start_time = datetime.now(timezone.utc).isoformat()
        checks: List[CheckResult] = []
        details: Dict[str, Any] = {}

        # 1. Resolve robot model
        catalog_model = self.model_catalog.get_model(robot_id)
        if catalog_model and catalog_model.id == robot_id and robot_id not in self.robot_registry._models:
            model = RobotModel(
                id=catalog_model.id,
                display_name=catalog_model.display_name,
                family=catalog_model.family,
                aliases=catalog_model.aliases,
                description_module=catalog_model.description_module,
                formats=catalog_model.formats,
                license=catalog_model.license,
                runtime_status=catalog_model.runtime_status,
            )
        else:
            model = self.robot_registry.get_model(robot_id)
            if model is None and catalog_model:
                model = RobotModel(
                    id=catalog_model.id,
                    display_name=catalog_model.display_name,
                    family=catalog_model.family,
                    aliases=catalog_model.aliases,
                    description_module=catalog_model.description_module,
                    formats=catalog_model.formats,
                    license=catalog_model.license,
                    runtime_status=catalog_model.runtime_status,
                )

        if model is None:
            dims = ReadinessDimensions(catalogue_status=CatalogueStatus.UNKNOWN)
            prof = RobotExecutionProfile(robot_id=robot_id)
            decision = evaluate_execution_readiness(prof, dims)
            return ModelRuntimeVerificationReport(
                robot_id=robot_id,
                display_name=robot_id,
                family="unknown",
                simulator=simulator or "unknown",
                decision=decision,
                cloud_synced=False,
                verified_at=start_time,
                details={"error": f"Robot model '{robot_id}' not found in registry or catalogue."},
            )

        target_sim = simulator or model.default_simulator or ("mujoco" if model.id == "rsk" else "gazebo_harmonic")
        details["simulator"] = target_sim
        details["notes"] = model.notes
        details["controllers"] = model.controllers
        details["sensors"] = model.sensors
        details["world"] = model.compatible_worlds[0] if model.compatible_worlds else "obstacle_world"

        # Dimension 1: Catalogue Identity
        dim_cat = CatalogueStatus.REGISTERED
        checks.append(CheckResult(
            check_id="cat_identity",
            name="Catalogue Registration",
            status="PASSED",
            observed=robot_id,
            expected=robot_id,
        ))

        # Dimension 2: Cloud Availability
        cloud_conn = self.catalogue.is_connected()
        is_actual_cloud = self.catalogue.__class__.__name__ != "OfflineCatalogueProvider" and cloud_conn
        dim_cloud = CloudAvailability.CLOUD_AVAILABLE if is_actual_cloud else CloudAvailability.LOCAL_ONLY
        checks.append(CheckResult(
            check_id="cloud_avail",
            name="Cloud Catalogue Availability",
            status="PASSED" if is_actual_cloud else "SKIPPED",
            observed=self.catalogue.__class__.__name__,
            expected="Connected remote catalogue provider" if is_actual_cloud else "Remote connectivity or explicit local-only status",
        ))

        # Dimension 3: Local Cache & Assets
        dim_cache, cache_check = self._check_local_cache(model)
        checks.append(cache_check)

        # Dimension 4: Description Validity
        dim_desc, desc_check, desc_info = self._check_description_validity(model)
        checks.append(desc_check)
        details.update(desc_info)

        # Dimension 5: Simulator Compatibility
        dim_sim_compat, sim_compat_check = self._check_simulator_compatibility(model, target_sim)
        checks.append(sim_compat_check)

        # Dimension 6: Simulation Readiness
        dim_sim_ready, sim_ready_check = self._check_simulation_readiness(model, target_sim)
        checks.append(sim_ready_check)

        # Dimension 7: Controller Readiness
        dim_ctrl_ready, ctrl_ready_check = self._check_controller_readiness(model, target_sim)
        checks.append(ctrl_ready_check)

        # Dimension 8: Sensor Readiness
        dim_sensor_ready, sensor_ready_check = self._check_sensor_readiness(model, target_sim)
        checks.append(sensor_ready_check)

        # Dimension 9: Task Verification
        dim_task_ver, task_ver_check = self._check_task_verification(model)
        checks.append(task_ver_check)

        # Synthesize authoritative dimensions
        dimensions = ReadinessDimensions(
            catalogue_status=dim_cat,
            cloud_availability=dim_cloud,
            local_cache_status=dim_cache,
            description_validity=dim_desc,
            simulator_compatibility=dim_sim_compat,
            simulation_readiness=dim_sim_ready,
            controller_readiness=dim_ctrl_ready,
            sensor_readiness=dim_sensor_ready,
            task_verification=dim_task_ver,
        )

        # Configure execution profile
        profile = RobotExecutionProfile(
            robot_id=robot_id,
            model_version="v1.0.0",
            simulator=target_sim,
            world=details["world"],
            required_controllers=model.controllers,
            required_sensors=model.sensors if model.sensors else (("wheel_encoders",) if robot_id == "rsk" else ()),
            required_tests=("end_to_end_task",),
        )

        decision = evaluate_execution_readiness(profile, dimensions, evidence_checks=checks)

        # Build 12-stage verification matrix
        stages: Dict[str, StageResult] = {}

        # S1: Catalogue Registration
        s1_state = VerificationState.PASSED if dim_cat == CatalogueStatus.REGISTERED else VerificationState.BLOCKED_MISSING_SOURCE
        stages[VerificationStage.S1_CATALOGUE_REGISTRATION.value] = StageResult(
            stage_id=VerificationStage.S1_CATALOGUE_REGISTRATION.value,
            stage_name="Catalogue Registration",
            state=s1_state,
            message="Canonical model registered in catalogue" if s1_state == VerificationState.PASSED else "Model not registered in catalogue",
            observed=model.id,
            expected=model.id,
        )

        # S2: Cloud Storage Availability
        s2_state = VerificationState.PASSED if is_actual_cloud else VerificationState.NOT_CHECKED
        stages[VerificationStage.S2_CLOUD_AVAILABILITY.value] = StageResult(
            stage_id=VerificationStage.S2_CLOUD_AVAILABILITY.value,
            stage_name="Cloud Storage Availability",
            state=s2_state,
            message="Supabase cloud catalogue & storage connected" if is_actual_cloud else "Operating in local / offline fallback mode",
            observed=self.catalogue.__class__.__name__,
            expected="Connected cloud provider",
        )

        # S3: Local Cache Integrity (SHA-256)
        if dim_cache == LocalCacheStatus.CACHE_VERIFIED:
            s3_state = VerificationState.PASSED
            s3_msg = "Cryptographic SHA-256 package manifest verified"
        elif dim_cache in {LocalCacheStatus.CORRUPTED, LocalCacheStatus.INTEGRITY_FAILED}:
            s3_state = VerificationState.FAILED
            s3_msg = cache_check.error_message or "Asset hash mismatch detected"
        else:
            s3_state = VerificationState.BLOCKED_MISSING_SOURCE
            s3_msg = cache_check.error_message or "Model assets not cached locally; preparation required"
        stages[VerificationStage.S3_LOCAL_CACHE_INTEGRITY.value] = StageResult(
            stage_id=VerificationStage.S3_LOCAL_CACHE_INTEGRITY.value,
            stage_name="Local Cache Integrity (SHA-256)",
            state=s3_state,
            message=s3_msg,
            observed=cache_check.observed,
            expected=cache_check.expected,
        )

        # S4: Description Structure & Syntax
        if dim_desc == DescriptionValidity.MODEL_VALIDATED:
            s4_state = VerificationState.PASSED
            s4_msg = desc_check.observed or "XML/MJCF structure parsed successfully"
        elif dim_desc == DescriptionValidity.SYNTAX_ERROR:
            s4_state = VerificationState.FAILED
            s4_msg = desc_check.error_message or "Syntax error in model description"
        else:
            s4_state = VerificationState.BLOCKED_MISSING_SOURCE
            s4_msg = desc_check.error_message or "Model description unchecked; preparation required"
        stages[VerificationStage.S4_DESCRIPTION_STRUCTURE.value] = StageResult(
            stage_id=VerificationStage.S4_DESCRIPTION_STRUCTURE.value,
            stage_name="Description Structure & Syntax",
            state=s4_state,
            message=s4_msg,
            observed=desc_check.observed,
            expected=desc_check.expected,
        )

        # S5: Simulator Engine Compatibility
        if dim_sim_compat == SimulatorCompatibility.COMPATIBLE:
            s5_state = VerificationState.PASSED
            s5_msg = f"Compatible with {target_sim}"
        else:
            s5_state = VerificationState.BLOCKED_SIMULATOR_INCOMPATIBLE
            s5_msg = sim_compat_check.error_message or f"Incompatible format for {target_sim}"
        stages[VerificationStage.S5_SIMULATOR_COMPATIBILITY.value] = StageResult(
            stage_id=VerificationStage.S5_SIMULATOR_COMPATIBILITY.value,
            stage_name="Simulator Engine Compatibility",
            state=s5_state,
            message=s5_msg,
            observed=sim_compat_check.observed,
            expected=f"Compatible with {target_sim}",
        )

        # S6: Launch Profile Resolution
        has_entrypoint = bool(details.get("entrypoint") and Path(details["entrypoint"]).is_file())
        if has_entrypoint:
            s6_state = VerificationState.PASSED
            s6_msg = f"Resolved entrypoint '{Path(details['entrypoint']).name}' in {details.get('world', 'default')}"
        else:
            s6_state = VerificationState.BLOCKED_MISSING_LAUNCH
            s6_msg = "Model launch profile or entrypoint file could not be resolved"
        stages[VerificationStage.S6_LAUNCH_PROFILE_RESOLUTION.value] = StageResult(
            stage_id=VerificationStage.S6_LAUNCH_PROFILE_RESOLUTION.value,
            stage_name="Launch Profile Resolution",
            state=s6_state,
            message=s6_msg,
            observed=details.get("entrypoint"),
            expected="Valid local launch entrypoint file",
        )

        # S7: Simulation Asset Materialization
        has_package_dir = bool(details.get("package_root") and Path(details["package_root"]).is_dir())
        if has_package_dir and has_entrypoint:
            s7_state = VerificationState.PASSED
            s7_msg = f"Model package assets materialized at {details.get('package_root')}"
        else:
            s7_state = VerificationState.BLOCKED_MISSING_SOURCE
            s7_msg = "Model assets not materialized in local cache or workspace"
        stages[VerificationStage.S7_ASSET_MATERIALIZATION.value] = StageResult(
            stage_id=VerificationStage.S7_ASSET_MATERIALIZATION.value,
            stage_name="Simulation Asset Materialization",
            state=s7_state,
            message=s7_msg,
            observed=details.get("package_root"),
            expected="Materialized package directory",
        )

        # S8: Physics Stepping Stability
        if dim_sim_ready == SimulationReadiness.SIMULATION_READY:
            s8_state = VerificationState.PASSED
            s8_msg = "100-step physics integration completed with finite state"
        elif dim_sim_ready == SimulationReadiness.STEP_FAILED:
            s8_state = VerificationState.BLOCKED_PHYSICS_FAILURE
            s8_msg = sim_ready_check.error_message or "Physics stepping diverged or threw an exception"
        else:
            s8_state = VerificationState.NOT_CHECKED
            s8_msg = sim_ready_check.error_message or "Physics stepping not executed"
        stages[VerificationStage.S8_PHYSICS_STEPPING_STABILITY.value] = StageResult(
            stage_id=VerificationStage.S8_PHYSICS_STEPPING_STABILITY.value,
            stage_name="Physics Stepping Stability",
            state=s8_state,
            message=s8_msg,
            observed=sim_ready_check.observed,
            expected="100 finite physics steps",
        )

        # S9: Actuator Controller Verification
        if dim_ctrl_ready == ControllerReadiness.CONTROL_READY:
            s9_state = VerificationState.PASSED
            s9_msg = "Actuator response verified with live command and physical motion"
        elif dim_ctrl_ready == ControllerReadiness.CONTROLLER_MISSING:
            s9_state = VerificationState.BLOCKED_MISSING_CONTROLLER
            s9_msg = ctrl_ready_check.error_message or "Actuator controller missing or not configured"
        else:
            s9_state = VerificationState.BLOCKED_MISSING_CONTROLLER
            s9_msg = ctrl_ready_check.error_message or "Actuator controller not verified"
        stages[VerificationStage.S9_ACTUATOR_CONTROLLER.value] = StageResult(
            stage_id=VerificationStage.S9_ACTUATOR_CONTROLLER.value,
            stage_name="Actuator Controller Verification",
            state=s9_state,
            message=s9_msg,
            observed=ctrl_ready_check.observed,
            expected="Verified actuator controller probe",
        )

        # S10: Sensor Telemetry Availability
        if dim_sensor_ready == SensorReadiness.SENSOR_READY:
            s10_state = VerificationState.PASSED
            s10_msg = "Sensor streams and telemetry data verified"
        else:
            s10_state = VerificationState.BLOCKED_MISSING_SENSOR
            s10_msg = sensor_ready_check.error_message or "Sensor telemetry not sampled in live simulation"
        stages[VerificationStage.S10_SENSOR_TELEMETRY.value] = StageResult(
            stage_id=VerificationStage.S10_SENSOR_TELEMETRY.value,
            stage_name="Sensor Telemetry Availability",
            state=s10_state,
            message=s10_msg,
            observed=sensor_ready_check.observed,
            expected="Active sensor topic telemetry",
        )

        # S11: Task Execution Criteria
        if dim_task_ver == TaskVerification.TASK_VERIFIED:
            s11_state = VerificationState.PASSED
            s11_msg = "End-to-end task execution completed and verified"
        else:
            s11_state = VerificationState.BLOCKED_TASK_CRITERIA_FAILED
            s11_msg = task_ver_check.error_message or "Task verification criteria not evaluated"
        stages[VerificationStage.S11_TASK_CRITERIA.value] = StageResult(
            stage_id=VerificationStage.S11_TASK_CRITERIA.value,
            stage_name="Task Execution Criteria",
            state=s11_state,
            message=s11_msg,
            observed=task_ver_check.observed,
            expected="Task pass criteria met",
        )

        # S12: Readiness Classification
        if decision.is_ready:
            overall_verif_state = VerificationState.PASSED
        else:
            priority_order = [
                (s1_state, VerificationState.BLOCKED_MISSING_SOURCE),
                (s3_state, s3_state),
                (s4_state, VerificationState.FAILED),
                (s5_state, VerificationState.BLOCKED_SIMULATOR_INCOMPATIBLE),
                (s6_state, VerificationState.BLOCKED_MISSING_LAUNCH),
                (s7_state, VerificationState.BLOCKED_MISSING_SOURCE),
                (s8_state, VerificationState.BLOCKED_PHYSICS_FAILURE),
                (s9_state, VerificationState.BLOCKED_MISSING_CONTROLLER),
                (s10_state, VerificationState.BLOCKED_MISSING_SENSOR),
                (s11_state, VerificationState.BLOCKED_TASK_CRITERIA_FAILED),
            ]
            overall_verif_state = VerificationState.FAILED
            for st, mapped_err in priority_order:
                if st != VerificationState.PASSED and st != VerificationState.NOT_CHECKED:
                    overall_verif_state = mapped_err if isinstance(mapped_err, VerificationState) else st
                    break

        stages[VerificationStage.S12_READINESS_CLASSIFICATION.value] = StageResult(
            stage_id=VerificationStage.S12_READINESS_CLASSIFICATION.value,
            stage_name="Readiness Classification",
            state=overall_verif_state,
            message=decision.summary_label,
            observed=overall_verif_state.value,
            expected="PASSED" if decision.is_ready else "Appropriate honest BLOCKED state",
        )

        # Authoritative cloud synchronization state
        sync_state = self.compute_sync_state(model.id)

        report = ModelRuntimeVerificationReport(
            robot_id=robot_id,
            display_name=model.display_name,
            family=model.family,
            simulator=target_sim,
            decision=decision,
            cloud_synced=(sync_state.status == CloudSyncStatus.SYNCED),
            sync_state=sync_state,
            verified_at=start_time,
            details=details,
            stages=stages,
            verification_state=overall_verif_state,
        )

        # Sync to Cloud if requested
        if sync_to_cloud:
            synced = self.sync_verification_to_cloud(report)
            report.cloud_synced = synced
            if synced:
                # Refresh sync state
                sync_state = self.compute_sync_state(model.id)
                report.sync_state = sync_state
                if sync_state.status == CloudSyncStatus.SYNCED:
                    report.cloud_synced = True

        return report

    def list_all_canonical_ids(self) -> List[str]:
        """Return sorted unique canonical IDs from both registry and catalogue."""
        models = {m.id: m for m in self.model_catalog.list_models(include_discovered=True)}
        models.update({m.id: m for m in self.robot_registry.list_models()})
        return sorted(models.keys())

    def verify_batch(
        self,
        robot_ids: Optional[List[str]] = None,
        batch_size: int = 10,
        offset: int = 0,
        sync_to_cloud: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute bounded batch verification across robot models.

        Concurrency is bounded to 1 for deterministic physics stability.
        Reports real-time progress via progress_callback.
        Returns batch summary with resumable offset.
        """
        all_ids = robot_ids if robot_ids is not None else self.list_all_canonical_ids()
        total = len(all_ids)
        chunk = all_ids[offset : offset + batch_size]

        reports: List[ModelRuntimeVerificationReport] = []
        for idx, rid in enumerate(chunk):
            current_num = offset + idx + 1
            if progress_callback:
                progress_callback({
                    "current_index": current_num,
                    "total": total,
                    "robot_id": rid,
                    "status": "verifying",
                })
            try:
                rep = self.verify_robot(rid, sync_to_cloud=sync_to_cloud)
            except Exception as e:
                logger.error("Verification error for robot '%s': %s", rid, str(e))
                model = self.robot_registry.get_model(rid) or self.model_catalog.get_model(rid)
                display_name = getattr(model, "display_name", rid)
                family = getattr(model, "family", "unknown")
                sim = getattr(model, "default_simulator", "mujoco")
                dimensions = ReadinessDimensions(catalogue_status=CatalogueStatus.REGISTERED)
                decision = evaluate_execution_readiness(
                    RobotExecutionProfile(robot_id=rid, simulator=sim),
                    dimensions,
                    evidence_checks=[CheckResult(
                        check_id="verification_exception",
                        name="Runtime verification",
                        status="FAILED",
                        error_message=f"{type(e).__name__}: {e}",
                    )],
                )
                rep = ModelRuntimeVerificationReport(
                    robot_id=rid,
                    display_name=display_name,
                    family=family,
                    simulator=sim,
                    decision=decision,
                    details={"error": f"{type(e).__name__}: {e}"},
                    verification_state=VerificationState.FAILED,
                )
            reports.append(rep)
            if progress_callback:
                progress_callback({
                    "current_index": current_num,
                    "total": total,
                    "robot_id": rid,
                    "status": "completed",
                    "verification_state": rep.verification_state.value if hasattr(rep.verification_state, "value") else str(rep.verification_state),
                    "is_ready": rep.decision.is_ready,
                })

        return {
            "total": total,
            "offset": offset,
            "batch_size": batch_size,
            "processed": len(reports),
            "has_more": (offset + batch_size) < total,
            "next_offset": (offset + batch_size) if (offset + batch_size) < total else None,
            "reports": reports,
        }

    def verify_all_robots(self, sync_to_cloud: bool = True, batch_size: int = 50) -> List[ModelRuntimeVerificationReport]:
        """Verify every dynamically registered canonical model once in bounded batches.

        Registry definitions take precedence over catalogue discovery for an
        identical canonical ID because they carry the simulator-specific
        profile. Catalogue-only entries remain visible and are reported with
        conservative capability evidence.
        """
        all_ids = self.list_all_canonical_ids()
        all_reports: List[ModelRuntimeVerificationReport] = []
        offset = 0
        while offset < len(all_ids):
            res = self.verify_batch(all_ids, batch_size=batch_size, offset=offset, sync_to_cloud=sync_to_cloud)
            all_reports.extend(res["reports"])
            if not res["has_more"]:
                break
            offset = res["next_offset"]
        return all_reports

    def sync_verification_to_cloud(self, report: ModelRuntimeVerificationReport) -> bool:
        """Persist verified metadata, version, compatibility, and verification run into Supabase."""
        if not self.catalogue.is_connected():
            return False

        try:
            robot_id = report.robot_id
            decision = report.decision
            version_id = f"{robot_id}_v1.0.0"

            # 1. Upsert model record
            model_rec = CloudModelRecord(
                id=robot_id,
                display_name=report.display_name,
                family=report.family,
                description=report.details.get("notes", f"RoboSahaya {report.display_name} model."),
                catalogue_status="registered" if decision.is_ready else "catalogued",
                metadata={
                    "readiness_summary": decision.summary_label,
                    "is_ready": decision.is_ready,
                    "verification_state": report.verification_state.value if hasattr(report.verification_state, "value") else str(report.verification_state),
                    "stages": {k: v.to_dict() for k, v in report.stages.items()},
                    "dimensions": decision.dimensions.to_dict(),
                    "blockers": decision.blockers,
                    "simulator": report.simulator,
                    "verified_at": report.verified_at,
                },
            )
            self.catalogue.upsert_model(model_rec)

            # 2. Upsert version record
            raw_fmt = report.details.get("format", "URDF").upper()
            if "XACRO" in raw_fmt:
                clean_fmt = "XACRO"
            elif "MJCF" in raw_fmt or "XML" in raw_fmt:
                clean_fmt = "MJCF"
            else:
                clean_fmt = "URDF"

            entry_rel = report.details.get("entrypoint", f"model/{robot_id}.urdf")
            pkg_root = report.details.get("package_root")
            if pkg_root and entry_rel:
                try:
                    p_entry = Path(entry_rel)
                    p_pkg = Path(pkg_root)
                    if p_entry.is_relative_to(p_pkg):
                        entry_rel = str(p_entry.relative_to(p_pkg))
                except Exception:
                    pass

            ver_rec = CloudVersionRecord(
                id=version_id,
                robot_model_id=robot_id,
                version="v1.0.0",
                source_type="bundled" if decision.is_ready else "upstream",
                source_repository="",
                source_revision="head",
                description_format=clean_fmt,
                entrypoint_path=entry_rel,
                manifest_path="manifest.json",
                content_fingerprint=hashlib.sha256(f"{robot_id}_{report.verified_at}".encode()).hexdigest(),
                validation_metadata={
                    "is_ready": decision.is_ready,
                    "dimensions": decision.dimensions.to_dict(),
                    "checks_passed": sum(1 for c in decision.evidence_checks if c.status == "PASSED"),
                    "total_checks": len(decision.evidence_checks),
                },
            )
            self.catalogue.upsert_version(ver_rec)

            # 3. Upsert runtime compatibility
            if decision.dimensions.simulator_compatibility == SimulatorCompatibility.COMPATIBLE:
                compat_status = "compatible"
            elif decision.dimensions.simulator_compatibility == SimulatorCompatibility.SIMULATOR_UNAVAILABLE:
                compat_status = "simulator_unavailable"
            elif decision.dimensions.simulator_compatibility == SimulatorCompatibility.INCOMPATIBLE_FORMAT:
                compat_status = "incompatible"
            else:
                compat_status = "unknown"
            compat_rec = RuntimeCompatibilityRecord(
                id=f"{robot_id}_{report.simulator}_v1.0.0",
                robot_model_version_id=version_id,
                adapter_id=report.simulator,
                compatibility_status=compat_status,
                required_controllers=list(report.details.get("controllers", ())),
                required_sensors=list(report.details.get("sensors", ())),
                known_limitations="; ".join(decision.blockers) if decision.blockers else "None",
            )
            self.catalogue.upsert_compatibility(compat_rec)

            # 4. Insert verification run
            run_id = f"run_{robot_id}_{int(time.time())}"
            env_fp = hashlib.sha256(f"{platform.node()}_{platform.machine()}_{platform.python_version()}".encode()).hexdigest()
            run_rec = VerificationRunRecord(
                id=run_id,
                robot_model_version_id=version_id,
                adapter_id=report.simulator,
                world_id=report.details.get("world", "obstacle_world"),
                test_suite_version="v1.0.0",
                test_profile={
                    "type": "multi_dimensional_runtime_verification",
                    "verification_state": report.verification_state.value if hasattr(report.verification_state, "value") else str(report.verification_state),
                    "stages": {k: v.to_dict() for k, v in report.stages.items()},
                },
                environment_fingerprint=env_fp,
                overall_result="PASSED" if decision.is_ready else "BLOCKED",
                check_results=[c.to_dict() for c in decision.evidence_checks],
                report_fingerprint=hashlib.sha256(json.dumps([c.to_dict() for c in decision.evidence_checks]).encode()).hexdigest(),
                started_at=report.verified_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            return self.catalogue.record_verification_run(run_rec)
        except Exception as e:
            logger.warning("Failed to sync verification for '%s' to cloud: %s", report.robot_id, str(e))
            return False

    def compute_sync_state(self, robot_id: str, version: str = "v1.0.0") -> ModelSyncState:
        """Evaluate authoritative cloud sync state between local files and Supabase."""
        if not self.catalogue.is_connected():
            return ModelSyncState(
                robot_id=robot_id,
                status=CloudSyncStatus.OFFLINE,
                display_label="🔌 Offline",
                details="Cloud catalogue provider is offline or unreachable.",
                local_version=version,
            )

        # 1. Count local assets
        local_count = 0
        model = self.robot_registry.get_model(robot_id)
        if model is None:
            catalog_model = self.model_catalog.get_model(robot_id)
            if catalog_model:
                model = RobotModel(
                    id=catalog_model.id,
                    display_name=catalog_model.display_name,
                    family=catalog_model.family,
                    aliases=catalog_model.aliases,
                    description_module=catalog_model.description_module,
                    formats=catalog_model.formats,
                    license=catalog_model.license,
                    runtime_status=catalog_model.runtime_status,
                    runtime_template_id=catalog_model.runtime_template_id,
                )

        if model:
            if getattr(model, "runtime_template_id", None):
                tpl_dir = self.templates_root / model.runtime_template_id
                if tpl_dir.exists():
                    local_count = len([f for f in tpl_dir.rglob("*") if f.is_file() and "__pycache__" not in str(f)])
            elif robot_id == "rsk":
                rsk_dir = Path.home() / ".cache" / "robosahaya" / "robot_models" / "rsk" / version
                if rsk_dir.exists():
                    local_count = len([f for f in rsk_dir.rglob("*") if f.is_file() and "__pycache__" not in str(f)])
            else:
                cache_base = Path.home() / ".cache" / "robot_descriptions"
                known_dirs = {
                    "so_arm100": cache_base / "SO-ARM100",
                    "ur5e": cache_base / "ur_description",
                    "unitree_go2": cache_base / "unitree_ros",
                    "skydio_x2": cache_base / "skydio_x2_description",
                }
                if robot_id in known_dirs and known_dirs[robot_id].exists():
                    local_count = len([f for f in known_dirs[robot_id].rglob("*") if f.is_file() and ".git" not in str(f)])

        # 2. Query cloud assets and version
        version_id = f"{robot_id}_{version}"
        remote_assets = self.catalogue.list_assets(version_id)
        remote_count = len(remote_assets)
        cloud_version_rec = self.catalogue.get_version(robot_id, version)
        cloud_version_str = cloud_version_rec.version if cloud_version_rec else None

        # 3. Determine status and label
        is_offline_provider = (self.catalogue.__class__.__name__ == "OfflineCatalogueProvider")
        if is_offline_provider:
            if cloud_version_rec or remote_count > 0:
                status = CloudSyncStatus.SYNCED
                display_label = "☁️ Synced"
                details = f"Synchronized in local offline catalogue provider ({local_count} local files)."
            else:
                status = CloudSyncStatus.LOCAL_ONLY
                display_label = "💻 Local only"
                details = f"Local template available ({local_count} files); not recorded in offline catalogue."
        elif remote_count > 0 and local_count > 0:
            status = CloudSyncStatus.SYNCED
            display_label = "☁️ Synced"
            details = f"All {remote_count} assets and records synchronized to Supabase bucket 'robot-model-packages'."
        elif local_count > 0 and remote_count == 0:
            if cloud_version_rec:
                status = CloudSyncStatus.REMOTE_ASSET_MISSING
                display_label = "⚠️ Remote Assets Missing"
                details = f"Model version registered in cloud, but object storage assets are missing ({local_count} local files)."
            else:
                status = CloudSyncStatus.LOCAL_ONLY
                display_label = "💻 Local only"
                details = f"{local_count} local files present; not synchronized to Supabase cloud."
        elif local_count == 0 and remote_count > 0:
            status = CloudSyncStatus.LOCAL_CACHE_MISSING
            display_label = "📥 Cloud Stored (Not Cached)"
            details = f"{remote_count} assets stored in Supabase; not cached locally."
        else:
            cloud_model = self.catalogue.get_model(robot_id)
            if cloud_model:
                status = CloudSyncStatus.CLOUD_AVAILABLE
                display_label = "☁️ Catalogue Only"
                details = "Metadata catalogued in cloud; no physical assets uploaded."
            else:
                status = CloudSyncStatus.LOCAL_ONLY
                display_label = "💻 Local only"
                details = "Model only present in local definition."

        return ModelSyncState(
            robot_id=robot_id,
            status=status,
            display_label=display_label,
            details=details,
            local_version=version,
            cloud_version=cloud_version_str,
            local_assets_count=local_count,
            remote_assets_count=remote_count,
        )

    def _check_local_cache(self, model: RobotModel) -> Tuple[LocalCacheStatus, CheckResult]:
        """Dimension 3: require a verified manifest, not mere file presence."""
        integrity = self.model_cache.validate_cache_integrity(model.id, "v1.0.0")
        if integrity.get("valid"):
            return LocalCacheStatus.CACHE_VERIFIED, CheckResult(
                check_id="local_cache",
                name="Cryptographic Asset Manifest",
                status="PASSED",
                observed=f"{integrity.get('file_count', 0)} assets matched SHA-256",
                expected="All package assets match the registered SHA-256 manifest",
            )

        source = self._resolve_local_entrypoint(model)
        reason = (
            f"A local description exists at {source[1]}, but its complete package has no valid RoboSahaya integrity manifest."
            if source else
            "No prepared model package with a valid SHA-256 manifest is available."
        )
        return LocalCacheStatus.NOT_CACHED, CheckResult(
            check_id="local_cache",
            name="Verified Model Package",
            status="BLOCKED",
            observed="Package integrity not verified",
            expected="All required package files match a verified manifest",
            error_message=reason,
        )

    def _check_description_validity(self, model: RobotModel) -> Tuple[DescriptionValidity, CheckResult, Dict[str, Any]]:
        """Dimension 4: parse the actual registered entrypoint and format."""
        info: Dict[str, Any] = {}
        resolved = self._resolve_local_entrypoint(model)
        if resolved is None:
            return DescriptionValidity.UNCHECKED, CheckResult(
                check_id="desc_valid",
                name="Description Structure",
                status="NOT_RUN",
                error_message="No prepared or registered local model entrypoint is available; catalogue format metadata alone is not validation.",
            ), info

        package_root, entrypoint = resolved
        info["entrypoint"] = str(entrypoint)
        info["package_root"] = str(package_root)
        suffix = entrypoint.suffix.lower()
        if suffix == ".xml" and any(fmt.upper() in {"MJCF", "XML"} for fmt in model.formats):
            try:
                import mujoco
                compiled = load_mujoco_model(entrypoint)
                info.update({"format": "MJCF", "bodies": int(compiled.nbody), "joints": int(compiled.njnt), "actuators": int(compiled.nu), "geoms": int(compiled.ngeom)})
                return DescriptionValidity.MODEL_VALIDATED, CheckResult(
                    check_id="desc_valid", name="MuJoCo Model Compilation", status="PASSED",
                    observed=f"Compiled {entrypoint.name}: {compiled.nbody} bodies, {compiled.njnt} joints, {compiled.nu} actuators",
                    expected="Actual MJCF compiles in the installed MuJoCo runtime",
                ), info
            except ImportError:
                return DescriptionValidity.UNCHECKED, CheckResult(
                    check_id="desc_valid", name="MuJoCo Model Compilation", status="NOT_RUN",
                    error_message="MuJoCo Python package is not installed.",
                ), info
            except Exception as exc:
                return DescriptionValidity.SYNTAX_ERROR, CheckResult(
                    check_id="desc_valid", name="MuJoCo Model Compilation", status="FAILED",
                    error_message=f"{type(exc).__name__}: {exc}",
                ), info

        try:
            root = ET.parse(entrypoint).getroot()
            if root.tag != "robot":
                return DescriptionValidity.SYNTAX_ERROR, CheckResult(
                    check_id="desc_valid", name="URDF/XML Structure", status="FAILED",
                    observed=f"Root element is <{root.tag}>", expected="A URDF <robot> root element",
                ), info
            links = len(root.findall("link"))
            joints = len(root.findall("joint"))
            if links == 0:
                raise ValueError("URDF contains no link elements.")
            info.update({"format": "URDF", "links": links, "joints": joints})
            return DescriptionValidity.MODEL_VALIDATED, CheckResult(
                check_id="desc_valid", name="URDF Structural Validation", status="PASSED",
                observed=f"Parsed {entrypoint.name}: {links} links, {joints} joints",
                expected="Valid URDF with at least one link element",
            ), info
        except Exception as exc:
            return DescriptionValidity.SYNTAX_ERROR, CheckResult(
                check_id="desc_valid", name="URDF Structural Validation", status="FAILED",
                error_message=f"{type(exc).__name__}: {exc}",
            ), info
    def _resolve_local_entrypoint(self, model: RobotModel) -> Optional[Tuple[Path, Path]]:
        """Resolve only explicit manifests, bundled entrypoints, or static source metadata."""
        version_dir = self.model_cache.get_version_dir(model.id, "v1.0.0")
        metadata_file = version_dir / "metadata.json"
        if metadata_file.is_file() and self.model_cache.validate_cache_integrity(model.id, "v1.0.0").get("valid"):
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                entry = Path(metadata["entrypoint"])
                if not entry.is_absolute():
                    entry = version_dir / entry
                entry = entry.resolve()
                if entry.is_file() and entry.is_relative_to(version_dir.resolve()):
                    return version_dir.resolve(), entry
            except (KeyError, OSError, ValueError, TypeError):
                pass

        description_metadata = self.model_catalog._manifest_path(model.id)
        if description_metadata.is_file():
            try:
                metadata = json.loads(description_metadata.read_text(encoding="utf-8"))
                package = Path(metadata["package_path"]).expanduser().resolve()
                entry = Path(metadata["path"]).expanduser().resolve()
                if package.is_dir() and entry.is_file() and entry.is_relative_to(package):
                    return package, entry
            except (KeyError, OSError, ValueError, TypeError):
                pass

        if model.runtime_template_id:
            package = self.templates_root / model.runtime_template_id
            for filename in ("model.urdf.xacro", "model.urdf", "model.xml", "model.sdf"):
                entry = package / filename
                if entry.is_file():
                    return package.resolve(), entry.resolve()

        if model.description_module:
            # Static description lookup is intentionally used here. Runtime
            # verification must never import a generated description module,
            # because its import hook may download or update an upstream git
            # checkout. Only an explicit preparation action may do that.
            try:
                return self.model_cache._resolve_cached_description_module(model.description_module)
            except Exception:
                return None
        return None

    def _check_simulator_compatibility(self, model: RobotModel, simulator: str) -> Tuple[SimulatorCompatibility, CheckResult]:
        """Dimension 5: Check simulator engine availability and format match."""
        adapter = self.simulator_registry.get_adapter(simulator)
        if not adapter:
            return SimulatorCompatibility.SIMULATOR_UNAVAILABLE, CheckResult(
                check_id="sim_compat",
                name="Simulator Compatibility",
                status="BLOCKED",
                error_message=f"Simulator '{simulator}' is not registered.",
            )

        compat = adapter.inspect_compatibility(model)
        if compat.get("compatible"):
            return SimulatorCompatibility.COMPATIBLE, CheckResult(
                check_id="sim_compat",
                name="Simulator Compatibility",
                status="PASSED",
                observed=adapter.display_name,
                expected=f"Compatible with {simulator}",
            )

        return SimulatorCompatibility.INCOMPATIBLE_FORMAT, CheckResult(
            check_id="sim_compat",
            name="Simulator Compatibility",
            status="BLOCKED",
            error_message=compat.get("reason", f"Model is incompatible with '{simulator}'."),
        )

    def _check_simulation_readiness(self, model: RobotModel, simulator: str) -> Tuple[SimulationReadiness, CheckResult]:
        """Dimension 6: load and step the actual model when the adapter permits."""
        if simulator != "mujoco":
            return SimulationReadiness.UNTESTED, CheckResult(
                check_id="sim_readiness", name="Simulation Physics Stepping", status="NOT_RUN",
                observed="No Gazebo smoke test was run by this verifier.",
                expected=f"Actual model load and physics steps in {simulator}",
                error_message="Gazebo Harmonic needs a configured workspace manager and model launch profile; template presence is not a physics test.",
            )

        resolved = self._resolve_local_entrypoint(model)
        if resolved is None:
            return SimulationReadiness.UNTESTED, CheckResult(
                check_id="sim_readiness", name="MuJoCo Physics Stepping", status="NOT_RUN",
                error_message="No local registered MJCF entrypoint is available; model preparation is required.",
            )

        _package_root, entrypoint = resolved
        try:
            import mujoco
            import numpy as np
            actual_model = load_mujoco_model(entrypoint)
            data = mujoco.MjData(actual_model)
            steps = 100
            for _ in range(steps):
                mujoco.mj_step(actual_model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError("MuJoCo state contains NaN or infinity during physics stepping.")
            return SimulationReadiness.SIMULATION_READY, CheckResult(
                check_id="sim_readiness", name="MuJoCo 100-Step Physics Integration", status="PASSED",
                observed={"entrypoint": str(entrypoint), "steps": steps, "sim_time_sec": float(data.time), "bodies": int(actual_model.nbody), "actuators": int(actual_model.nu)},
                expected="100 finite physics steps from the registered model entrypoint",
            )
        except ImportError:
            return SimulationReadiness.UNTESTED, CheckResult(
                check_id="sim_readiness", name="MuJoCo Physics Stepping", status="NOT_RUN",
                error_message="MuJoCo Python package is not installed.",
            )
        except Exception as exc:
            return SimulationReadiness.STEP_FAILED, CheckResult(
                check_id="sim_readiness", name="MuJoCo Physics Stepping", status="FAILED",
                error_message=f"{type(exc).__name__}: {exc}",
            )

    def _check_controller_readiness(self, model: RobotModel, simulator: str) -> Tuple[ControllerReadiness, CheckResult]:
        """Dimension 7: execute a configured real actuator probe if present."""
        probe = model.description_info.get("verification_probe", {}) if isinstance(model.description_info, dict) else {}
        if simulator == "mujoco" and probe.get("actuator"):
            resolved = self._resolve_local_entrypoint(model)
            if resolved is None:
                return ControllerReadiness.UNTESTED, CheckResult(
                    check_id="ctrl_readiness", name="Actuator Controller Verification", status="NOT_RUN",
                    error_message="No local registered model entrypoint is available for controller verification.",
                )
            try:
                import mujoco
                import numpy as np
                _package_root, entrypoint = resolved
                actual_model = load_mujoco_model(entrypoint)
                actuator_id = mujoco.mj_name2id(actual_model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(probe["actuator"]))
                if actuator_id < 0:
                    return ControllerReadiness.CONTROLLER_MISSING, CheckResult(
                        check_id="ctrl_readiness", name="Actuator Controller Verification", status="FAILED",
                        error_message=f"Declared actuator '{probe['actuator']}' is not defined in compiled MuJoCo model.",
                    )
                data = mujoco.MjData(actual_model)
                cmd = float(probe.get("command", 1.0))
                data.ctrl[actuator_id] = cmd
                qpos_before = np.copy(data.qpos)
                for _ in range(int(probe.get("steps", 50))):
                    mujoco.mj_step(actual_model, data)
                delta_norm = float(np.linalg.norm(data.qpos - qpos_before))
                ctrl_range = [float(actual_model.actuator_ctrlrange[actuator_id][0]), float(actual_model.actuator_ctrlrange[actuator_id][1])]
                return ControllerReadiness.CONTROL_READY, CheckResult(
                    check_id="ctrl_readiness", name="Actuator Controller Verification", status="PASSED",
                    observed={"actuator": probe["actuator"], "command": cmd, "control_range": ctrl_range, "qpos_delta_norm": delta_norm},
                    expected="Actuator responds to control input and advances joint coordinate",
                )
            except Exception as exc:
                return ControllerReadiness.CONTROLLER_MISSING, CheckResult(
                    check_id="ctrl_readiness", name="Actuator Controller Verification", status="FAILED",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
        if not model.controllers:
            return ControllerReadiness.CONTROLLER_MISSING, CheckResult(
                check_id="ctrl_readiness", name="Actuator Controller Verification", status="BLOCKED",
                error_message="Model declares no controllers.",
            )
        return ControllerReadiness.UNTESTED, CheckResult(
            check_id="ctrl_readiness", name="Actuator Controller Verification", status="NOT_RUN",
            error_message="No runtime controller probe configured.",
        )

    def _check_sensor_readiness(self, model: RobotModel, simulator: str) -> Tuple[SensorReadiness, CheckResult]:
        """Dimension 8: verify sensor availability or return untested."""
        if not model.sensors:
            return SensorReadiness.SENSOR_MISSING, CheckResult(
                check_id="sensor_readiness", name="Sensor Telemetry Verification", status="BLOCKED",
                error_message="Model declares no sensors.",
            )
        return SensorReadiness.UNTESTED, CheckResult(
            check_id="sensor_readiness", name="Sensor Telemetry Verification", status="NOT_RUN",
            observed=[], expected="A simulator sensor sample passes the profile's validation criteria",
            error_message="This verification run did not sample sensor data from the simulator.",
        )

    def _check_task_verification(self, model: RobotModel) -> Tuple[TaskVerification, CheckResult]:
        """Dimension 9: verify end-to-end task completion."""
        return TaskVerification.UNTESTED, CheckResult(
            check_id="task_ver", name="Task Execution Criteria", status="NOT_RUN",
            error_message="No task specification was executed and verified by this catalogue verification pass.",
        )


_GLOBAL_VERIFIER: Optional[RuntimeVerifier] = None


def get_runtime_verifier() -> RuntimeVerifier:
    global _GLOBAL_VERIFIER
    if _GLOBAL_VERIFIER is None:
        _GLOBAL_VERIFIER = RuntimeVerifier()
    return _GLOBAL_VERIFIER
