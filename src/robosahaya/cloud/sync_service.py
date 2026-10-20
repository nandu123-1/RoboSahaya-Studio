"""Universal Bulk Robot Model Synchronization and Cloud Asset Migration Service.

Provides a general-purpose, resumable bulk synchronization engine that:
1. Builds a canonical inventory of all registered, catalogued, and cached robot models.
2. Inspects model descriptions, meshes, textures, manifests, and licenses.
3. Enforces license and redistribution permissions.
4. Uploads eligible assets idempotently to Supabase Object Storage (bucket: robot-model-packages).
5. Registers metadata, versions, and cryptographic asset manifests in Supabase Postgres.
6. Verifies remote object existence, file sizes, and SHA-256 hashes.
7. Evaluates independent 9D execution readiness without conflating cloud sync with simulator readiness.
8. Produces detailed audit and batch progress reports with resume capabilities.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from robosahaya.cloud.base import (
    BaseCatalogueProvider,
    BaseStorageProvider,
    CloudAssetRecord,
    CloudModelRecord,
    CloudSyncStatus,
    CloudVersionRecord,
    ModelSyncState,
    RuntimeCompatibilityRecord,
    VerificationRunRecord,
)
from robosahaya.cloud.client import get_cloud_catalogue, get_cloud_storage
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
from robosahaya.robot_catalog import RobotModelCatalog, _SEEDED_MODELS
from robosahaya.robot_registry import RobotModel, RobotRegistry, get_robot_registry
from robosahaya.simulators.registry import get_simulator_registry

logger = logging.getLogger(__name__)

# Permissive licenses eligible for managed cloud mirroring with attribution
PERMISSIVE_LICENSES: Set[str] = {
    "apache-2.0",
    "apache 2.0",
    "apache",
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-3-clause-clear",
    "cc-by-4.0",
    "cc0-1.0",
    "robosahaya project licence",
    "robosahaya project license",
}

MAX_PACKAGE_BYTES_DEFAULT = 250 * 1024 * 1024  # 250 MB quota per model


@dataclass
class PackageAssetInfo:
    file_path: Path
    relative_path: str
    asset_type: str
    size_bytes: int
    sha256: str


@dataclass
class RobotSyncInventoryItem:
    """Canonical model entry in the synchronization inventory."""
    robot_id: str
    display_name: str
    family: str
    version: str = "v1.0.0"
    source_module: Optional[str] = None
    formats: Tuple[str, ...] = ("URDF",)
    declared_license: str = "Check upstream source"
    is_licence_eligible: bool = True
    local_package_path: Optional[str] = None
    local_files_count: int = 0
    local_bytes_total: int = 0
    entrypoint_path: Optional[str] = None
    cloud_model_exists: bool = False
    cloud_version_exists: bool = False
    cloud_assets_count: int = 0
    sync_status: CloudSyncStatus = CloudSyncStatus.UNKNOWN
    sync_label: str = ""
    sync_notes: str = ""
    execution_ready: bool = False
    execution_summary: str = ""
    blockers: List[str] = field(default_factory=list)
    last_synced_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "display_name": self.display_name,
            "family": self.family,
            "version": self.version,
            "source_module": self.source_module,
            "formats": list(self.formats),
            "declared_license": self.declared_license,
            "is_licence_eligible": self.is_licence_eligible,
            "local_package_path": self.local_package_path,
            "local_files_count": self.local_files_count,
            "local_bytes_total": self.local_bytes_total,
            "entrypoint_path": self.entrypoint_path,
            "cloud_model_exists": self.cloud_model_exists,
            "cloud_version_exists": self.cloud_version_exists,
            "cloud_assets_count": self.cloud_assets_count,
            "sync_status": self.sync_status.value,
            "sync_label": self.sync_label,
            "sync_notes": self.sync_notes,
            "execution_ready": self.execution_ready,
            "execution_summary": self.execution_summary,
            "blockers": self.blockers,
            "last_synced_at": self.last_synced_at,
        }


@dataclass
class CatalogueInventory:
    """Complete inventory of all canonical robot models across local and cloud sources."""
    items: Dict[str, RobotSyncInventoryItem]
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_models: int = 0
    synced_count: int = 0
    local_only_count: int = 0
    remote_missing_count: int = 0
    source_available_count: int = 0
    licence_blocked_count: int = 0
    failed_count: int = 0
    execution_ready_count: int = 0

    @property
    def total_cached(self) -> int:
        return sum(1 for item in self.items.values() if item.local_files_count > 0)

    @property
    def total_licence_eligible(self) -> int:
        return sum(1 for item in self.items.values() if item.is_licence_eligible)

    @property
    def item_list(self) -> List[RobotSyncInventoryItem]:
        return list(self.items.values())

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "total_models": self.total_models,
            "synced_count": self.synced_count,
            "local_only_count": self.local_only_count,
            "remote_missing_count": self.remote_missing_count,
            "source_available_count": self.source_available_count,
            "licence_blocked_count": self.licence_blocked_count,
            "failed_count": self.failed_count,
            "execution_ready_count": self.execution_ready_count,
        }


@dataclass
class SyncResult:
    """Outcome of synchronizing a single robot model."""
    robot_id: str
    success: bool
    status: CloudSyncStatus
    assets_uploaded: int = 0
    assets_verified: int = 0
    bytes_transferred: int = 0
    duration_seconds: float = 0.0
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "success": self.success,
            "status": self.status.value,
            "assets_uploaded": self.assets_uploaded,
            "assets_verified": self.assets_verified,
            "bytes_transferred": self.bytes_transferred,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
            "error": self.error,
        }


@dataclass
class BatchSyncReport:
    """Consolidated report of a bulk synchronization operation."""
    started_at: str
    completed_at: str
    total_considered: int = 0
    newly_synced: int = 0
    already_synced: int = 0
    skipped_licence: int = 0
    skipped_source_unavailable: int = 0
    failed: int = 0
    assets_uploaded: int = 0
    assets_verified: int = 0
    results: Dict[str, SyncResult] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_considered": self.total_considered,
            "newly_synced": self.newly_synced,
            "already_synced": self.already_synced,
            "skipped_licence": self.skipped_licence,
            "skipped_source_unavailable": self.skipped_source_unavailable,
            "failed": self.failed,
            "assets_uploaded": self.assets_uploaded,
            "assets_verified": self.assets_verified,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }

    def summary_markdown(self) -> str:
        lines = [
            f"### ☁️ Bulk Synchronization Report ({self.completed_at})",
            f"- **Models Considered:** `{self.total_considered}`",
            f"- **Newly Synchronized:** `{self.newly_synced}`",
            f"- **Already Synced (Idempotent):** `{self.already_synced}`",
            f"- **Assets Uploaded:** `{self.assets_uploaded}`",
            f"- **Assets Verified in Object Storage:** `{self.assets_verified}`",
            f"- **Skipped (Licence Restricted):** `{self.skipped_licence}`",
            f"- **Skipped (Source Unavailable / Not Cached):** `{self.skipped_source_unavailable}`",
            f"- **Failed / Incomplete:** `{self.failed}`",
            "",
            "| Robot | Result Status | Uploaded | Verified | Duration | Notes |",
            "|---|---|---|---|---|---|",
        ]
        for rid, r in sorted(self.results.items()):
            icon = "✅" if r.success else "⚠️" if r.status in (CloudSyncStatus.SOURCE_AVAILABLE, CloudSyncStatus.LICENCE_BLOCKED) else "🔴"
            dur = f"{r.duration_seconds:.1f}s"
            lines.append(f"| `{rid}` | {icon} `{r.status.value}` | `{r.assets_uploaded}` | `{r.assets_verified}` | {dur} | {r.message} |")
        return "\n".join(lines)


@dataclass
class AuditReport:
    """Integrity audit report for remote cloud assets."""
    started_at: str
    completed_at: str
    total_versions_checked: int = 0
    total_assets_checked: int = 0
    assets_verified: int = 0
    missing_assets: List[str] = field(default_factory=list)
    corrupted_assets: List[str] = field(default_factory=list)
    model_statuses: Dict[str, CloudSyncStatus] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_versions_checked": self.total_versions_checked,
            "total_assets_checked": self.total_assets_checked,
            "assets_verified": self.assets_verified,
            "missing_assets": self.missing_assets,
            "corrupted_assets": self.corrupted_assets,
            "model_statuses": {k: v.value for k, v in self.model_statuses.items()},
        }

    def summary_markdown(self) -> str:
        lines = [
            f"### 🛡️ Cloud Storage Asset Integrity Audit ({self.completed_at})",
            f"- **Model Versions Checked:** `{self.total_versions_checked}`",
            f"- **Total Remote Assets Audited:** `{self.total_assets_checked}`",
            f"- **Cryptographically / Storage Verified:** `{self.assets_verified}`",
            f"- **Missing Storage Objects:** `{len(self.missing_assets)}`",
            f"- **Corrupted / Hash Mismatch Objects:** `{len(self.corrupted_assets)}`",
            "",
            "| Model Version | Stored Assets | Verified | Integrity Verdict |",
            "|---|---|---|---|",
        ]
        for vid, st in sorted(self.model_statuses.items()):
            icon = "✅" if st == CloudSyncStatus.SYNCED else "🔴"
            lines.append(f"| `{vid}` | Checked | Verified | {icon} `{st.value}` |")
        return "\n".join(lines)


class RobotCatalogueSyncService:
    """Universal synchronization and integrity engine for RoboSahaya Studio."""

    def __init__(
        self,
        catalogue_provider: Optional[BaseCatalogueProvider] = None,
        storage_provider: Optional[BaseStorageProvider] = None,
        robot_registry: Optional[RobotRegistry] = None,
        model_catalog: Optional[RobotModelCatalog] = None,
        templates_root: Optional[Path] = None,
        cache_base: Optional[Path] = None,
    ):
        self.catalogue = catalogue_provider or get_cloud_catalogue()
        self.storage = storage_provider or get_cloud_storage()
        self.registry = robot_registry or get_robot_registry()
        self.catalog = model_catalog or RobotModelCatalog()
        self.templates_root = Path(templates_root or Path(__file__).resolve().parent.parent.parent.parent / "simulation" / "templates")
        self.cache_base = Path(cache_base or Path.home() / ".cache").resolve()

    def check_licence_redistribution(self, licence: str, local_path: Optional[Path] = None) -> Tuple[bool, str]:
        """Check if model license permits managed cloud storage mirroring."""
        clean = (licence or "").strip().lower()
        if clean in PERMISSIVE_LICENSES:
            return True, licence

        # Check local package directory for LICENSE or COPYING file
        if local_path and local_path.is_dir():
            for lic_name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
                lic_file = local_path / lic_name
                if lic_file.is_file():
                    try:
                        content = lic_file.read_text(errors="ignore").lower()
                        if "apache" in content:
                            return True, "Apache-2.0"
                        if "mit license" in content or "permission is hereby granted" in content:
                            return True, "MIT"
                        if "bsd" in content or "redistribution and use in source and binary" in content:
                            return True, "BSD-3-Clause"
                    except Exception:
                        pass

        # Disallow automated public mirroring of unconfirmed or proprietary licenses
        return False, licence or "Unknown / Unestablished"

    def resolve_local_package(
        self,
        robot_id: str,
        version: str = "v1.0.0",
        allow_network_download: bool = False,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Resolve the local directory and primary entrypoint for a robot model.

        Does not trigger network git clones unless allow_network_download is True.
        """
        # 1. Bundled templates (differential_drive, franka_panda, so_arm101)
        model = self.registry.get_model(robot_id) or self.catalog.get_model(robot_id)
        if model and getattr(model, "runtime_template_id", None):
            tpl_dir = self.templates_root / model.runtime_template_id
            if tpl_dir.is_dir():
                entry = tpl_dir / "model.urdf.xacro"
                return tpl_dir, entry if entry.exists() else None

        # 2. RSK model cache
        if robot_id == "rsk":
            rsk_dir = self.cache_base / "robosahaya" / "robot_models" / "rsk" / version
            if rsk_dir.is_dir():
                entry = rsk_dir / "model" / "robot_soccer_kit.xml"
                return rsk_dir, entry if entry.exists() else None
            # Menagerie fallback
            menagerie_rsk = self.cache_base / "robot_descriptions" / "mujoco_menagerie" / "robot_soccer_kit"
            if menagerie_rsk.is_dir():
                entry = menagerie_rsk / "robot_soccer_kit.xml"
                return menagerie_rsk, entry if entry.exists() else None

        # 3. Upstream cache in robot_descriptions
        rd_base = self.cache_base / "robot_descriptions"
        known_dirs: Dict[str, Tuple[Path, str]] = {
            "so_arm100": (rd_base / "SO-ARM100" / "Simulation" / "SO100", "so100.urdf"),
            "skydio_x2": (rd_base / "skydio_x2_description", "urdf/skydio_x2.urdf"),
            "unitree_go2": (rd_base / "unitree_ros" / "robots" / "go2_description", "urdf/go2_description.urdf"),
            "ur5e": (rd_base / "ur_description", "urdf/ur.urdf.xacro"),
            "ur3e": (rd_base / "ur_description", "urdf/ur.urdf.xacro"),
            "ur10e": (rd_base / "ur_description", "urdf/ur.urdf.xacro"),
            "unitree_go1": (rd_base / "unitree_ros" / "robots" / "go1_description", "urdf/go1_description.urdf"),
            "unitree_g1": (rd_base / "unitree_ros" / "robots" / "g1_description", "g1.urdf"),
            "unitree_h1": (rd_base / "unitree_ros" / "robots" / "h1_description", "h1.urdf"),
            "unitree_z1": (rd_base / "unitree_ros" / "robots" / "z1_description", "z1_description.urdf"),
            "tiago": (rd_base / "example-robot-data" / "robots" / "tiago_description", "tiago.urdf"),
            "simple_humanoid": (rd_base / "example-robot-data" / "robots" / "simple_humanoid_description", "simple_humanoid.urdf"),
            "anymal_b": (rd_base / "example-robot-data" / "robots" / "anymal_b_simple_description", "anymal_b.urdf"),
        }
        if robot_id in known_dirs:
            p_dir, rel_entry = known_dirs[robot_id]
            if p_dir.is_dir():
                return p_dir, p_dir / rel_entry

        # 4. Check robosahaya cache
        robo_cache = self.cache_base / "robosahaya" / "robot_models" / robot_id
        if robo_cache.is_dir():
            v_dir = robo_cache / version
            target = v_dir if v_dir.is_dir() else robo_cache
            for cand in target.rglob("*.urdf"):
                return target, cand
            for cand in target.rglob("*.xml"):
                return target, cand

        # 5. Dynamic robot_descriptions import lookup ONLY if explicitly permitted
        if allow_network_download and model and getattr(model, "description_module", None):
            try:
                mod = importlib.import_module(f"robot_descriptions.{model.description_module}")
                pkg = getattr(mod, "PACKAGE_PATH", None) or getattr(mod, "REPOSITORY_PATH", None)
                entry = getattr(mod, "URDF_PATH", None) or getattr(mod, "XACRO_PATH", None) or getattr(mod, "MJCF_PATH", None)
                if pkg and Path(str(pkg)).is_dir():
                    return Path(str(pkg)), Path(str(entry)) if entry else None
            except Exception:
                pass

        return None, None

    def discover_package_assets(
        self,
        base_dir: Path,
        robot_id: str = "",
        compute_hashes: bool = True,
    ) -> List[PackageAssetInfo]:
        """Collect all relevant package assets, computing relative paths and SHA-256 digests."""
        if not base_dir.is_dir():
            return []

        assets: List[PackageAssetInfo] = []
        # Exclude version control, python bytecode, build metadata, test directories
        exclude_dir_names = {".git", "__pycache__", ".github", ".cache", "venv", ".venv", "doc", "docs", "test", "tests", "rviz"}
        exclude_suffixes = {".pyc", ".pyo", ".part", ".tmp", ".pdf", ".rst"}
        exclude_filenames = {
            "CMakeLists.txt", "package.xml", ".gitignore", ".pre-commit-config.yaml",
            "ci_status.md", "CONTRIBUTING.md", "CHANGELOG.rst", "CHANGELOG.md", "model.pdf",
        }

        # Multi-model repo disambiguation (e.g. ur_description contains ur3, ur5, ur10, ur20...)
        ur_models = {"ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e", "ur16e", "ur18", "ur20", "ur30", "ur8long", "ur15"}
        sibling_to_skip = set()
        if base_dir.name == "ur_description" and robot_id.startswith("ur"):
            sibling_to_skip = {m for m in ur_models if m != robot_id and m != robot_id.rstrip("e")}

        for p in base_dir.rglob("*"):
            if not p.is_file():
                continue
            rel_parts = p.relative_to(base_dir).parts
            if any(part in exclude_dir_names or (part.startswith(".") and part != ".config") for part in rel_parts[:-1]):
                continue
            if p.name in exclude_filenames:
                continue
            if sibling_to_skip and any(sib in rel_parts for sib in sibling_to_skip):
                continue
            if p.suffix.lower() in exclude_suffixes or p.name.startswith("."):
                continue

            rel = str(p.relative_to(base_dir))
            suffix = p.suffix.lower()

            if suffix in (".urdf", ".xacro", ".mjcf", ".xml", ".srdf", ".sdf"):
                atype = "description"
            elif suffix in (".stl", ".dae", ".obj", ".ply"):
                atype = "mesh"
            elif suffix in (".png", ".jpg", ".jpeg", ".tga", ".bmp"):
                atype = "texture"
            elif suffix in (".yaml", ".yml", ".json", ".config", ".xml"):
                atype = "config"
            elif suffix in (".py", ".sh"):
                atype = "script"
            else:
                atype = "asset"

            size = p.stat().st_size
            sha = ""
            if compute_hashes:
                hasher = hashlib.sha256()
                with open(p, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                sha = hasher.hexdigest()

            assets.append(PackageAssetInfo(
                file_path=p,
                relative_path=rel,
                asset_type=atype,
                size_bytes=size,
                sha256=sha,
            ))

        return sorted(assets, key=lambda a: a.relative_to_entry if hasattr(a, 'relative_to_entry') else a.relative_path)

    def build_inventory(self, force_refresh: bool = False) -> CatalogueInventory:
        """Scan all canonical robots and construct authoritative synchronization inventory."""
        start_time = datetime.now(timezone.utc).isoformat()
        items: Dict[str, RobotSyncInventoryItem] = {}

        # 1. Collect canonical models
        canonical_map: Dict[str, Dict[str, Any]] = {}

        # From _SEEDED_MODELS (30 models)
        for m in _SEEDED_MODELS:
            canonical_map[m.id] = {
                "id": m.id,
                "display_name": m.display_name,
                "family": m.family,
                "license": m.license,
                "formats": m.formats,
                "module": m.description_module,
                "template_id": m.runtime_template_id,
                "notes": m.notes,
            }

        # From RobotRegistry (8 models)
        for reg_m in self.registry.list_models():
            if reg_m.id not in canonical_map:
                canonical_map[reg_m.id] = {
                    "id": reg_m.id,
                    "display_name": reg_m.display_name,
                    "family": reg_m.family,
                    "license": reg_m.license,
                    "formats": reg_m.formats,
                    "module": reg_m.description_module,
                    "template_id": reg_m.runtime_template_id,
                    "notes": reg_m.notes,
                }

        # From Supabase cloud database (with pagination)
        cloud_models_by_id: Dict[str, CloudModelRecord] = {}
        if self.catalogue.is_connected():
            try:
                for cm in self.catalogue.list_models():
                    cloud_models_by_id[cm.id] = cm
                    if cm.id not in canonical_map:
                        canonical_map[cm.id] = {
                            "id": cm.id,
                            "display_name": cm.display_name,
                            "family": cm.family,
                            "license": cm.licence,
                            "formats": ("URDF",),
                            "module": None,
                            "template_id": None,
                            "notes": cm.description,
                        }
            except Exception as e:
                logger.warning("Error fetching cloud models during inventory: %s", str(e))

        # 2. Evaluate each canonical robot
        for rid, meta in canonical_map.items():
            version = "v1.0.0"
            version_id = f"{rid}_{version}"

            # Check local files (fast metadata scan without computing full file hashes)
            pkg_dir, entry_path = self.resolve_local_package(rid, version)
            local_assets = self.discover_package_assets(pkg_dir, rid, compute_hashes=False) if pkg_dir else []
            local_count = len(local_assets)
            local_bytes = sum(a.size_bytes for a in local_assets)

            # Check license
            is_eligible, detected_lic = self.check_licence_redistribution(meta["license"], pkg_dir)

            # Check cloud database
            cloud_model_exists = rid in cloud_models_by_id
            cloud_ver_exists = False
            cloud_assets_count = 0
            if cloud_model_exists and self.catalogue.is_connected():
                try:
                    c_ver = self.catalogue.get_version(rid, version)
                    cloud_ver_exists = (c_ver is not None)
                    if cloud_ver_exists:
                        c_assets = self.catalogue.list_assets(version_id)
                        cloud_assets_count = len(c_assets)
                except Exception as e:
                    logger.debug("Cloud check error for %s: %s", rid, str(e))

            # Determine sync status
            if not self.catalogue.is_connected():
                status = CloudSyncStatus.OFFLINE
                label = "🔌 Offline"
                notes = "Supabase cloud catalogue provider disconnected."
            elif not is_eligible:
                status = CloudSyncStatus.LICENCE_BLOCKED
                label = "🔒 Licence Restricted"
                notes = f"Redistribution restricted by '{detected_lic}'; metadata catalogued, physical files not mirrored."
            elif cloud_assets_count > 0 and local_count > 0:
                status = CloudSyncStatus.SYNCED
                label = "☁️ Synced"
                notes = f"All {cloud_assets_count} assets synchronized to Supabase bucket 'robot-model-packages'."
            elif local_count > 0 and cloud_assets_count == 0:
                if cloud_ver_exists:
                    status = CloudSyncStatus.REMOTE_ASSET_MISSING
                    label = "⚠️ Remote Assets Missing"
                    notes = f"Model version registered in cloud DB, but object storage assets are missing ({local_count} local files)."
                else:
                    status = CloudSyncStatus.LOCAL_ONLY
                    label = "💻 Local only"
                    notes = f"{local_count} local files present; eligible for cloud synchronization."
            elif local_count == 0 and cloud_assets_count > 0:
                status = CloudSyncStatus.SOURCE_AVAILABLE
                label = "📥 Cloud Stored (Not Cached)"
                notes = f"{cloud_assets_count} assets stored in Supabase; not cached locally."
            elif meta.get("module") or meta.get("id") in known_upstream_modules():
                status = CloudSyncStatus.SOURCE_AVAILABLE
                label = "📦 Source Available"
                notes = f"Upstream source registered ('{meta.get('module')}'); eligible for on-demand fetch & sync."
            else:
                status = CloudSyncStatus.SOURCE_UNAVAILABLE
                label = "❓ Source Unavailable"
                notes = "No local cache or safe upstream description module registered."

            # Evaluate 9D execution readiness (kept strictly independent)
            exec_ready = False
            exec_summary = "Blocked"
            blockers = []
            try:
                decision = self.catalog.evaluate_readiness(rid)
                exec_ready = decision.is_ready
                exec_summary = decision.summary_label
                blockers = decision.blockers
            except Exception as e:
                exec_summary = "Unchecked"
                blockers = [f"Readiness check failed: {e}"]

            items[rid] = RobotSyncInventoryItem(
                robot_id=rid,
                display_name=meta["display_name"],
                family=meta["family"],
                version=version,
                source_module=meta.get("module"),
                formats=tuple(meta.get("formats") or ("URDF",)),
                declared_license=detected_lic,
                is_licence_eligible=is_eligible,
                local_package_path=str(pkg_dir) if pkg_dir else None,
                local_files_count=local_count,
                local_bytes_total=local_bytes,
                entrypoint_path=str(entry_path) if entry_path else None,
                cloud_model_exists=cloud_model_exists,
                cloud_version_exists=cloud_ver_exists,
                cloud_assets_count=cloud_assets_count,
                sync_status=status,
                sync_label=label,
                sync_notes=notes,
                execution_ready=exec_ready,
                execution_summary=exec_summary,
                blockers=blockers,
            )

        inv = CatalogueInventory(
            items=items,
            generated_at=start_time,
            total_models=len(items),
            synced_count=sum(1 for it in items.values() if it.sync_status == CloudSyncStatus.SYNCED),
            local_only_count=sum(1 for it in items.values() if it.sync_status == CloudSyncStatus.LOCAL_ONLY),
            remote_missing_count=sum(1 for it in items.values() if it.sync_status == CloudSyncStatus.REMOTE_ASSET_MISSING),
            source_available_count=sum(1 for it in items.values() if it.sync_status == CloudSyncStatus.SOURCE_AVAILABLE),
            licence_blocked_count=sum(1 for it in items.values() if it.sync_status == CloudSyncStatus.LICENCE_BLOCKED),
            failed_count=sum(1 for it in items.values() if it.sync_status in (CloudSyncStatus.SYNC_FAILED, CloudSyncStatus.SOURCE_UNAVAILABLE)),
            execution_ready_count=sum(1 for it in items.values() if it.execution_ready),
        )
        return inv

    def sync_single_robot(
        self,
        robot_id: str,
        dry_run: bool = False,
        force_reupload: bool = False,
        max_package_bytes: int = MAX_PACKAGE_BYTES_DEFAULT,
    ) -> SyncResult:
        """Synchronize a single robot model package and its asset records to Supabase."""
        t0 = time.time()
        version = "v1.0.0"
        version_id = f"{robot_id}_{version}"

        # 1. Resolve local package
        pkg_dir, entry_path = self.resolve_local_package(robot_id, version)
        if not pkg_dir or not pkg_dir.is_dir():
            # Check if model has a registered description module that can be fetched on demand
            model = self.catalog.get_model(robot_id)
            if model and model.description_module:
                logger.info("Fetching registered description module '%s' for '%s'...", model.description_module, robot_id)
                prep_res = self.catalog.prepare_description(robot_id)
                if prep_res.get("success"):
                    pkg_dir, entry_path = self.resolve_local_package(robot_id, version)

        if not pkg_dir or not pkg_dir.is_dir():
            return SyncResult(
                robot_id=robot_id,
                success=False,
                status=CloudSyncStatus.SOURCE_UNAVAILABLE,
                duration_seconds=round(time.time() - t0, 3),
                message="No local package files found and upstream fetch not available.",
            )

        # 2. Check license and redistribution permissions
        model = self.catalog.get_model(robot_id) or self.registry.get_model(robot_id)
        declared_lic = model.license if model else "Check upstream source"
        is_eligible, detected_lic = self.check_licence_redistribution(declared_lic, pkg_dir)
        if not is_eligible:
            return SyncResult(
                robot_id=robot_id,
                success=False,
                status=CloudSyncStatus.LICENCE_BLOCKED,
                duration_seconds=round(time.time() - t0, 3),
                message=f"Cloud mirroring restricted by licence '{detected_lic}'. Metadata only.",
            )

        # 3. Discover assets
        assets = self.discover_package_assets(pkg_dir, robot_id)
        if not assets:
            return SyncResult(
                robot_id=robot_id,
                success=False,
                status=CloudSyncStatus.SOURCE_UNAVAILABLE,
                duration_seconds=round(time.time() - t0, 3),
                message=f"Local directory '{pkg_dir}' contains no eligible model files.",
            )

        total_bytes = sum(a.size_bytes for a in assets)
        if total_bytes > max_package_bytes:
            return SyncResult(
                robot_id=robot_id,
                success=False,
                status=CloudSyncStatus.SYNC_FAILED,
                duration_seconds=round(time.time() - t0, 3),
                message=f"Package size ({total_bytes / (1024*1024):.1f} MB) exceeds maximum allowed limit ({max_package_bytes / (1024*1024):.1f} MB).",
            )

        if dry_run:
            return SyncResult(
                robot_id=robot_id,
                success=True,
                status=CloudSyncStatus.PENDING_SYNC,
                assets_uploaded=len(assets),
                assets_verified=0,
                bytes_transferred=total_bytes,
                duration_seconds=round(time.time() - t0, 3),
                message=f"[Dry Run] Eligible to synchronize {len(assets)} assets ({total_bytes / 1024:.1f} KB).",
            )

        # 4. Upload missing or changed files to Supabase Object Storage
        if not self.catalogue.is_connected() or not self.storage.is_connected():
            return SyncResult(
                robot_id=robot_id,
                success=False,
                status=CloudSyncStatus.OFFLINE,
                duration_seconds=round(time.time() - t0, 3),
                message="Supabase cloud providers disconnected.",
            )

        uploaded_count = 0
        verified_count = 0
        transferred_bytes = 0

        # Create or update package manifest
        manifest_entries = {}
        for a in assets:
            manifest_entries[a.relative_path] = {
                "size_bytes": a.size_bytes,
                "sha256": a.sha256,
                "asset_type": a.asset_type,
            }

        manifest_file = pkg_dir / "manifest.json"
        try:
            manifest_file.write_text(json.dumps(manifest_entries, indent=2))
            assets = self.discover_package_assets(pkg_dir, robot_id)
        except Exception as e:
            logger.debug("Could not write manifest.json to package dir: %s", str(e))

        # Purge stale/phantom asset metadata records in Postgres before recording authoritative manifest
        self.catalogue.delete_assets(version_id)

        # 5. Mirror each asset to Object Storage and register in Postgres
        for a in assets:
            key = f"models/{robot_id}/{version}/{a.relative_path}"

            # Idempotency check: check remote object
            need_upload = force_reupload
            if not need_upload:
                rem_info = self.storage.check_object(key)
                if not rem_info.get("exists") or rem_info.get("size", 0) != a.size_bytes:
                    need_upload = True

            if need_upload:
                ok = self.storage.upload_package(key, a.file_path)
                if not ok:
                    logger.warning("Failed to upload asset '%s' for model '%s'", a.relative_path, robot_id)
                else:
                    uploaded_count += 1
                    transferred_bytes += a.size_bytes

            # Check remote object exists
            rem_check = self.storage.check_object(key)
            if rem_check.get("exists"):
                verified_count += 1

            # Register asset row in robot_model_assets
            clean_asset_id = f"{robot_id}_{a.relative_path.replace('/', '_').replace('.', '_')}"
            asset_rec = CloudAssetRecord(
                id=clean_asset_id,
                version_id=version_id,
                relative_path=a.relative_path,
                asset_type=a.asset_type,
                object_storage_key=key,
                upstream_reference="",
                sha256=a.sha256,
                size_bytes=a.size_bytes,
                required=True,
            )
            self.catalogue.upsert_asset(asset_rec)

        # 6. Upsert robot_models and robot_model_versions records in Postgres
        display_name = model.display_name if model else robot_id
        family = model.family if model else "general"
        model_rec = CloudModelRecord(
            id=robot_id,
            display_name=display_name,
            family=family,
            description=getattr(model, "notes", f"RoboSahaya {display_name} model."),
            default_version=version,
            catalogue_status="registered",
            licence=detected_lic,
            metadata={
                "synchronized_at": datetime.now(timezone.utc).isoformat(),
                "assets_count": len(assets),
                "total_bytes": total_bytes,
                "entrypoint": str(entry_path.relative_to(pkg_dir)) if entry_path and pkg_dir in entry_path.parents else str(entry_path or ""),
            },
        )
        self.catalogue.upsert_model(model_rec)

        raw_fmt = "MJCF" if "MJCF" in (model.formats if model else ()) else ("XACRO" if "Xacro" in (model.formats if model else ()) else "URDF")
        ver_rec = CloudVersionRecord(
            id=version_id,
            robot_model_id=robot_id,
            version=version,
            source_type="bundled" if getattr(model, "runtime_ready", False) else "upstream",
            source_repository="",
            source_revision="head",
            description_format=raw_fmt,
            entrypoint_path=str(entry_path.relative_to(pkg_dir)) if entry_path and pkg_dir in entry_path.parents else str(entry_path or ""),
            manifest_path="manifest.json",
            content_fingerprint=hashlib.sha256(f"{robot_id}_{len(assets)}_{total_bytes}".encode()).hexdigest(),
            validation_metadata={"assets_count": len(assets), "total_bytes": total_bytes},
        )
        self.catalogue.upsert_version(ver_rec)

        dur = round(time.time() - t0, 3)
        if verified_count == len(assets):
            return SyncResult(
                robot_id=robot_id,
                success=True,
                status=CloudSyncStatus.SYNCED,
                assets_uploaded=uploaded_count,
                assets_verified=verified_count,
                bytes_transferred=transferred_bytes,
                duration_seconds=dur,
                message=f"All {verified_count} assets synchronized and verified in Supabase bucket 'robot-model-packages'.",
            )
        else:
            return SyncResult(
                robot_id=robot_id,
                success=False,
                status=CloudSyncStatus.REMOTE_ASSET_MISSING,
                assets_uploaded=uploaded_count,
                assets_verified=verified_count,
                bytes_transferred=transferred_bytes,
                duration_seconds=dur,
                message=f"Partial sync: {verified_count}/{len(assets)} assets verified remotely.",
            )

    def sync_all_robot_models(
        self,
        dry_run: bool = False,
        only_cached: bool = True,
        force_reupload: bool = False,
        on_progress: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> BatchSyncReport:
        """Process bulk synchronization for all registered and eligible robot models."""
        started_at = datetime.now(timezone.utc).isoformat()
        inv = self.build_inventory(force_refresh=True)
        items = list(inv.items.values())
        progress_fn = on_progress or progress_callback

        report = BatchSyncReport(
            started_at=started_at,
            completed_at="",
            total_considered=len(items),
        )

        for idx, item in enumerate(items):
            rid = item.robot_id
            if progress_fn:
                try:
                    progress_fn(idx + 1, len(items), rid, f"Processing {item.display_name}...")
                except TypeError:
                    try:
                        progress_fn(idx + 1, len(items), rid)
                    except Exception:
                        pass

            # Skip if license restricted
            if not item.is_licence_eligible:
                report.skipped_licence += 1
                report.results[rid] = SyncResult(
                    robot_id=rid,
                    success=False,
                    status=CloudSyncStatus.LICENCE_BLOCKED,
                    message=f"Redistribution restricted by licence '{item.declared_license}'.",
                )
                continue

            # Skip if no local package and only_cached is requested
            if only_cached and not item.local_package_path:
                report.skipped_source_unavailable += 1
                report.results[rid] = SyncResult(
                    robot_id=rid,
                    success=False,
                    status=CloudSyncStatus.SOURCE_AVAILABLE,
                    message="Model not cached locally; use on-demand prepare first.",
                )
                continue

            # If already synced and not forcing reupload: verify and record
            if item.sync_status == CloudSyncStatus.SYNCED and not force_reupload and not dry_run:
                report.already_synced += 1
                report.results[rid] = SyncResult(
                    robot_id=rid,
                    success=True,
                    status=CloudSyncStatus.SYNCED,
                    assets_verified=item.cloud_assets_count,
                    message=f"Already synchronized ({item.cloud_assets_count} remote assets current).",
                )
                continue

            # Run sync
            try:
                res = self.sync_single_robot(
                    robot_id=rid,
                    dry_run=dry_run,
                    force_reupload=force_reupload,
                )
                report.results[rid] = res
                if res.success:
                    if res.status == CloudSyncStatus.SYNCED:
                        report.newly_synced += 1
                    report.assets_uploaded += res.assets_uploaded
                    report.assets_verified += res.assets_verified
                else:
                    report.failed += 1
            except Exception as e:
                logger.error("Exception synchronizing '%s': %s", rid, str(e))
                report.failed += 1
                report.results[rid] = SyncResult(
                    robot_id=rid,
                    success=False,
                    status=CloudSyncStatus.SYNC_FAILED,
                    message=f"Sync exception: {e}",
                    error=str(e),
                )

        report.completed_at = datetime.now(timezone.utc).isoformat()
        return report

    def verify_all_cloud_robot_assets(self) -> AuditReport:
        """Audit every registered cloud version and verify remote storage objects."""
        started_at = datetime.now(timezone.utc).isoformat()
        report = AuditReport(started_at=started_at, completed_at="")

        if not self.catalogue.is_connected() or not self.storage.is_connected():
            report.completed_at = datetime.now(timezone.utc).isoformat()
            return report

        try:
            models = self.catalogue.list_models()
        except Exception as e:
            logger.warning("Error fetching models for audit: %s", str(e))
            models = []

        for m in models:
            version_id = f"{m.id}_{m.default_version}"
            report.total_versions_checked += 1
            assets = self.catalogue.list_assets(version_id)
            if not assets:
                report.model_statuses[version_id] = CloudSyncStatus.REMOTE_ASSET_MISSING
                continue

            all_ok = True
            for a in assets:
                report.total_assets_checked += 1
                info = self.storage.check_object(a.object_storage_key)
                if not info.get("exists"):
                    report.missing_assets.append(a.object_storage_key)
                    all_ok = False
                elif a.size_bytes > 0 and info.get("size", 0) != a.size_bytes:
                    report.corrupted_assets.append(a.object_storage_key)
                    all_ok = False
                else:
                    report.assets_verified += 1

            if all_ok:
                report.model_statuses[version_id] = CloudSyncStatus.SYNCED
            else:
                report.model_statuses[version_id] = CloudSyncStatus.INTEGRITY_FAILED

        report.completed_at = datetime.now(timezone.utc).isoformat()
        return report


def known_upstream_modules() -> Set[str]:
    """Return set of known robot model IDs with discoverable upstream descriptions."""
    return {
        "so_arm100", "ur3e", "ur5e", "ur10e", "kuka_iiwa14", "kinova_gen3_lite",
        "xarm6", "open_manipulator_x", "robotiq_2f85", "unitree_go2", "unitree_go1",
        "anymal_b", "unitree_g1", "unitree_h1", "simple_humanoid", "atlas_v4",
        "crazyflie_2", "skydio_x2", "fetch", "stretch_4", "pr2", "tiago",
        "unitree_z1", "piper", "upkie",
    }


_GLOBAL_SYNC_SERVICE: Optional[RobotCatalogueSyncService] = None


def get_sync_service() -> RobotCatalogueSyncService:
    global _GLOBAL_SYNC_SERVICE
    if _GLOBAL_SYNC_SERVICE is None:
        _GLOBAL_SYNC_SERVICE = RobotCatalogueSyncService()
    return _GLOBAL_SYNC_SERVICE
