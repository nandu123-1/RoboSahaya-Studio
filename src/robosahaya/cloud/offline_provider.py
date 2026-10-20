"""Offline / Local Fallback Providers for RoboSahaya Studio.

Guarantees full functionality and zero-crash degradation when cloud services are unconfigured or offline.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    BaseCatalogueProvider,
    BaseStorageProvider,
    CloudAssetRecord,
    CloudModelRecord,
    CloudVersionRecord,
    RuntimeCompatibilityRecord,
    VerificationRunRecord,
)

logger = logging.getLogger(__name__)


class OfflineCatalogueProvider(BaseCatalogueProvider):
    """Local, in-memory catalogue seeded from registered models."""

    def __init__(self, seeded_models: Optional[List[CloudModelRecord]] = None):
        self._models: Dict[str, CloudModelRecord] = {}
        self._versions: Dict[str, List[CloudVersionRecord]] = {}
        self._assets: Dict[str, List[CloudAssetRecord]] = {}
        self._compatibilities: Dict[str, RuntimeCompatibilityRecord] = {}
        self._verification_runs: List[VerificationRunRecord] = []

        if seeded_models:
            for m in seeded_models:
                self._models[m.id] = m
        else:
            self._seed_default_models()

    def is_connected(self) -> bool:
        # Local catalogue is always accessible offline
        return True

    def _seed_default_models(self) -> None:
        defaults = [
            CloudModelRecord(
                id="differential_drive",
                display_name="RoboSahaya Differential Drive",
                family="wheeled/mobile",
                description="Bundled verified differential-drive mobile robot with 2D LiDAR and camera.",
                aliases=["diff drive", "differential drive", "wheeled base", "sahaya bot", "sahaya_bot"],
                default_version="v1.0.0",
                catalogue_status="registered",
                licence="Apache-2.0",
                licence_url="https://www.apache.org/licenses/LICENSE-2.0",
            ),
            CloudModelRecord(
                id="franka_panda",
                display_name="Franka Emika Panda",
                family="robotic arm",
                description="7-DOF articulated research manipulator arm with parallel gripper.",
                aliases=["franka panda", "panda arm", "panda robot", "robotic_arm_6dof"],
                default_version="v1.0.0",
                catalogue_status="registered",
                licence="Apache-2.0",
            ),
            CloudModelRecord(
                id="so_arm101",
                display_name="SO-ARM101 / SO-101",
                family="robotic arm",
                description="3D-printable 6-DOF follower arm template with Feetech STS3215 bus servos.",
                aliases=["so-101", "so101", "so arm 101", "so-arm101", "lerobot so101"],
                default_version="v1.0.0",
                catalogue_status="registered",
                licence="Apache-2.0",
            ),
            CloudModelRecord(
                id="rsk",
                display_name="Robot Soccer Kit (RSK)",
                family="wheeled/mobile",
                description="MuJoCo Menagerie Robot Soccer Kit 4-wheel omnidirectional mobile soccer robot.",
                aliases=["rsk", "robot soccer kit", "robot_soccer_kit", "soccer robot"],
                default_version="v1.0.0",
                catalogue_status="registered",
                licence="Apache-2.0",
                licence_url="https://github.com/google-deepmind/mujoco_menagerie/tree/main/robot_soccer_kit",
            ),
            CloudModelRecord(
                id="skydio_x2",
                display_name="Skydio X2",
                family="drone/aerial",
                description="Autonomous quadcopter aerial vehicle (Rejection gate target).",
                aliases=["skydio", "skydio drone", "x2 drone"],
                catalogue_status="catalogued",
                licence="Apache-2.0",
            ),
            CloudModelRecord(
                id="so_arm100",
                display_name="SO-ARM100 / SO-100",
                family="robotic arm",
                description="3D-printable 6-DOF arm description (on-demand catalogue target).",
                aliases=["so-100", "so100", "so arm 100", "so-arm100"],
                catalogue_status="catalogued",
                licence="Apache-2.0",
            ),
            CloudModelRecord(
                id="ur5e",
                display_name="Universal Robots UR5e",
                family="robotic arm",
                description="Industrial 6-DOF collaborative robotic arm description.",
                aliases=["ur5", "ur 5e", "universal robots ur5e"],
                catalogue_status="catalogued",
                licence="BSD-3-Clause",
            ),
            CloudModelRecord(
                id="unitree_go2",
                display_name="Unitree Go2",
                family="quadruped",
                description="Quadruped robot description.",
                aliases=["go2", "unitree go2", "go 2"],
                catalogue_status="catalogued",
                licence="BSD-3-Clause",
            ),
        ]
        for m in defaults:
            self._models[m.id] = m
            # Seed default version
            v_record = CloudVersionRecord(
                id=f"{m.id}_v1.0.0",
                robot_model_id=m.id,
                version="v1.0.0",
                source_type="bundled" if m.catalogue_status == "registered" else "upstream",
                source_repository="",
                source_revision="head",
                description_format="MJCF" if m.id == "rsk" else "URDF",
                entrypoint_path=f"model/{m.id}.xml" if m.id == "rsk" else f"model/{m.id}.urdf",
                manifest_path="manifest.json",
                content_fingerprint=hashlib.sha256(m.id.encode()).hexdigest(),
            )
            self._versions[m.id] = [v_record]

    def list_models(self, family: str = "All", search: str = "") -> List[CloudModelRecord]:
        result = []
        q = (search or "").lower().strip()
        for m in self._models.values():
            if family and family != "All" and m.family.lower() != family.lower():
                continue
            if q:
                searchable = f"{m.id} {m.display_name} {m.family} {' '.join(m.aliases)}".lower()
                if q not in searchable:
                    continue
            result.append(m)
        return sorted(result, key=lambda x: x.display_name)

    def get_model(self, model_id: str) -> Optional[CloudModelRecord]:
        if not model_id:
            return None
        if model_id in self._models:
            return self._models[model_id]
        # Check aliases
        q = model_id.lower().strip()
        for m in self._models.values():
            if q == m.id.lower() or q == m.display_name.lower() or any(q == a.lower() for a in m.aliases):
                return m
        return None

    def upsert_model(self, model: CloudModelRecord) -> bool:
        self._models[model.id] = model
        return True

    def get_version(self, model_id: str, version: str = "latest") -> Optional[CloudVersionRecord]:
        versions = self._versions.get(model_id, [])
        if not versions:
            return None
        if version == "latest":
            return versions[-1]
        for v in versions:
            if v.version == version:
                return v
        return None

    def upsert_version(self, version: CloudVersionRecord) -> bool:
        versions = self._versions.setdefault(version.robot_model_id, [])
        for idx, v in enumerate(versions):
            if v.version == version.version:
                versions[idx] = version
                return True
        versions.append(version)
        return True

    def list_assets(self, version_id: str) -> List[CloudAssetRecord]:
        return self._assets.get(version_id, [])

    def upsert_compatibility(self, compat: RuntimeCompatibilityRecord) -> bool:
        self._compatibilities[compat.id] = compat
        return True

    def record_verification_run(self, run: VerificationRunRecord) -> bool:
        self._verification_runs.append(run)
        return True


class OfflineStorageProvider(BaseStorageProvider):
    """Local file-based storage provider for caching and testing."""

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = Path(root_dir or Path.home() / ".cache" / "robosahaya" / "storage_mock").expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def is_connected(self) -> bool:
        return True

    def check_object(self, object_key: str) -> Dict[str, Any]:
        target = self.root_dir / object_key.lstrip("/")
        if target.exists() and target.is_file():
            hasher = hashlib.sha256()
            with open(target, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            return {"exists": True, "size": target.stat().st_size, "sha256": hasher.hexdigest()}
        return {"exists": False, "size": 0, "sha256": None}

    def resolve_download_url(self, object_key: str, expires_in: int = 3600) -> Optional[str]:
        target = self.root_dir / object_key.lstrip("/")
        if target.exists():
            return f"file://{target.resolve()}"
        return None

    def download_package(self, object_key: str, destination: Path) -> bool:
        target = self.root_dir / object_key.lstrip("/")
        if not target.exists():
            return False
        import shutil
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, destination)
        return True

    def upload_package(self, object_key: str, file_path: Path, metadata: Optional[Dict[str, str]] = None) -> bool:
        if not file_path.exists():
            return False
        import shutil
        target = self.root_dir / object_key.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target)
        return True
