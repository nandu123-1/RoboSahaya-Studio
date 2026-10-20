"""Abstract interfaces for Cloud Robot Catalogue and Storage Providers.

Decouples core RoboSahaya Studio logic from any specific cloud vendor (Supabase, AWS S3, local filesystem).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class CloudSyncStatus(str, Enum):
    LOCAL_ONLY = "local_only"
    CLOUD_AVAILABLE = "cloud_available"
    PENDING_UPLOAD = "pending_upload"
    UPLOADING = "uploading"
    SYNCED = "synced"
    SYNC_FAILED = "sync_failed"
    VERSION_CONFLICT = "version_conflict"
    REMOTE_ASSET_MISSING = "remote_asset_missing"
    LOCAL_CACHE_MISSING = "local_cache_missing"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass
class ModelSyncState:
    robot_id: str
    status: CloudSyncStatus
    display_label: str
    details: str
    local_version: str = "v1.0.0"
    cloud_version: Optional[str] = None
    local_assets_count: int = 0
    remote_assets_count: int = 0
    verified_run_count: int = 0
    last_synced_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "status": self.status.value,
            "display_label": self.display_label,
            "details": self.details,
            "local_version": self.local_version,
            "cloud_version": self.cloud_version,
            "local_assets_count": self.local_assets_count,
            "remote_assets_count": self.remote_assets_count,
            "verified_run_count": self.verified_run_count,
            "last_synced_at": self.last_synced_at,
        }


@dataclass
class CloudModelRecord:
    id: str
    display_name: str
    family: str
    description: str = ""
    manufacturer: str = "Unknown"
    aliases: List[str] = field(default_factory=list)
    default_version: str = "v1.0.0"
    catalogue_status: str = "catalogued"
    licence: str = "Check upstream source"
    licence_url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloudVersionRecord:
    id: str
    robot_model_id: str
    version: str
    source_type: str = "upstream"
    source_repository: str = ""
    source_revision: str = "head"
    description_format: str = "URDF"
    entrypoint_path: str = ""
    manifest_path: str = "manifest.json"
    content_fingerprint: str = ""
    validation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloudAssetRecord:
    id: str
    version_id: str
    relative_path: str
    asset_type: str
    object_storage_key: str
    upstream_reference: str = ""
    sha256: str = ""
    size_bytes: int = 0
    required: bool = True


@dataclass
class RuntimeCompatibilityRecord:
    id: str
    robot_model_version_id: str
    adapter_id: str
    compatibility_status: str
    simulator_version: str = ""
    required_plugins: List[str] = field(default_factory=list)
    required_controllers: List[str] = field(default_factory=list)
    required_sensors: List[str] = field(default_factory=list)
    required_bridges: List[str] = field(default_factory=list)
    known_limitations: str = ""
    last_validation_id: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class VerificationRunRecord:
    id: str
    robot_model_version_id: str
    adapter_id: str
    world_id: str
    test_suite_version: str
    environment_fingerprint: str
    overall_result: str
    check_results: List[Dict[str, Any]]
    report_fingerprint: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    test_profile: Dict[str, Any] = field(default_factory=dict)


class BaseCatalogueProvider(ABC):
    """Abstract interface for querying and persisting robot metadata."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the remote database is accessible."""
        ...

    @abstractmethod
    def list_models(self, family: str = "All", search: str = "") -> List[CloudModelRecord]:
        """List available models with optional search and family filter."""
        ...

    @abstractmethod
    def get_model(self, model_id: str) -> Optional[CloudModelRecord]:
        """Get model metadata by canonical ID or alias."""
        ...

    @abstractmethod
    def upsert_model(self, model: CloudModelRecord) -> bool:
        """Persist or update a model record."""
        ...

    @abstractmethod
    def get_version(self, model_id: str, version: str = "latest") -> Optional[CloudVersionRecord]:
        """Get a specific version record for a robot model."""
        ...

    @abstractmethod
    def upsert_version(self, version: CloudVersionRecord) -> bool:
        """Persist or update a model version record."""
        ...

    @abstractmethod
    def list_assets(self, version_id: str) -> List[CloudAssetRecord]:
        """List all asset records belonging to a model version."""
        ...

    @abstractmethod
    def upsert_asset(self, asset: CloudAssetRecord) -> bool:
        """Persist or update a model asset record."""
        ...

    @abstractmethod
    def upsert_compatibility(self, compat: RuntimeCompatibilityRecord) -> bool:
        """Persist or update a simulator compatibility record."""
        ...

    @abstractmethod
    def record_verification_run(self, run: VerificationRunRecord) -> bool:
        """Persist a verified execution or simulation run record."""
        ...


class BaseStorageProvider(ABC):
    """Abstract interface for downloading and uploading robot packages and assets."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if object storage is configured and reachable."""
        ...

    @abstractmethod
    def check_object(self, object_key: str) -> Dict[str, Any]:
        """Check if an object exists and retrieve its metadata (exists, size, sha256)."""
        ...

    @abstractmethod
    def resolve_download_url(self, object_key: str, expires_in: int = 3600) -> Optional[str]:
        """Get a short-lived signed URL or direct URI for downloading an object."""
        ...

    @abstractmethod
    def download_package(self, object_key: str, destination: Path) -> bool:
        """Download a model package archive or asset to destination."""
        ...

    @abstractmethod
    def upload_package(self, object_key: str, file_path: Path, metadata: Optional[Dict[str, str]] = None) -> bool:
        """Upload a model package archive to object storage (when permitted)."""
        ...
