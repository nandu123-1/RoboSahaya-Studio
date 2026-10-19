"""Supabase Cloud Provider implementation for RoboSahaya Studio.

Uses HTTP PostgREST and Supabase Storage v1 REST APIs via httpx.
Operates with backend-only service credentials.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

from .base import (
    BaseCatalogueProvider,
    BaseStorageProvider,
    CloudAssetRecord,
    CloudModelRecord,
    CloudVersionRecord,
    VerificationRunRecord,
)

logger = logging.getLogger(__name__)


class SupabaseCatalogueProvider(BaseCatalogueProvider):
    """PostgREST client communicating with Supabase database."""

    def __init__(self, base_url: str, service_key: str, timeout: float = 10.0):
        self.base_url = (base_url or "").rstrip("/")
        self.service_key = service_key or ""
        self.timeout = timeout
        self._rest_url = f"{self.base_url}/rest/v1" if self.base_url else ""

    def _headers(self) -> Dict[str, str]:
        return {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def is_connected(self) -> bool:
        if not self._rest_url or not self.service_key:
            return False
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self._rest_url}/robot_models?select=id&limit=1", headers=self._headers())
                return res.status_code in (200, 206)
        except Exception:
            return False

    def list_models(self, family: str = "All", search: str = "") -> List[CloudModelRecord]:
        if not self._rest_url or not self.service_key:
            return []
        try:
            params: Dict[str, str] = {"select": "*", "order": "display_name.asc"}
            if family and family != "All":
                params["family"] = f"eq.{family}"
            if search:
                clean_s = search.strip().replace("'", "")
                params["or"] = f"(id.ilike.%{clean_s}%,display_name.ilike.%{clean_s}%)"

            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self._rest_url}/robot_models", headers=self._headers(), params=params)
                if res.status_code != 200:
                    logger.warning("Supabase list_models query returned status %d", res.status_code)
                    return []
                data = res.json()
                records = []
                for row in data:
                    records.append(CloudModelRecord(
                        id=row["id"],
                        display_name=row["display_name"],
                        family=row["family"],
                        description=row.get("description", ""),
                        manufacturer=row.get("manufacturer", "Unknown"),
                        aliases=row.get("aliases") or [],
                        default_version=row.get("default_model_version", "v1.0.0"),
                        catalogue_status=row.get("catalogue_status", "catalogued"),
                        licence=row.get("licence_identifier", "Check upstream source"),
                        licence_url=row.get("licence_url", ""),
                        metadata=row.get("metadata_json") or {},
                    ))
                return records
        except Exception as e:
            logger.warning("Error querying Supabase models: %s", str(e))
            return []

    def get_model(self, model_id: str) -> Optional[CloudModelRecord]:
        if not self._rest_url or not self.service_key or not model_id:
            return None
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(
                    f"{self._rest_url}/robot_models",
                    headers=self._headers(),
                    params={"id": f"eq.{model_id}", "select": "*", "limit": "1"}
                )
                if res.status_code == 200:
                    data = res.json()
                    if data:
                        row = data[0]
                        return CloudModelRecord(
                            id=row["id"],
                            display_name=row["display_name"],
                            family=row["family"],
                            description=row.get("description", ""),
                            manufacturer=row.get("manufacturer", "Unknown"),
                            aliases=row.get("aliases") or [],
                            default_version=row.get("default_model_version", "v1.0.0"),
                            catalogue_status=row.get("catalogue_status", "catalogued"),
                            licence=row.get("licence_identifier", "Check upstream source"),
                            licence_url=row.get("licence_url", ""),
                            metadata=row.get("metadata_json") or {},
                        )
        except Exception as e:
            logger.warning("Error fetching model '%s' from Supabase: %s", model_id, str(e))
        return None

    def get_version(self, model_id: str, version: str = "latest") -> Optional[CloudVersionRecord]:
        if not self._rest_url or not self.service_key:
            return None
        try:
            params = {"robot_model_id": f"eq.{model_id}", "select": "*", "limit": "1"}
            if version != "latest":
                params["version"] = f"eq.{version}"
            else:
                params["order"] = "created_at.desc"

            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(f"{self._rest_url}/robot_model_versions", headers=self._headers(), params=params)
                if res.status_code == 200:
                    data = res.json()
                    if data:
                        row = data[0]
                        return CloudVersionRecord(
                            id=row["id"],
                            robot_model_id=row["robot_model_id"],
                            version=row["version"],
                            source_type=row.get("source_type", "upstream"),
                            source_repository=row.get("source_repository", ""),
                            source_revision=row.get("source_revision", ""),
                            description_format=row["description_format"],
                            entrypoint_path=row["entrypoint_path"],
                            manifest_path=row.get("manifest_path", ""),
                            content_fingerprint=row["content_fingerprint"],
                            validation_metadata=row.get("validation_metadata") or {},
                        )
        except Exception as e:
            logger.warning("Error fetching version for '%s': %s", model_id, str(e))
        return None

    def list_assets(self, version_id: str) -> List[CloudAssetRecord]:
        if not self._rest_url or not self.service_key or not version_id:
            return []
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(
                    f"{self._rest_url}/robot_model_assets",
                    headers=self._headers(),
                    params={"robot_model_version_id": f"eq.{version_id}", "select": "*"}
                )
                if res.status_code == 200:
                    records = []
                    for row in res.json():
                        records.append(CloudAssetRecord(
                            id=row["id"],
                            version_id=row["robot_model_version_id"],
                            relative_path=row["relative_path"],
                            asset_type=row["asset_type"],
                            object_storage_key=row.get("object_storage_key", ""),
                            upstream_reference=row.get("upstream_reference", ""),
                            sha256=row["sha256"],
                            size_bytes=row.get("size_bytes", 0),
                            required=row.get("required", True),
                        ))
                    return records
        except Exception as e:
            logger.warning("Error fetching assets for version '%s': %s", version_id, str(e))
        return []

    def record_verification_run(self, run: VerificationRunRecord) -> bool:
        if not self._rest_url or not self.service_key:
            return False
        try:
            payload = {
                "id": run.id,
                "robot_model_version_id": run.robot_model_version_id,
                "adapter_id": run.adapter_id,
                "world_id": run.world_id,
                "test_suite_version": run.test_suite_version,
                "environment_fingerprint": run.environment_fingerprint,
                "overall_result": run.overall_result,
                "check_results": run.check_results,
                "report_fingerprint": run.report_fingerprint,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
            }
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self._rest_url}/robot_verification_runs", headers=self._headers(), json=payload)
                return res.status_code in (200, 201)
        except Exception as e:
            logger.warning("Failed to record verification run to Supabase: %s", str(e))
            return False


class SupabaseStorageProvider(BaseStorageProvider):
    """Client for Supabase Storage v1 REST API."""

    def __init__(self, base_url: str, service_key: str, bucket_name: str = "robot-model-packages", timeout: float = 30.0):
        self.base_url = (base_url or "").rstrip("/")
        self.service_key = service_key or ""
        self.bucket = bucket_name or "robot-model-packages"
        self.timeout = timeout
        self._storage_url = f"{self.base_url}/storage/v1" if self.base_url else ""

    def _headers(self) -> Dict[str, str]:
        return {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
        }

    def is_connected(self) -> bool:
        if not self._storage_url or not self.service_key:
            return False
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self._storage_url}/bucket/{self.bucket}", headers=self._headers())
                return res.status_code in (200, 404)  # 404 means server reachable but bucket not yet created
        except Exception:
            return False

    def check_object(self, object_key: str) -> Dict[str, Any]:
        if not self._storage_url or not self.service_key or not object_key:
            return {"exists": False, "size": 0, "sha256": None}
        try:
            clean_key = object_key.lstrip("/")
            with httpx.Client(timeout=self.timeout) as client:
                res = client.head(f"{self._storage_url}/object/authenticated/{self.bucket}/{clean_key}", headers=self._headers())
                if res.status_code == 200:
                    size = int(res.headers.get("content-length", 0))
                    etag = res.headers.get("etag", "").strip('"')
                    return {"exists": True, "size": size, "sha256": etag}
        except Exception as e:
            logger.debug("check_object failed for %s: %s", object_key, str(e))
        return {"exists": False, "size": 0, "sha256": None}

    def resolve_download_url(self, object_key: str, expires_in: int = 3600) -> Optional[str]:
        if not self._storage_url or not self.service_key or not object_key:
            return None
        try:
            clean_key = object_key.lstrip("/")
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    f"{self._storage_url}/object/sign/{self.bucket}/{clean_key}",
                    headers=self._headers(),
                    json={"expiresIn": expires_in}
                )
                if res.status_code == 200:
                    signed_url = res.json().get("signedURL")
                    if signed_url:
                        return f"{self.base_url}/storage/v1{signed_url}"
        except Exception as e:
            logger.warning("Failed to sign download URL for '%s': %s", object_key, str(e))
        return None

    def download_package(self, object_key: str, destination: Path, max_retries: int = 3) -> bool:
        if not self._storage_url or not self.service_key or not object_key:
            return False
        clean_key = object_key.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = destination.with_suffix(f"{destination.suffix}.part")

        import time
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    with client.stream("GET", f"{self._storage_url}/object/authenticated/{self.bucket}/{clean_key}", headers=self._headers()) as res:
                        if res.status_code == 200:
                            with open(temp_dest, "wb") as f:
                                for chunk in res.iter_bytes(chunk_size=65536):
                                    f.write(chunk)
                            temp_dest.replace(destination)
                            return True
                        elif res.status_code == 404:
                            logger.warning("Supabase storage object not found (404): %s", clean_key)
                            return False
                        else:
                            logger.warning("Supabase download returned status %d on attempt %d", res.status_code, attempt + 1)
            except Exception as e:
                logger.warning("Error downloading package '%s' (attempt %d/%d): %s", object_key, attempt + 1, max_retries, str(e))
                time.sleep(0.5 * (attempt + 1))
            finally:
                if temp_dest.exists() and not destination.exists():
                    temp_dest.unlink(missing_ok=True)

        return False

    def upload_package(self, object_key: str, file_path: Path, metadata: Optional[Dict[str, str]] = None) -> bool:
        if not self._storage_url or not self.service_key or not file_path.exists():
            return False
        try:
            clean_key = object_key.lstrip("/")
            headers = self._headers()
            headers["Content-Type"] = "application/octet-stream"
            with open(file_path, "rb") as f:
                with httpx.Client(timeout=self.timeout * 2) as client:
                    res = client.post(
                        f"{self._storage_url}/object/{self.bucket}/{clean_key}",
                        headers=headers,
                        content=f.read()
                    )
                    return res.status_code in (200, 201)
        except Exception as e:
            logger.warning("Error uploading package '%s': %s", object_key, str(e))
            return False
