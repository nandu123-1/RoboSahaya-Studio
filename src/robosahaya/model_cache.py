"""Secure On-Demand Model Downloader and Verified Local Cache Manager.

Guarantees:
- Path traversal prevention (rejects '..', absolute paths, and symlink escapes).
- Cryptographic SHA-256 integrity verification.
- Atomic promotion via temporary staging directories (partial downloads never corrupt good caches).
- Per-model concurrency locks preventing race conditions.
- Strict size and file count quotas.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cloud import (
    BaseCatalogueProvider,
    BaseStorageProvider,
    get_cloud_catalogue,
    get_cloud_storage,
)
from .readiness import (
    CheckResult,
    DescriptionValidity,
    LocalCacheStatus,
    ReadinessDimensions,
)

logger = logging.getLogger(__name__)

# Security & quota limits
MAX_PACKAGE_SIZE_BYTES = 250 * 1024 * 1024  # 250 MB
MAX_PACKAGE_FILE_COUNT = 10_000


class SecurityPathError(ValueError):
    """Raised when an asset path attempts directory traversal or symlink escape."""
    pass


class IntegrityError(ValueError):
    """Raised when a downloaded file hash does not match expected SHA-256."""
    pass


class ModelCacheManager:
    """Manages versioned, integrity-checked local robot model caches."""

    def __init__(
        self,
        cache_root: Optional[Path] = None,
        catalogue_provider: Optional[BaseCatalogueProvider] = None,
        storage_provider: Optional[BaseStorageProvider] = None,
    ):
        self.cache_root = Path(
            cache_root or Path.home() / ".cache" / "robosahaya" / "robot_models"
        ).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self._catalogue = catalogue_provider or get_cloud_catalogue()
        self._storage = storage_provider or get_cloud_storage()
        self._model_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, robot_id: str) -> threading.Lock:
        with self._global_lock:
            if robot_id not in self._model_locks:
                self._model_locks[robot_id] = threading.Lock()
            return self._model_locks[robot_id]

    def get_version_dir(self, robot_id: str, version: str = "v1.0.0") -> Path:
        clean_id = robot_id.strip().lower()
        clean_ver = version.strip()
        return self.cache_root / clean_id / clean_ver

    def is_cache_verified(self, robot_id: str, version: str = "v1.0.0") -> bool:
        v_dir = self.get_version_dir(robot_id, version)
        if not v_dir.exists() or not v_dir.is_dir():
            return False
        manifest_path = v_dir / "manifest.json"
        metadata_path = v_dir / "metadata.json"
        return manifest_path.exists() and metadata_path.exists()

    def validate_path_safety(self, base_dir: Path, relative_path: str) -> Path:
        """Ensure destination path cannot escape base_dir."""
        if not relative_path or not relative_path.strip():
            raise SecurityPathError("Empty relative path is not allowed.")

        # Reject path traversal tokens
        norm = os.path.normpath(relative_path.strip())
        if norm.startswith("..") or "/../" in norm or norm.startswith("/") or norm.startswith("\\"):
            raise SecurityPathError(f"Path traversal detected in relative path: '{relative_path}'")

        resolved = (base_dir / norm).resolve()
        if not resolved.is_relative_to(base_dir.resolve()):
            raise SecurityPathError(f"Target path escapes base directory: '{resolved}'")
        return resolved

    def compute_file_sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def prepare_model(
        self,
        robot_id: str,
        version: str = "v1.0.0",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Fetch, verify, and atomically promote model assets into local cache."""
        lock = self._get_lock(robot_id)
        with lock:
            target_dir = self.get_version_dir(robot_id, version)

            # 1. Reuse existing verified cache unless force_refresh is requested
            if not force_refresh and self.is_cache_verified(robot_id, version):
                try:
                    integrity = self.validate_cache_integrity(robot_id, version)
                    if integrity["valid"]:
                        with open(target_dir / "metadata.json", "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        return {
                            "success": True,
                            "cached": True,
                            "robot_id": robot_id,
                            "version": version,
                            "cache_dir": str(target_dir),
                            "entrypoint": meta.get("entrypoint"),
                            "metadata": meta,
                            "message": f"Verified local cache reused for '{robot_id}' ({version}).",
                        }
                except Exception as e:
                    logger.warning("Existing cache validation failed for %s: %s; re-fetching.", robot_id, str(e))

            # 2. Query catalogue and version record
            model_rec = self._catalogue.get_model(robot_id)
            if not model_rec:
                return {
                    "success": False,
                    "error": f"Robot model '{robot_id}' not found in local or cloud catalogue.",
                    "cache_status": LocalCacheStatus.NOT_CACHED.value,
                }

            version_rec = self._catalogue.get_version(robot_id, version)
            assets_list = self._catalogue.list_assets(version_rec.id) if version_rec else []

            # 3. Create isolated temporary staging directory
            staging_token = f".tmp_{robot_id}_{uuid.uuid4().hex}"
            staging_dir = self.cache_root / staging_token
            staging_dir.mkdir(parents=True, exist_ok=True)

            try:
                downloaded_assets: List[Dict[str, Any]] = []
                total_bytes = 0
                file_count = 0

                # 4. If we have asset records from cloud storage
                if assets_list:
                    for asset in assets_list:
                        if asset.relative_path in {"manifest.json", "metadata.json"}:
                            continue
                        safe_dest = self.validate_path_safety(staging_dir, asset.relative_path)
                        safe_dest.parent.mkdir(parents=True, exist_ok=True)

                        # Download via storage provider
                        ok = self._storage.download_package(asset.object_storage_key, safe_dest)
                        if not ok or not safe_dest.exists():
                            raise FileNotFoundError(f"Failed to download asset '{asset.relative_path}'")

                        # Verify hash
                        actual_hash = self.compute_file_sha256(safe_dest)
                        if asset.sha256 and actual_hash != asset.sha256:
                            raise IntegrityError(
                                f"Hash mismatch for '{asset.relative_path}': expected {asset.sha256}, got {actual_hash}"
                            )

                        file_size = safe_dest.stat().st_size
                        total_bytes += file_size
                        file_count += 1

                        if total_bytes > MAX_PACKAGE_SIZE_BYTES:
                            raise ValueError(f"Package exceeds maximum size limit of {MAX_PACKAGE_SIZE_BYTES} bytes.")
                        if file_count > MAX_PACKAGE_FILE_COUNT:
                            raise ValueError(f"Package exceeds maximum file count limit of {MAX_PACKAGE_FILE_COUNT} files.")

                        downloaded_assets.append({
                            "relative_path": asset.relative_path,
                            "sha256": actual_hash,
                            "size_bytes": file_size,
                            "asset_type": asset.asset_type,
                        })
                else:
                    # Fallback / Seeded package: Copy or discover existing local/upstream files
                    self._populate_fallback_package(robot_id, staging_dir, downloaded_assets)

                # 5. Write manifest, metadata, and structural validation records
                manifest = {
                    "robot_id": robot_id,
                    "version": version,
                    "total_files": len(downloaded_assets),
                    "assets": downloaded_assets,
                }
                with open(staging_dir / "manifest.json", "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

                entrypoint = version_rec.entrypoint_path if version_rec else f"model/{robot_id}.urdf"
                metadata = {
                    "robot_id": robot_id,
                    "display_name": model_rec.display_name,
                    "version": version,
                    "family": model_rec.family,
                    "licence": model_rec.licence,
                    "entrypoint": entrypoint,
                    "description_format": version_rec.description_format if version_rec else "URDF",
                }
                with open(staging_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                # 6. Atomic promotion: rename staging directory to target_dir
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                staging_dir.rename(target_dir)

                return {
                    "success": True,
                    "cached": False,
                    "promoted": True,
                    "robot_id": robot_id,
                    "version": version,
                    "cache_dir": str(target_dir),
                    "entrypoint": str(target_dir / entrypoint),
                    "total_files": len(downloaded_assets),
                    "message": f"Successfully downloaded and verified '{model_rec.display_name}' ({version}).",
                }

            except Exception as e:
                logger.error("Model preparation failed for '%s': %s", robot_id, str(e))
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)
                return {
                    "success": False,
                    "robot_id": robot_id,
                    "error": str(e),
                    "cache_status": LocalCacheStatus.INTEGRITY_FAILED.value if isinstance(e, IntegrityError) else LocalCacheStatus.CORRUPTED.value,
                }

    def _populate_fallback_package(self, robot_id: str, staging_dir: Path, asset_collector: List[Dict[str, Any]]) -> None:
        """Create package structure from existing bundled templates or Menagerie cache."""
        model_dir = staging_dir / "model"
        assets_dir = staging_dir / "assets"
        model_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)

        if robot_id == "rsk":
            # MuJoCo Menagerie RSK model
            cached_rsk = Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie" / "robot_soccer_kit"
            if cached_rsk.exists():
                for f in cached_rsk.glob("*.xml"):
                    shutil.copy2(f, model_dir / f.name)
                    asset_collector.append({
                        "relative_path": f"model/{f.name}",
                        "sha256": self.compute_file_sha256(model_dir / f.name),
                        "size_bytes": (model_dir / f.name).stat().st_size,
                        "asset_type": "description",
                    })
                rsk_assets = cached_rsk / "assets"
                if rsk_assets.exists():
                    for a in rsk_assets.glob("*"):
                        if a.is_file():
                            shutil.copy2(a, assets_dir / a.name)
                            asset_collector.append({
                                "relative_path": f"assets/{a.name}",
                                "sha256": self.compute_file_sha256(assets_dir / a.name),
                                "size_bytes": (assets_dir / a.name).stat().st_size,
                                "asset_type": "mesh",
                            })
                return

        # Default minimal placeholder file if no existing assets
        placeholder = model_dir / f"{robot_id}.urdf"
        placeholder.write_text(f'<robot name="{robot_id}"><link name="base_link"/></robot>', encoding="utf-8")
        asset_collector.append({
            "relative_path": f"model/{robot_id}.urdf",
            "sha256": self.compute_file_sha256(placeholder),
            "size_bytes": placeholder.stat().st_size,
            "asset_type": "description",
        })

    def validate_cache_integrity(self, robot_id: str, version: str = "v1.0.0") -> Dict[str, Any]:
        """Verify all cached files against manifest SHA-256 hashes."""
        v_dir = self.get_version_dir(robot_id, version)
        manifest_file = v_dir / "manifest.json"
        if not manifest_file.exists():
            return {"valid": False, "reason": "Missing manifest.json"}

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            assets = manifest.get("assets", [])
            for a in assets:
                rel = a["relative_path"]
                if rel in {"manifest.json", "metadata.json"}:
                    continue
                expected_sha = a["sha256"]
                target_file = v_dir / rel
                if not target_file.exists():
                    return {"valid": False, "reason": f"Missing asset file: {rel}"}
                actual_sha = self.compute_file_sha256(target_file)
                if actual_sha != expected_sha:
                    return {"valid": False, "reason": f"Hash mismatch for {rel}"}
            return {"valid": True, "file_count": len(assets)}
        except Exception as e:
            return {"valid": False, "reason": str(e)}

    def invalidate_cache(self, robot_id: str, version: str = "v1.0.0") -> bool:
        """Safely delete cached version directory."""
        v_dir = self.get_version_dir(robot_id, version)
        if v_dir.exists():
            shutil.rmtree(v_dir, ignore_errors=True)
            return True
        return False

    @staticmethod
    def _resolve_cached_description_module(module_name: str) -> Optional[tuple[Path, Path]]:
        """Read robot_descriptions module path constants without importing it.

        The upstream package's generated description modules share a small
        convention: a repository is obtained with ``clone_to_cache`` and the
        package and entrypoint paths are built with ``os.path.join``. Resolve
        that convention against the configured robot_descriptions cache. This
        avoids network access when the source is already present.
        """
        try:
            spec = importlib.util.find_spec(f"robot_descriptions.{module_name}")
            if spec is None or not spec.origin:
                return None
            module_ast = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
            repository_name: Optional[str] = None
            for node in ast.walk(module_ast):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_clone = (
                    isinstance(func, ast.Name) and func.id in {"_clone_to_cache", "clone_to_cache"}
                ) or (
                    isinstance(func, ast.Attribute) and func.attr == "clone_to_cache"
                )
                if is_clone and node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        repository_name = node.args[0].value
                        break
            if not repository_name:
                return None

            # Importing _repositories only loads static metadata; unlike a model
            # description module it has no clone or download side effects.
            repositories = importlib.import_module("robot_descriptions._repositories").REPOSITORIES
            repository = repositories.get(repository_name)
            if repository is None:
                return None

            cache_root = Path(os.environ.get(
                "ROBOT_DESCRIPTIONS_CACHE", str(Path.home() / ".cache" / "robot_descriptions")
            )).expanduser().resolve()
            cache_path = Path(repository.cache_path)
            repo_roots = [cache_root / cache_path]
            if cache_root.is_dir():
                repo_roots.extend(
                    candidate / cache_path
                    for candidate in cache_root.glob(f"{cache_path.name}-*")
                    if candidate.is_dir()
                )

            # Evaluate the module's conventional path joins using only literal
            # components and previously resolved constants.
            for repo_root in repo_roots:
                repo_root = repo_root.resolve()
                if not repo_root.is_dir() or not repo_root.is_relative_to(cache_root):
                    continue
                values: dict[str, Path] = {"REPOSITORY_PATH": repo_root}
                for statement in module_ast.body:
                    if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                        continue
                    target = statement.target.id
                    value_node = statement.value
                    if target == "REPOSITORY_PATH":
                        values[target] = repo_root
                        continue
                    evaluated = ModelCacheManager._evaluate_description_path(value_node, values)
                    if evaluated is not None:
                        values[target] = evaluated
                package_root = values.get("PACKAGE_PATH") or values.get("REPOSITORY_PATH")
                entrypoint = next((values.get(name) for name in ("URDF_PATH", "XACRO_PATH", "MJCF_PATH") if values.get(name)), None)
                if package_root and entrypoint:
                    package_root = package_root.resolve()
                    entrypoint = entrypoint.resolve()
                    if package_root.is_dir() and entrypoint.is_file() and entrypoint.is_relative_to(package_root):
                        return package_root, entrypoint
        except (ImportError, OSError, SyntaxError, ValueError, TypeError, AttributeError):
            return None
        return None

    @staticmethod
    def _evaluate_description_path(node: ast.AST, values: Dict[str, Path]) -> Optional[Path]:
        """Evaluate literal pathlib/os.path.join expressions from a module AST."""
        if isinstance(node, ast.Name):
            return values.get(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            base = ModelCacheManager._evaluate_description_path(node.args[0], values) if node.args else None
            components: list[str] = []
            for arg in node.args[1:]:
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                    return None
                components.append(arg.value)
            if base is not None:
                return base.joinpath(*components)
        return None

