"""Unit tests for Secure Model Downloader and Local Cache Manager."""
import json
from pathlib import Path
import pytest

from robosahaya.model_cache import (
    ModelCacheManager,
    SecurityPathError,
    IntegrityError,
)
from robosahaya.cloud.offline_provider import OfflineCatalogueProvider, OfflineStorageProvider


@pytest.fixture
def cache_mgr(tmp_path):
    cat = OfflineCatalogueProvider()
    storage = OfflineStorageProvider(root_dir=tmp_path / "mock_storage")
    return ModelCacheManager(
        cache_root=tmp_path / "cache",
        catalogue_provider=cat,
        storage_provider=storage,
    )


def test_path_safety_rejects_traversal(cache_mgr, tmp_path):
    """Rule 5.3: Reject path traversal in asset paths."""
    base_dir = tmp_path / "sandbox"
    base_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(SecurityPathError, match="Path traversal detected"):
        cache_mgr.validate_path_safety(base_dir, "../outside.txt")

    with pytest.raises(SecurityPathError, match="Path traversal detected"):
        cache_mgr.validate_path_safety(base_dir, "subdir/../../outside.txt")

    with pytest.raises(SecurityPathError, match="Path traversal detected"):
        cache_mgr.validate_path_safety(base_dir, "/etc/passwd")


def test_prepare_model_and_atomic_promotion(cache_mgr):
    """Test on-demand model preparation, manifest creation, and atomic promotion."""
    res = cache_mgr.prepare_model("rsk", version="v1.0.0")
    assert res["success"] is True
    assert res["robot_id"] == "rsk"
    assert res["version"] == "v1.0.0"

    cache_dir = Path(res["cache_dir"])
    assert cache_dir.exists()
    assert (cache_dir / "manifest.json").exists()
    assert (cache_dir / "metadata.json").exists()

    # Second call should reuse verified cache
    res2 = cache_mgr.prepare_model("rsk", version="v1.0.0")
    assert res2["success"] is True
    assert res2["cached"] is True
    assert "Verified local cache reused" in res2["message"]


def test_cache_integrity_detects_tampered_files(cache_mgr):
    """Test SHA-256 integrity check catches corrupted or tampered files."""
    res = cache_mgr.prepare_model("rsk", version="v1.0.0")
    assert res["success"] is True

    # Untouched cache passes integrity check
    integrity = cache_mgr.validate_cache_integrity("rsk", version="v1.0.0")
    assert integrity["valid"] is True

    # Tamper with a file
    cache_dir = Path(res["cache_dir"])
    with open(cache_dir / "manifest.json", "r") as f:
        manifest = json.load(f)

    if manifest.get("assets"):
        first_asset = manifest["assets"][0]["relative_path"]
        target = cache_dir / first_asset
        target.write_bytes(b"tampered and corrupted bytes")

        tampered_integrity = cache_mgr.validate_cache_integrity("rsk", version="v1.0.0")
        assert tampered_integrity["valid"] is False
        assert "Hash mismatch" in tampered_integrity["reason"]


def test_invalidate_cache(cache_mgr):
    """Test safe cache invalidation and directory cleanup."""
    res = cache_mgr.prepare_model("rsk", version="v1.0.0")
    assert res["success"] is True
    assert cache_mgr.is_cache_verified("rsk", version="v1.0.0") is True

    ok = cache_mgr.invalidate_cache("rsk", version="v1.0.0")
    assert ok is True
    assert cache_mgr.is_cache_verified("rsk", version="v1.0.0") is False
