"""Unit tests for Cloud Catalogue and Storage Providers."""
import hashlib
from pathlib import Path
import pytest
import httpx

from robosahaya.cloud.base import CloudModelRecord, VerificationRunRecord
from robosahaya.cloud.offline_provider import OfflineCatalogueProvider, OfflineStorageProvider
from robosahaya.cloud.supabase_provider import SupabaseCatalogueProvider, SupabaseStorageProvider
from robosahaya.cloud.client import get_cloud_status, reset_cloud_clients


def test_offline_catalogue_provider():
    cat = OfflineCatalogueProvider()
    assert cat.is_connected() is True

    # List and filter
    all_models = cat.list_models()
    assert len(all_models) >= 8
    arm_models = cat.list_models(family="robotic arm")
    assert all(m.family == "robotic arm" for m in arm_models)

    # Search
    search_rsk = cat.list_models(search="soccer")
    assert len(search_rsk) == 1
    assert search_rsk[0].id == "rsk"

    # Alias lookup
    by_alias = cat.get_model("panda arm")
    assert by_alias is not None
    assert by_alias.id == "franka_panda"

    # Version lookup
    v = cat.get_version("rsk", "v1.0.0")
    assert v is not None
    assert v.description_format == "MJCF"

    # Verification run recording
    run = VerificationRunRecord(
        id="run_001",
        robot_model_version_id="rsk_v1.0.0",
        adapter_id="mujoco",
        world_id="soccer_pitch",
        test_suite_version="v1.0.0",
        environment_fingerprint="test_env",
        overall_result="PASSED",
        check_results=[{"check": "physics", "passed": True}],
        report_fingerprint="hash_abc",
    )
    assert cat.record_verification_run(run) is True


def test_offline_storage_provider(tmp_path):
    storage = OfflineStorageProvider(root_dir=tmp_path / "storage")
    assert storage.is_connected() is True

    # Create dummy package
    src_file = tmp_path / "package.tar.gz"
    src_file.write_bytes(b"dummy package bytes for robot")
    expected_sha = hashlib.sha256(b"dummy package bytes for robot").hexdigest()

    # Upload
    ok = storage.upload_package("models/rsk/v1.0.0.tar.gz", src_file)
    assert ok is True

    # Check
    info = storage.check_object("models/rsk/v1.0.0.tar.gz")
    assert info["exists"] is True
    assert info["size"] == len(b"dummy package bytes for robot")
    assert info["sha256"] == expected_sha

    # Download
    dest_file = tmp_path / "downloaded.tar.gz"
    ok = storage.download_package("models/rsk/v1.0.0.tar.gz", dest_file)
    assert ok is True
    assert dest_file.read_bytes() == b"dummy package bytes for robot"


def test_supabase_provider_mock_transport():
    """Test Supabase provider API mapping without connecting to live cloud."""
    def mock_handler(request: httpx.Request):
        url_str = str(request.url)
        # Verify authentication headers are present
        assert "apikey" in request.headers
        assert "Authorization" in request.headers
        assert request.headers["apikey"] == "test_service_key"

        if "robot_models" in url_str:
            return httpx.Response(200, json=[{
                "id": "rsk",
                "display_name": "Robot Soccer Kit (RSK)",
                "family": "wheeled/mobile",
                "description": "Test RSK",
                "aliases": ["rsk"],
                "default_model_version": "v1.0.0",
                "catalogue_status": "registered",
                "licence_identifier": "Apache-2.0",
            }])
        if "object/sign" in url_str:
            return httpx.Response(200, json={"signedURL": "/object/sign/robot-model-packages/rsk.tar.gz?token=mock"})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)

    provider = SupabaseCatalogueProvider(base_url="https://mock.supabase.co", service_key="test_service_key")
    # Patch httpx.Client inside provider calls
    orig_client = httpx.Client
    try:
        httpx.Client = lambda **kwargs: orig_client(transport=transport, **kwargs)
        assert provider.is_connected() is True
        models = provider.list_models()
        assert len(models) == 1
        assert models[0].id == "rsk"

        storage = SupabaseStorageProvider(base_url="https://mock.supabase.co", service_key="test_service_key")
        signed_url = storage.resolve_download_url("rsk.tar.gz")
        assert signed_url is not None
        assert "mock" in signed_url
    finally:
        httpx.Client = orig_client


def test_cloud_status_diagnostics_no_secret_leakage(monkeypatch):
    """Rule 4.3: Diagnostics must not leak API keys or tokens."""
    monkeypatch.setenv("SUPABASE_URL", "https://xyzcompany.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "secret_service_role_key_99999")
    reset_cloud_clients()

    status = get_cloud_status()
    # Check that secrets are never exposed in the dictionary
    for k, v in status.items():
        assert "secret_service_role_key_99999" not in str(v)
    reset_cloud_clients()
