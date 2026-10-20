"""Comprehensive unit tests for the Universal Bulk Robot Model Synchronization Service."""
import tempfile
from pathlib import Path
import pytest

from robosahaya.cloud.base import CloudAssetRecord, CloudModelRecord, CloudSyncStatus, CloudVersionRecord
from robosahaya.cloud.offline_provider import OfflineCatalogueProvider, OfflineStorageProvider
from robosahaya.cloud.sync_service import (
    PERMISSIVE_LICENSES,
    AuditReport,
    BatchSyncReport,
    CatalogueInventory,
    RobotCatalogueSyncService,
    RobotSyncInventoryItem,
    SyncResult,
)
from robosahaya.robot_catalog import RobotModelCatalog


@pytest.fixture
def temp_workspace(tmp_path: Path):
    """Set up temporary directories simulating templates and cache packages."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir(parents=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    # 1. Dummy differential drive template
    diff_dir = templates_dir / "differential_drive"
    diff_dir.mkdir()
    (diff_dir / "model.urdf.xacro").write_text("<robot name='diff_drive'></robot>")
    (diff_dir / "model.config").write_text("<model></model>")

    # 2. Dummy SO-ARM100 cache package with Apache-2.0 LICENSE
    so100_dir = cache_dir / "robot_descriptions" / "SO-ARM100" / "Simulation" / "SO100"
    so100_dir.mkdir(parents=True)
    (so100_dir / "so100.urdf").write_text("<robot name='so100'></robot>")
    (so100_dir / "LICENSE").write_text("Apache License Version 2.0")
    mesh_dir = so100_dir / "assets"
    mesh_dir.mkdir()
    (mesh_dir / "link.stl").write_bytes(b"solid dummy stl binary content")

    # 3. Dummy restricted license robot
    prop_dir = cache_dir / "robot_descriptions" / "proprietary_bot"
    prop_dir.mkdir(parents=True)
    (prop_dir / "robot.urdf").write_text("<robot name='prop'></robot>")
    (prop_dir / "LICENSE").write_text("Proprietary and Confidential. No redistribution.")

    storage_root = tmp_path / "mock_storage"
    storage_root.mkdir()

    return {
        "templates": templates_dir,
        "cache": cache_dir,
        "storage_root": storage_root,
        "so100_dir": so100_dir,
    }


def test_license_redistribution_check(temp_workspace):
    offline_cat = OfflineCatalogueProvider()
    storage = OfflineStorageProvider(root_dir=temp_workspace["storage_root"])
    svc = RobotCatalogueSyncService(
        catalogue_provider=offline_cat,
        storage_provider=storage,
        templates_root=temp_workspace["templates"],
        cache_base=temp_workspace["cache"],
    )

    # Permissive licenses
    ok, lic = svc.check_licence_redistribution("Apache-2.0")
    assert ok is True
    assert lic == "Apache-2.0"

    ok, lic = svc.check_licence_redistribution("MIT")
    assert ok is True

    ok, lic = svc.check_licence_redistribution("BSD-3-Clause")
    assert ok is True

    # File-based inspection
    ok, lic = svc.check_licence_redistribution("Check upstream source", temp_workspace["so100_dir"])
    assert ok is True
    assert lic == "Apache-2.0"

    # Restricted licenses
    ok, lic = svc.check_licence_redistribution("Non-Commercial Only")
    assert ok is False


def test_discover_package_assets(temp_workspace):
    offline_cat = OfflineCatalogueProvider()
    storage = OfflineStorageProvider(root_dir=temp_workspace["storage_root"])
    svc = RobotCatalogueSyncService(
        catalogue_provider=offline_cat,
        storage_provider=storage,
        templates_root=temp_workspace["templates"],
        cache_base=temp_workspace["cache"],
    )

    assets = svc.discover_package_assets(temp_workspace["so100_dir"], "so_arm100")
    assert len(assets) == 3  # so100.urdf, LICENSE, link.stl
    rel_paths = {a.relative_path for a in assets}
    assert "so100.urdf" in rel_paths
    assert "assets/link.stl" in rel_paths
    assert "LICENSE" in rel_paths

    stl_asset = next(a for a in assets if a.relative_path == "assets/link.stl")
    assert stl_asset.asset_type == "mesh"
    assert stl_asset.size_bytes == len(b"solid dummy stl binary content")
    assert len(stl_asset.sha256) == 64


def test_dry_run_sync(temp_workspace):
    offline_cat = OfflineCatalogueProvider()
    storage = OfflineStorageProvider(root_dir=temp_workspace["storage_root"])
    svc = RobotCatalogueSyncService(
        catalogue_provider=offline_cat,
        storage_provider=storage,
        templates_root=temp_workspace["templates"],
        cache_base=temp_workspace["cache"],
    )

    # Dry run should not write to storage or catalogue
    res = svc.sync_single_robot("so_arm100", dry_run=True)
    assert res.success is True
    assert res.status == CloudSyncStatus.PENDING_SYNC
    assert res.assets_uploaded == 3
    assert res.assets_verified == 0
    assert "[Dry Run]" in res.message

    # Storage should be empty
    assert len(list(temp_workspace["storage_root"].rglob("*"))) == 0


def test_live_sync_offline_provider(temp_workspace):
    offline_cat = OfflineCatalogueProvider()
    storage = OfflineStorageProvider(root_dir=temp_workspace["storage_root"])
    svc = RobotCatalogueSyncService(
        catalogue_provider=offline_cat,
        storage_provider=storage,
        templates_root=temp_workspace["templates"],
        cache_base=temp_workspace["cache"],
    )

    res = svc.sync_single_robot("so_arm100", dry_run=False)
    assert res.success is True
    assert res.status == CloudSyncStatus.SYNCED
    assert res.assets_uploaded >= 3
    assert res.assets_verified >= 3

    # Check that metadata rows were written
    model_rec = offline_cat.get_model("so_arm100")
    assert model_rec is not None
    assert model_rec.id == "so_arm100"

    version_rec = offline_cat.get_version("so_arm100", "v1.0.0")
    assert version_rec is not None

    assets = offline_cat.list_assets("so_arm100_v1.0.0")
    assert len(assets) >= 3

    # Check files in storage
    for a in assets:
        chk = storage.check_object(a.object_storage_key)
        assert chk["exists"] is True
        assert chk["size"] == a.size_bytes


def test_decoupled_execution_readiness(temp_workspace):
    """Ensure cloud sync NEVER alters simulator execution readiness."""
    offline_cat = OfflineCatalogueProvider()
    storage = OfflineStorageProvider(root_dir=temp_workspace["storage_root"])
    svc = RobotCatalogueSyncService(
        catalogue_provider=offline_cat,
        storage_provider=storage,
        templates_root=temp_workspace["templates"],
        cache_base=temp_workspace["cache"],
    )

    # Sync so_arm100 to cloud
    res = svc.sync_single_robot("so_arm100", dry_run=False)
    assert res.status == CloudSyncStatus.SYNCED

    # Build inventory: so_arm100 must be Synced in cloud, but execution_ready MUST be False
    inv = svc.build_inventory()
    so100_item = inv.items.get("so_arm100")
    assert so100_item is not None
    assert so100_item.sync_status == CloudSyncStatus.SYNCED
    assert so100_item.execution_ready is False
    assert len(so100_item.blockers) > 0


def test_batch_sync_orchestrator(temp_workspace):
    offline_cat = OfflineCatalogueProvider()
    storage = OfflineStorageProvider(root_dir=temp_workspace["storage_root"])
    svc = RobotCatalogueSyncService(
        catalogue_provider=offline_cat,
        storage_provider=storage,
        templates_root=temp_workspace["templates"],
        cache_base=temp_workspace["cache"],
    )

    progress_events = []

    def on_progress(cur, total, rid):
        progress_events.append((cur, total, rid))

    report = svc.sync_all_robot_models(
        only_cached=True,
        dry_run=False,
        progress_callback=on_progress,
    )
    assert isinstance(report, BatchSyncReport)
    assert report.total_considered > 0
    assert len(progress_events) == report.total_considered
    assert report.results["so_arm100"].success is True


def test_asset_audit_report(temp_workspace):
    offline_cat = OfflineCatalogueProvider()
    storage = OfflineStorageProvider(root_dir=temp_workspace["storage_root"])
    svc = RobotCatalogueSyncService(
        catalogue_provider=offline_cat,
        storage_provider=storage,
        templates_root=temp_workspace["templates"],
        cache_base=temp_workspace["cache"],
    )

    # Sync so_arm100
    svc.sync_single_robot("so_arm100", dry_run=False)

    # Audit should pass
    audit = svc.verify_all_cloud_robot_assets()
    assert isinstance(audit, AuditReport)
    assert audit.total_versions_checked >= 1
    assert "so_arm100_v1.0.0" in audit.model_statuses
    assert audit.model_statuses["so_arm100_v1.0.0"] == CloudSyncStatus.SYNCED
    assert len(audit.missing_assets) == 0

    # Simulate missing asset
    target = temp_workspace["storage_root"] / "models/so_arm100/v1.0.0/so100.urdf"
    if target.exists():
        target.unlink()

    # Audit should detect corruption / missing object
    audit2 = svc.verify_all_cloud_robot_assets()
    assert audit2.model_statuses["so_arm100_v1.0.0"] == CloudSyncStatus.INTEGRITY_FAILED
    assert len(audit2.missing_assets) > 0


def test_catalog_table_integration():
    cat = RobotModelCatalog()
    inv = cat.get_sync_inventory(refresh_remote=False)
    assert isinstance(inv, CatalogueInventory)
    assert len(inv.items) >= 8

    rows = cat.format_sync_table_rows(inv)
    assert len(rows) == len(inv.items)
    for r in rows:
        assert len(r) == 8  # 8 columns
        assert r[0] in inv.items
