"""Unit tests for multi-state cloud synchronization and asset tracking."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from robosahaya.cloud.base import (
    CloudAssetRecord,
    CloudModelRecord,
    CloudSyncStatus,
    CloudVersionRecord,
    ModelSyncState,
)
from robosahaya.cloud.offline_provider import OfflineCatalogueProvider
from robosahaya.cloud.verify_runtimes import RuntimeVerifier
from robosahaya.robot_catalog import RobotModelCatalog


def test_model_sync_state_dataclass():
    st = ModelSyncState(
        robot_id="so_arm101",
        status=CloudSyncStatus.SYNCED,
        display_label="☁️ Synced",
        details="16 assets synchronized",
        local_assets_count=16,
        remote_assets_count=16,
    )
    d = st.to_dict()
    assert d["robot_id"] == "so_arm101"
    assert d["status"] == "synced"
    assert d["display_label"] == "☁️ Synced"
    assert d["local_assets_count"] == 16
    assert d["remote_assets_count"] == 16


def test_compute_sync_state_offline_provider():
    offline_cat = OfflineCatalogueProvider(seeded_models=[])
    verifier = RuntimeVerifier(catalogue_provider=offline_cat)

    # Before syncing: local only
    st_before = verifier.compute_sync_state("so_arm101")
    assert st_before.status == CloudSyncStatus.LOCAL_ONLY
    assert "Local only" in st_before.display_label

    # Verify robot with sync_to_cloud=True
    report = verifier.verify_robot("so_arm101", sync_to_cloud=True)
    assert report.sync_state is not None
    assert report.sync_state.status == CloudSyncStatus.SYNCED
    assert "Synced" in report.sync_state.display_label


def test_compute_sync_state_disconnected():
    mock_cat = MagicMock()
    mock_cat.is_connected.return_value = False
    verifier = RuntimeVerifier(catalogue_provider=mock_cat)

    st = verifier.compute_sync_state("franka_panda")
    assert st.status == CloudSyncStatus.OFFLINE
    assert "Offline" in st.display_label


def test_compute_sync_state_remote_asset_missing():
    mock_cat = MagicMock()
    mock_cat.is_connected.return_value = True
    mock_cat.list_assets.return_value = []
    mock_cat.get_version.return_value = CloudVersionRecord(
        id="so_arm100_v1.0.0",
        robot_model_id="so_arm100",
        version="v1.0.0",
        description_format="URDF",
        entrypoint_path="model.urdf",
        content_fingerprint="abc",
    )
    mock_cat.get_model.return_value = CloudModelRecord(
        id="so_arm100", display_name="SO-100", family="arm"
    )

    verifier = RuntimeVerifier(catalogue_provider=mock_cat)
    st = verifier.compute_sync_state("so_arm100")
    # Has local description files, version in cloud, but 0 cloud assets
    assert st.status == CloudSyncStatus.REMOTE_ASSET_MISSING
    assert "Remote Assets Missing" in st.display_label


def test_compute_sync_state_fully_synced():
    mock_cat = MagicMock()
    mock_cat.is_connected.return_value = True
    mock_cat.list_assets.return_value = [
        CloudAssetRecord(
            id="a1",
            version_id="franka_panda_v1.0.0",
            relative_path="model.urdf.xacro",
            asset_type="description",
            object_storage_key="models/franka_panda/v1.0.0/model.urdf.xacro",
            sha256="12345",
        )
    ]
    mock_cat.get_version.return_value = CloudVersionRecord(
        id="franka_panda_v1.0.0",
        robot_model_id="franka_panda",
        version="v1.0.0",
        description_format="URDF",
        entrypoint_path="model.urdf.xacro",
        content_fingerprint="abc",
    )
    mock_cat.get_model.return_value = CloudModelRecord(
        id="franka_panda", display_name="Franka Emika Panda", family="robotic arm"
    )

    verifier = RuntimeVerifier(catalogue_provider=mock_cat)
    st = verifier.compute_sync_state("franka_panda")
    assert st.status == CloudSyncStatus.SYNCED
    assert "Synced" in st.display_label
    assert st.remote_assets_count == 1
    assert st.local_assets_count > 0


def test_format_verification_table_rows_reflects_sync_state():
    offline_cat = OfflineCatalogueProvider()
    verifier = RuntimeVerifier(catalogue_provider=offline_cat)

    report_synced = verifier.verify_robot("rsk", sync_to_cloud=True)
    catalog = RobotModelCatalog()
    rows = catalog.format_verification_table_rows([report_synced])

    assert len(rows) == 1
    assert rows[0][0] == "Robot Soccer Kit (RSK)"
    assert rows[0][2] == "mujoco"
    assert rows[0][4] == "Ready"
    assert "Synced" in rows[0][5]
