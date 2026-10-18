import pytest
import shutil
from pathlib import Path
from robosahaya.workspace_manager import WorkspaceManager

@pytest.fixture
def temp_ws_manager(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    templates_dir = repo_root / "simulation" / "templates"
    return WorkspaceManager(base_workspace_root=tmp_path, templates_dir=templates_dir)

def test_assemble_and_validate_template(temp_ws_manager):
    task_id = "test_task_01"
    ws = temp_ws_manager.assemble_simulation_template(task_id, "differential_drive")
    assert ws.exists()
    assert (ws / "model.urdf").exists()
    assert (ws / "bridge.yaml").exists()
    assert (ws / "manifest.json").exists()

    # Validate syntax
    val = temp_ws_manager.validate_workspace_files(ws)
    assert val["valid"] is True
    assert val["checks"]["urdf_xml"] == "VALID"
    assert val["checks"]["bridge_yaml"] == "VALID"

def test_inject_fault_and_apply_patch(temp_ws_manager):
    task_id = "test_fault_task"
    ws = temp_ws_manager.assemble_simulation_template(task_id, "differential_drive")

    # Inject fault
    proposal = temp_ws_manager.inject_demo_fault(ws, "missing_bridge_scan")
    assert proposal.patch_id == "patch_enable_laser_bridge"
    assert "scan_disabled" in (ws / "bridge.yaml").read_text()

    # Apply without token -> should fail
    fail_res = temp_ws_manager.apply_reviewed_patch(ws, proposal, "wrong_token")
    assert fail_res["success"] is False

    # Apply with correct token
    valid_token = f"approved_{proposal.patch_id}_{proposal.diff_sha256[:12]}"
    succ_res = temp_ws_manager.apply_reviewed_patch(ws, proposal, valid_token)
    assert succ_res["success"] is True
    assert (ws / "bridge.yaml.bak").exists()
    assert 'ros_topic_name: "/scan"' in (ws / "bridge.yaml").read_text()
