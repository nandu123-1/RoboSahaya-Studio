from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from robosahaya.command_catalogue import (
    BLOCKED_TOOL_IDS,
    ToolCatalogueError,
    get_tool,
    validate_tool_request,
)
from robosahaya.evidence_models import (
    ConfidenceLabel,
    EvidenceRecord,
    EvidenceSource,
    EvidenceStatus,
    PatchProposal,
    PermissionTier,
    TaskEvent,
    TaskStage,
    ToolRequest,
)
from robosahaya.policy_gate import PolicyGate, PolicyViolation


def valid_patch(**overrides):
    fields = {
        "patch_id": "patch_001",
        "target_file": "simulation/differential_drive/bridge.yaml",
        "unified_diff": "--- a/bridge.yaml\n+++ b/bridge.yaml\n@@\n-old: true\n+new: true\n",
        "explanation": "Restore the registered LiDAR bridge mapping.",
        "verification_test_id": "lidar_stream_valid",
    }
    fields.update(overrides)
    return PatchProposal(**fields)


def test_evidence_record_defaults_timestamp_and_validates_id():
    item = EvidenceRecord(
        evidence_id="E1",
        source=EvidenceSource.MESSAGE_SAMPLE,
        target="/scan",
        result={"received": 3},
        status=EvidenceStatus.SUCCESS,
    )
    assert item.evidence_id == "E1"
    assert item.observed_at.tzinfo is not None

    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="evidence-1",
            source=EvidenceSource.MESSAGE_SAMPLE,
            target="/scan",
            status=EvidenceStatus.SUCCESS,
        )


def test_patch_hash_is_computed_and_mismatch_rejected():
    patch = valid_patch()
    assert patch.sha256 == hashlib.sha256(patch.unified_diff.encode()).hexdigest()
    with pytest.raises(ValidationError, match="sha256 does not match"):
        valid_patch(sha256="0" * 64)


@pytest.mark.parametrize("path", ["../etc/passwd", "/tmp/escape", "simulation/../../outside", "C:\\Windows\\win.ini"])
def test_patch_target_rejects_paths_outside_workspace(path):
    with pytest.raises(ValidationError):
        valid_patch(target_file=path)


def test_tool_catalogue_rejects_unknown_and_arbitrary_shell_tools():
    assert "run_shell" in BLOCKED_TOOL_IDS
    with pytest.raises(ToolCatalogueError, match="permanently blocked"):
        get_tool("run_shell")
    with pytest.raises(ToolCatalogueError, match="unknown tool_id"):
        get_tool("invented_tool")


def test_topic_tool_accepts_only_a_bounded_topic_name():
    validate_tool_request(ToolRequest(tool_id="get_topic_info", arguments={"topic": "/scan"}))
    with pytest.raises(ToolCatalogueError):
        validate_tool_request(ToolRequest(tool_id="get_topic_info", arguments={"topic": "/scan; rm -rf /"}))
    with pytest.raises(ToolCatalogueError, match="unsupported arguments"):
        validate_tool_request(ToolRequest(tool_id="get_topic_info", arguments={"topic": "/scan", "shell": "whoami"}))


def test_sample_duration_is_bounded():
    with pytest.raises(ToolCatalogueError, match="between 0.1 and 3.0"):
        validate_tool_request(ToolRequest(tool_id="sample_topic_messages", arguments={"topic": "/scan", "duration_sec": 30}))


def test_policy_gate_respects_permission_tier(tmp_path):
    gate = PolicyGate(tmp_path)
    read_tool = get_tool("list_ros_nodes")
    runtime_tool = get_tool("launch_simulation_profile")
    gate.authorize_tool(read_tool, max_tier=PermissionTier.READ_ONLY)
    with pytest.raises(PolicyViolation, match="requires Tier 2"):
        gate.authorize_tool(runtime_tool, max_tier=PermissionTier.WORKSPACE)


def test_workspace_path_resolution_rejects_traversal(tmp_path):
    gate = PolicyGate(tmp_path)
    good = gate.resolve_workspace_path("simulation/world.sdf")
    assert good == (tmp_path.resolve() / "simulation/world.sdf")
    with pytest.raises(PolicyViolation):
        gate.resolve_workspace_path("../../outside")


def test_patch_approval_is_bound_to_hash_and_single_use(tmp_path):
    gate = PolicyGate(tmp_path)
    patch = valid_patch()
    receipt = gate.issue_patch_approval(patch, approved_by="local-user")
    consumed = gate.consume_patch_approval(patch, receipt.token)
    assert consumed.diff_sha256 == patch.sha256
    assert consumed.approved_by == "local-user"
    with pytest.raises(PolicyViolation, match="invalid, expired, or already used"):
        gate.consume_patch_approval(patch, receipt.token)


def test_patch_approval_cannot_be_reused_for_changed_diff(tmp_path):
    gate = PolicyGate(tmp_path)
    original = valid_patch()
    receipt = gate.issue_patch_approval(original, approved_by="local-user")
    changed = valid_patch(unified_diff="--- a\n+++ b\n@@\n-left\n+right\n")
    with pytest.raises(PolicyViolation, match="does not match"):
        gate.consume_patch_approval(changed, receipt.token)


def test_task_event_contract():
    event = TaskEvent(
        task_id="task_001",
        event_id="event_001",
        stage=TaskStage.TESTING,
        user_facing_message="Checking LiDAR message payloads.",
        speak_aloud=True,
    )
    assert event.speak_aloud is True
    assert event.stage == TaskStage.TESTING
