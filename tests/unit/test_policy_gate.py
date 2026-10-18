import pytest
from pathlib import Path
from robosahaya.policy_gate import PolicyGate, PermissionTier
from robosahaya.evidence_models import PatchProposal

def test_policy_gate_tiers():
    gate = PolicyGate(Path("/tmp/robosahaya_ws"))

    # Tier 0
    t0 = gate.check_tool_permission("list_ros_topics")
    assert t0.allowed is True
    assert t0.tier == PermissionTier.TIER_0_READ_ONLY

    # Tier 1
    t1 = gate.check_tool_permission("validate_project_files")
    assert t1.allowed is True
    assert t1.tier == PermissionTier.TIER_1_WORKSPACE_BUILD

    # Tier 2
    t2 = gate.check_tool_permission("apply_reviewed_patch")
    assert t2.allowed is True
    assert t2.requires_user_approval is True

    # Tier 3 (Blocked)
    t3 = gate.check_tool_permission("run_shell")
    assert t3.allowed is False
    assert t3.tier == PermissionTier.TIER_3_BLOCKED

    # Unregistered tool
    t_unreg = gate.check_tool_permission("arbitrary_hacker_command")
    assert t_unreg.allowed is False

def test_policy_gate_path_traversal():
    root = Path("/tmp/robosahaya_sandbox")
    gate = PolicyGate(root)

    # Valid relative path
    assert gate.validate_workspace_path(Path("src/robot/model.urdf"), root) is True
    # Traversal escape
    assert gate.validate_workspace_path(Path("../../etc/shadow"), root) is False
    assert gate.validate_workspace_path(Path("/etc/passwd"), root) is False

def test_policy_gate_patch_approval_verification():
    root = Path("/tmp/robosahaya_sandbox")
    gate = PolicyGate(root)

    diff = """--- a/bridge.yaml\n+++ b/bridge.yaml\n@@ -1 +1 @@\n- topic: /scan_disabled\n+ topic: /scan\n"""
    diff_hash = PolicyGate.compute_diff_sha256(diff)

    proposal = PatchProposal(
        patch_id="patch_001",
        target_file="config/bridge.yaml",
        unified_diff=diff,
        diff_sha256=diff_hash,
        explanation="Enable /scan bridge mapping",
        requires_approval=True,
        verification_test_id="test_laser_scan_payload",
        backup_required=True
    )

    token = f"approved_{proposal.patch_id}_{diff_hash[:12]}"

    # Matching approval
    dec = gate.verify_patch_approval(proposal, token, diff)
    assert dec.allowed is True

    # Mutated diff
    mutated_diff = diff.replace("/scan", "/scan_wrong")
    dec_mut = gate.verify_patch_approval(proposal, token, mutated_diff)
    assert dec_mut.allowed is False
    assert "mutated" in dec_mut.reason

    # Invalid token
    dec_bad_tok = gate.verify_patch_approval(proposal, "invalid_token", diff)
    assert dec_bad_tok.allowed is False
    assert "token is invalid" in dec_bad_tok.reason
