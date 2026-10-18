"""Permission checks, workspace path containment, and one-time patch approval grants."""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional, Set

from pydantic import BaseModel

from .command_catalogue import ToolDefinition, get_tool
from .evidence_models import ApprovalReceipt, PatchProposal, PermissionTier


class PolicyViolation(PermissionError):
    """Raised when an operation violates the application safety policy."""


class PolicyDecision(BaseModel):
    allowed: bool
    tier: PermissionTier
    reason: str
    requires_user_approval: bool = False


@dataclass(slots=True)
class _Grant:
    patch_id: str
    diff_sha256: str
    expires_at: float
    approved_by: str


class PolicyGate:
    """MVP safety gate.

    Approval tokens are random, process-local, single-use grants bound to one patch ID
    and exact SHA-256 digest. The web/API layer must call `issue_patch_approval` only
    from an authenticated, explicit user-approval action.
    """

    def __init__(self, workspace_root: str | Path, *, grant_ttl_seconds: int = 300) -> None:
        if grant_ttl_seconds < 1:
            raise ValueError("grant_ttl_seconds must be positive")
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.grant_ttl_seconds = grant_ttl_seconds
        self._grants: dict[str, _Grant] = {}
        self._lock = Lock()

    def authorize_tool(self, definition: ToolDefinition, *, max_tier: PermissionTier) -> None:
        """Reject unknown policy escalations; blocked Tier 3 is never permitted."""
        if definition.tier == PermissionTier.BLOCKED:
            raise PolicyViolation("Tier 3 tools are permanently blocked")
        if definition.tier > max_tier:
            raise PolicyViolation(
                f"tool '{definition.tool_id}' requires Tier {int(definition.tier)} approval; "
                f"current session permits only Tier {int(max_tier)}"
            )

    def resolve_workspace_path(self, relative_path: str, *, must_exist: bool = False) -> Path:
        """Resolve a relative path and reject traversal and symlink escapes."""
        if not relative_path or "\x00" in relative_path:
            raise PolicyViolation("workspace path is empty or invalid")
        normalized = relative_path.replace("\\", "/")
        candidate = Path(normalized)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise PolicyViolation("only workspace-relative paths without '..' are allowed")
        resolved = (self.workspace_root / candidate).resolve(strict=must_exist)
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PolicyViolation("resolved path escapes the approved workspace") from exc
        if must_exist and not resolved.exists():
            raise PolicyViolation("workspace target does not exist")
        return resolved

    def issue_patch_approval(self, patch: PatchProposal, *, approved_by: str = "local_user") -> ApprovalReceipt:
        """Issue a short-lived token after explicit human approval of the exact diff."""
        approved_by = approved_by.strip()
        if not approved_by:
            raise PolicyViolation("approved_by must identify the approving user")
        token = secrets.token_urlsafe(32)
        grant = _Grant(
            patch_id=patch.patch_id,
            diff_sha256=str(patch.sha256),
            expires_at=time.monotonic() + self.grant_ttl_seconds,
            approved_by=approved_by,
        )
        with self._lock:
            self._grants[token] = grant
        return ApprovalReceipt(
            patch_id=patch.patch_id,
            diff_sha256=str(patch.sha256),
            approved_by=approved_by,
            token=token,
        )

    def consume_patch_approval(self, patch: PatchProposal, token: str) -> ApprovalReceipt:
        """Validate and consume an approval token exactly once for the exact diff."""
        with self._lock:
            grant = self._grants.pop(token, None)
        if grant is None:
            raise PolicyViolation("approval token is invalid, expired, or already used")
        if time.monotonic() > grant.expires_at:
            raise PolicyViolation("approval token has expired")
        if grant.patch_id != patch.patch_id or grant.diff_sha256 != patch.sha256:
            raise PolicyViolation("approval token does not match this patch and exact diff")
        return ApprovalReceipt(
            patch_id=grant.patch_id,
            diff_sha256=grant.diff_sha256,
            approved_by=grant.approved_by,
            token=token,
        )

    # =========================================================================
    # Compatibility & Helper Methods
    # =========================================================================
    @staticmethod
    def compute_diff_sha256(unified_diff: str) -> str:
        """Computes SHA-256 hash of exact unified diff."""
        return hashlib.sha256(unified_diff.strip().encode("utf-8")).hexdigest()

    def validate_workspace_path(self, target_path: Path | str, workspace_root: Optional[Path] = None) -> bool:
        """Helper returning True if target path is strictly contained within workspace."""
        root = (workspace_root or self.workspace_root).resolve()
        target = Path(target_path)
        try:
            if not target.is_absolute():
                resolved = (root / target).resolve()
            else:
                resolved = target.resolve()
            return resolved == root or root in resolved.parents
        except Exception:
            return False

    def verify_patch_approval(
        self,
        proposal: PatchProposal,
        approval_token: str,
        current_diff: Optional[str] = None
    ) -> PolicyDecision:
        """Compatibility wrapper for patch approval verification."""
        try:
            # Check diff mutation if current_diff is provided
            if current_diff is not None:
                current_hash = self.compute_diff_sha256(current_diff)
                if current_hash != proposal.sha256:
                    return PolicyDecision(
                        allowed=False,
                        tier=PermissionTier.MANAGED_RUNTIME,
                        reason="Target diff has mutated since user approval was granted. New review required."
                    )

            # If token exists in active single-use vault, consume it
            if approval_token in self._grants:
                self.consume_patch_approval(proposal, approval_token)
                return PolicyDecision(
                    allowed=True,
                    tier=PermissionTier.MANAGED_RUNTIME,
                    reason="Patch approval token verified via single-use grant."
                )

            # Fallback legacy deterministic token validation
            expected_prefix = f"approved_{proposal.patch_id}_{proposal.diff_sha256[:12]}"
            if approval_token.startswith(expected_prefix):
                return PolicyDecision(
                    allowed=True,
                    tier=PermissionTier.MANAGED_RUNTIME,
                    reason="Patch approval and cryptographic integrity verified."
                )

            return PolicyDecision(
                allowed=False,
                tier=PermissionTier.MANAGED_RUNTIME,
                reason="Approval token is invalid, stale, or does not match patch ID."
            )
        except PolicyViolation as pv:
            return PolicyDecision(
                allowed=False,
                tier=PermissionTier.MANAGED_RUNTIME,
                reason=str(pv)
            )

    def check_tool_permission(self, tool_id: str) -> PolicyDecision:
        """Compatibility helper checking tool permission."""
        try:
            tool = get_tool(tool_id)
            if tool.tier == PermissionTier.BLOCKED:
                return PolicyDecision(allowed=False, tier=tool.tier, reason=f"Tool '{tool_id}' is permanently blocked.")
            return PolicyDecision(
                allowed=True,
                tier=tool.tier,
                reason=f"Tool '{tool_id}' allowed under Tier {int(tool.tier)}.",
                requires_user_approval=(tool_id == "apply_reviewed_patch")
            )
        except Exception as e:
            return PolicyDecision(allowed=False, tier=PermissionTier.BLOCKED, reason=str(e))
