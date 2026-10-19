"""Strict Pydantic v2 contracts shared by the UI, orchestrator, inspectors and tools.

These models validate data; they do not themselves inspect ROS 2 or execute tools.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)


class TaskStage(str, Enum):
    RECEIVED = "received"
    TRANSCRIBED = "transcribed"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    PREPARING = "preparing"
    BUILDING = "building"
    LAUNCHING = "launching"
    TESTING = "testing"
    DIAGNOSING = "diagnosing"
    PATCH_PROPOSED = "patch_proposed"
    PATCH_APPROVED = "patch_approved"
    APPLYING_PATCH = "applying_patch"
    RE_TESTING = "re_testing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    PARTIAL_SUCCESS = "partial_success"


class EvidenceSource(str, Enum):
    GRAPH_TOPIC = "graph_topic"
    GRAPH_NODE = "graph_node"
    ENDPOINT_QOS = "endpoint_qos"
    TF_LOOKUP = "tf_lookup"
    PARAM_CLIENT = "param_client"
    MESSAGE_SAMPLE = "message_sample"
    BUILD_LOG = "build_log"
    SIMULATOR_STATUS = "simulator_status"
    USER_LOG = "user_log"
    FIXTURE_REPLAY = "fixture_replay"


class EvidenceStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ConfidenceLabel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class PermissionTier(int, Enum):
    READ_ONLY = 0
    WORKSPACE = 1
    MANAGED_RUNTIME = 2
    BLOCKED = 3

    # Backwards-compatible aliases
    TIER_0_READ_ONLY = 0
    TIER_1_WORKSPACE_BUILD = 1
    TIER_2_MANAGED_SIM_PATCH = 2
    TIER_3_BLOCKED = 3


class QoSPolicy(BaseModel):
    reliability: str = "SYSTEM_DEFAULT"  # RELIABLE, BEST_EFFORT, UNKNOWN
    durability: str = "SYSTEM_DEFAULT"   # VOLATILE, TRANSIENT_LOCAL, UNKNOWN
    history: Optional[str] = None
    depth: Optional[int] = None


class EndpointInfo(BaseModel):
    node_name: str
    node_namespace: str = "/"
    topic_type: Optional[str] = None
    qos: QoSPolicy = Field(default_factory=QoSPolicy)


class TopicEndpointEvidence(BaseModel):
    topic: str
    exists_in_graph: bool
    publishers: List[EndpointInfo] = Field(default_factory=list)
    subscriptions: List[EndpointInfo] = Field(default_factory=list)


class TfEvidence(BaseModel):
    source_frame: str
    target_frame: str
    transform_available: bool
    status: EvidenceStatus
    error_message: Optional[str] = None
    delay_sec: Optional[float] = None


class ParamEvidence(BaseModel):
    node_name: str
    param_name: str
    value: Any = None
    status: EvidenceStatus = EvidenceStatus.SUCCESS


class MessageSampleEvidence(BaseModel):
    topic: str
    messages_received: int = 0
    sample_window_sec: float = 1.0
    frequency_hz: float = 0.0
    latest_message_snippet: Optional[str] = None
    valid_samples: int = 0
    invalid_samples: int = 0
    payload_valid: Optional[bool] = None
    clock_advancing: Optional[bool] = None
    validation_detail: Optional[str] = None
    sampling_error: Optional[str] = None


class GraphSnapshot(BaseModel):
    nodes: List[str] = Field(default_factory=list)
    topics: Dict[str, List[str]] = Field(default_factory=dict)
    endpoints: Dict[str, TopicEndpointEvidence] = Field(default_factory=dict)
    parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    tf_frames: List[str] = Field(default_factory=list)
    clock_active: bool = False
    clock_advancing: Optional[bool] = None


class EvidenceRecord(StrictModel):
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$", examples=["E1"])
    source: EvidenceSource
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target: str = Field(min_length=1, max_length=256)
    result: dict[str, Any] = Field(default_factory=dict)
    status: EvidenceStatus
    limitations: str = Field(default="", max_length=1000)
    source_mode: Literal["live", "replay", "user_provided"] = "live"


class Hypothesis(StrictModel):
    label: str = Field(min_length=1, max_length=240)
    evidence_ids: list[str] = Field(default_factory=list, max_length=30)
    confidence: ConfidenceLabel = ConfidenceLabel.INSUFFICIENT
    caveat: str = Field(default="", max_length=1000)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_valid(cls, value: list[str]) -> list[str]:
        if any(not re.fullmatch(r"E[1-9][0-9]*", item) for item in value):
            raise ValueError("evidence_ids must contain IDs such as E1 or E12")
        if len(value) != len(set(value)):
            raise ValueError("evidence_ids must not contain duplicates")
        return value


class CommandItem(StrictModel):
    """A UI-visible catalogue reference, never a shell command to evaluate."""

    command_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    purpose: str = Field(min_length=1, max_length=300)
    expected_observation: str = Field(min_length=1, max_length=500)
    display_hint: str = Field(default="", max_length=500)
    raw_command: str = ""


class PatchProposal(StrictModel):
    # Unified diff whitespace and its final newline are digest-bound file content.
    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=False)

    patch_id: str = Field(pattern=r"^[A-Za-z0-9_-]{4,80}$")
    target_file: str = Field(min_length=1, max_length=300)
    unified_diff: str = Field(min_length=1, max_length=100_000)
    explanation: str = Field(min_length=1, max_length=2000)
    verification_test_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,79}$")
    requires_approval: Literal[True] = True
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    base_file_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    backup_required: bool = True

    @property
    def diff_sha256(self) -> str:
        return str(self.sha256)

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "diff_sha256" in data and "sha256" not in data:
                data["sha256"] = data.pop("diff_sha256")
            elif "diff_sha256" in data:
                data.pop("diff_sha256")
        return data

    @field_validator("target_file")
    @classmethod
    def target_is_relative_workspace_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or normalized.startswith("/"):
            raise ValueError("target_file must be a relative workspace path")
        if any(part in {"..", ""} for part in path.parts):
            raise ValueError("target_file cannot contain traversal or empty path parts")
        if path.parts and ":" in path.parts[0]:
            raise ValueError("drive-qualified paths are not allowed")
        if "\x00" in value:
            raise ValueError("NUL bytes are not allowed in target_file")
        return path.as_posix()

    @model_validator(mode="after")
    def bind_diff_digest(self) -> "PatchProposal":
        computed = hashlib.sha256(self.unified_diff.encode("utf-8")).hexdigest()
        if self.sha256 is not None and self.sha256 != computed:
            raise ValueError("sha256 does not match unified_diff")
        object.__setattr__(self, "sha256", computed)
        return self


class DiagnosisResponse(StrictModel):
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,79}$")
    issue_summary: str = Field(min_length=1, max_length=1000)
    plain_language_analogy: str = Field(min_length=1, max_length=1000)
    observations: list[str] = Field(default_factory=list, max_length=100)
    hypotheses: list[Hypothesis] = Field(default_factory=list, max_length=20)
    confidence_label: ConfidenceLabel = ConfidenceLabel.INSUFFICIENT
    voice_script: str = Field(min_length=1, max_length=300)
    terminal_card: list[CommandItem] = Field(default_factory=list, max_length=30)
    proposed_patch: PatchProposal | None = None
    next_question: str | None = Field(default=None, max_length=500)
    limitations: list[str] = Field(default_factory=list, max_length=30)
    evidence_records: list[EvidenceRecord] = Field(default_factory=list, max_length=100)
    reasoning_source: Literal["rules", "evon", "replay"] = "rules"

    @field_validator("observations")
    @classmethod
    def observations_are_evidence_ids(cls, value: list[str]) -> list[str]:
        if any(not re.fullmatch(r"E[1-9][0-9]*", item) for item in value):
            raise ValueError("observations must reference evidence IDs such as E1")
        if len(value) != len(set(value)):
            raise ValueError("observations must not contain duplicate evidence IDs")
        return value


class TaskEvent(StrictModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9_-]{4,80}$")
    event_id: str = Field(pattern=r"^[A-Za-z0-9_-]{4,100}$")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage: TaskStage
    severity: Literal["debug", "info", "warning", "error", "critical", "success"] = "info"
    user_facing_message: str = Field(min_length=1, max_length=1000)
    speak_aloud: bool = False
    evidence_reference: str | None = Field(default=None, pattern=r"^E[1-9][0-9]*$")
    tool_name: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{1,63}$")
    status: Literal["started", "running", "succeeded", "failed", "skipped", "blocked", "in_progress", "completed", "cancelled"] = "running"


class ToolRequest(StrictModel):
    """A structured request to a registered tool. No raw shell string field exists."""

    tool_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    arguments: dict[str, Any] = Field(default_factory=dict)
    task_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{4,80}$")


class ToolResult(StrictModel):
    tool_id: str
    success: bool
    status: Literal["success", "partial", "timeout", "unavailable", "error", "blocked"]
    data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    message: str = Field(min_length=1, max_length=1000)
    duration_ms: float | None = Field(default=None, ge=0)


class ApprovalReceipt(StrictModel):
    patch_id: str
    target_file: str = Field(min_length=1, max_length=300)
    diff_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    base_file_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    approved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: str = Field(min_length=1, max_length=128)
    token: str = Field(min_length=20, max_length=256)


class TestResult(StrictModel):
    __test__ = False
    test_id: str
    passed: bool
    measured_values: Dict[str, Any] = Field(default_factory=dict)
    expected_criteria: Dict[str, Any] = Field(default_factory=dict)
    log_excerpt: str = ""
    source_mode: Literal["live", "replay"] = "replay"
    status: Literal["passed", "failed", "blocked", "not_run"] = "not_run"


class SensorPayloadValidation(StrictModel):
    topic: str
    message_type: str
    total_samples: int
    finite_range_count: int
    nan_or_inf_count: int
    is_valid: bool
    frequency_hz: float
    error_reason: Optional[str] = None


def compute_plan_digest(
    user_goal: str,
    robot_template: str,
    world_template: str,
    requested_capabilities: List[str],
    verification_tests: List[str],
    tools_to_run: List[str],
    simulator_adapter: str = "",
    model_format: str = ""
) -> str:
    payload = f"{user_goal.strip()}|{robot_template.strip()}|{world_template.strip()}|{sorted(requested_capabilities)}|{sorted(verification_tests)}|{sorted(tools_to_run)}|{simulator_adapter.strip()}|{model_format.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TaskPlan(StrictModel):
    task_id: str
    user_goal: str
    robot_template: str
    world_template: str
    requested_capabilities: List[str] = Field(default_factory=list)
    unsupported_capabilities: List[str] = Field(default_factory=list)
    is_supported: bool = True
    rejection_reason: Optional[str] = None
    ordered_stages: List[TaskStage] = Field(default_factory=list)
    tools_to_run: List[str] = Field(default_factory=list)
    permission_tiers: Dict[str, PermissionTier] = Field(default_factory=dict)
    verification_tests: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    estimated_side_effects: List[str] = Field(default_factory=list)
    plan_digest: str = Field(default="", pattern=r"^([a-f0-9]{64})?$")
    robot_selection_reason: str = ""
    world_selection_reason: str = ""
    selection_source: str = "rule_based"
    is_auto_selected: bool = True
    is_manual_override: bool = False
    manual_overrides: Dict[str, str] = Field(default_factory=dict)
    sensors_selected: List[str] = Field(default_factory=list)
    ros_topics: List[str] = Field(default_factory=list)
    expected_behavior: str = ""
    measurable_criteria: Dict[str, Any] = Field(default_factory=dict)
    mcp_verified: bool = False
    mcp_verified_facts: List[str] = Field(default_factory=list)
    is_stale: bool = False
    simulator_adapter: str = "gazebo_harmonic"
    model_format: str = "URDF"
    milestones: Dict[str, Any] = Field(default_factory=dict)


