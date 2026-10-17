from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class EvidenceSource(str, Enum):
    GRAPH_TOPIC = "graph_topic"
    ENDPOINT_QOS = "endpoint_qos"
    TF_LOOKUP = "tf_lookup"
    PARAM_CLIENT = "param_client"
    MESSAGE_SAMPLE = "message_sample"
    USER_LOG = "user_log"

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
    latest_message_snippet: Optional[str] = None

class GraphSnapshot(BaseModel):
    nodes: List[str] = Field(default_factory=list)
    topics: Dict[str, List[str]] = Field(default_factory=dict)  # topic -> types
    endpoints: Dict[str, TopicEndpointEvidence] = Field(default_factory=dict)
    parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # node -> {param: val}
    tf_frames: List[str] = Field(default_factory=list)
    clock_active: bool = False

class EvidenceRecord(BaseModel):
    evidence_id: str = Field(..., pattern=r"^E\d+$", description="Unique evidence ID e.g. E1")
    source: EvidenceSource
    observed_at: str
    target: str
    result: Dict[str, Any]
    status: EvidenceStatus
    limitations: str

class Hypothesis(BaseModel):
    label: str
    evidence_ids: List[str]
    confidence: ConfidenceLabel
    caveat: str

class CommandItem(BaseModel):
    command_id: str
    raw_command: str
    purpose: str
    expected_observation: str

class DiagnosisResponse(BaseModel):
    case_id: str
    issue_summary: str
    plain_language_analogy: str = Field(..., description="Intuitive analogy for non-roboticist evaluators")
    observations: List[str] = Field(..., description="List of referenced evidence IDs e.g. ['E1', 'E2']")
    hypotheses: List[Hypothesis]
    confidence_label: ConfidenceLabel
    voice_script: str = Field(..., max_length=300, description="Concise text optimized for Timbre TTS without raw CLI commands")
    terminal_card: List[CommandItem]
    next_question: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
