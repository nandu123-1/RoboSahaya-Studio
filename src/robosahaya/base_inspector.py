from abc import ABC, abstractmethod
from robosahaya.evidence_models import (
    GraphSnapshot,
    TopicEndpointEvidence,
    ParamEvidence,
    TfEvidence,
    MessageSampleEvidence
)

class BaseInspector(ABC):
    """Abstract interface for inspecting ROS 2 graph and runtime states."""

    @abstractmethod
    def inspect_graph(self) -> GraphSnapshot:
        """Collect current nodes, topics, parameters, and frames."""
        pass

    @abstractmethod
    def inspect_topic_endpoints(self, topic: str) -> TopicEndpointEvidence:
        """Inspect publishers, subscribers, and QoS profiles for a given topic."""
        pass

    @abstractmethod
    def inspect_parameter(self, node: str, param: str) -> ParamEvidence:
        """Query parameter value from a target node."""
        pass

    @abstractmethod
    def lookup_transform(self, source_frame: str, target_frame: str) -> TfEvidence:
        """Perform a bounded TF transform lookup between source and target frames."""
        pass

    @abstractmethod
    def sample_messages(self, topic: str, duration_sec: float = 1.0) -> MessageSampleEvidence:
        """Sample messages arriving on a topic for a bounded duration window."""
        pass
