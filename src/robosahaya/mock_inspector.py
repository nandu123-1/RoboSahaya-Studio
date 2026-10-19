import json
from pathlib import Path
from typing import Optional, Dict
from robosahaya.config import settings
from robosahaya.base_inspector import BaseInspector
from robosahaya.evidence_models import (
    GraphSnapshot,
    TopicEndpointEvidence,
    ParamEvidence,
    TfEvidence,
    MessageSampleEvidence,
    EvidenceStatus
)

class MockInspector(BaseInspector):
    """Zero-dependency inspector that replays pre-recorded ROS 2 graph snapshots."""

    def __init__(self, scenario_name: Optional[str] = None):
        self.scenario_name = scenario_name or settings.ROBOSAHAYA_MOCK_SCENARIO
        self.fixtures_dir = settings.FIXTURES_DIR
        self._snapshot_data: Dict = {}
        self.load_scenario(self.scenario_name)

    def load_scenario(self, scenario_name: str):
        self.scenario_name = scenario_name
        fixture_path = self.fixtures_dir / f"{scenario_name}.json"
        if not fixture_path.exists():
            # Fallback to default fixture if specific one not found
            fallback = self.fixtures_dir / "qos_reliability_mismatch.json"
            if fallback.exists():
                fixture_path = fallback
            else:
                self._snapshot_data = self._create_in_memory_default()
                return

        with open(fixture_path, "r", encoding="utf-8") as f:
            self._snapshot_data = json.load(f)

    def _create_in_memory_default(self) -> Dict:
        return {
            "nodes": ["/robot_state_publisher", "/navigation_node"],
            "topics": {"/scan": ["sensor_msgs/msg/LaserScan"], "/cmd_vel": ["geometry_msgs/msg/Twist"]},
            "endpoints": {
                "/scan": {
                    "topic": "/scan",
                    "exists_in_graph": True,
                    "publishers": [{"node_name": "/rplidar_node", "qos": {"reliability": "BEST_EFFORT", "durability": "VOLATILE"}}],
                    "subscriptions": [{"node_name": "/navigation_node", "qos": {"reliability": "RELIABLE", "durability": "VOLATILE"}}]
                }
            },
            "parameters": {"/navigation_node": {"use_sim_time": True}},
            "tf_frames": ["base_link", "laser_frame"],
            "clock_active": True
        }

    def inspect_graph(self) -> GraphSnapshot:
        return GraphSnapshot(
            nodes=self._snapshot_data.get("nodes", []),
            topics=self._snapshot_data.get("topics", {}),
            endpoints={
                k: TopicEndpointEvidence(**v)
                for k, v in self._snapshot_data.get("endpoints", {}).items()
            },
            parameters=self._snapshot_data.get("parameters", {}),
            tf_frames=self._snapshot_data.get("tf_frames", []),
            clock_active=self._snapshot_data.get("clock_active", False)
        )

    def inspect_topic_endpoints(self, topic: str) -> TopicEndpointEvidence:
        endpoints_dict = self._snapshot_data.get("endpoints", {})
        if topic in endpoints_dict:
            return TopicEndpointEvidence(**endpoints_dict[topic])
        
        # Check if topic exists in graph
        topics_dict = self._snapshot_data.get("topics", {})
        exists = topic in topics_dict
        return TopicEndpointEvidence(topic=topic, exists_in_graph=exists, publishers=[], subscriptions=[])

    def inspect_parameter(self, node: str, param: str) -> ParamEvidence:
        params = self._snapshot_data.get("parameters", {}).get(node, {})
        if param in params:
            return ParamEvidence(node_name=node, param_name=param, value=params[param], status=EvidenceStatus.SUCCESS)
        return ParamEvidence(node_name=node, param_name=param, value=None, status=EvidenceStatus.UNAVAILABLE)

    def lookup_transform(self, source_frame: str, target_frame: str) -> TfEvidence:
        tf_data = self._snapshot_data.get("tf_lookups", {}).get(f"{source_frame}->{target_frame}")
        if tf_data:
            return TfEvidence(**tf_data)
        
        frames = self._snapshot_data.get("tf_frames", [])
        if source_frame in frames and target_frame in frames:
            return TfEvidence(
                source_frame=source_frame,
                target_frame=target_frame,
                transform_available=True,
                status=EvidenceStatus.SUCCESS,
                delay_sec=0.01
            )
        return TfEvidence(
            source_frame=source_frame,
            target_frame=target_frame,
            transform_available=False,
            status=EvidenceStatus.ERROR,
            error_message=f"Frame '{target_frame}' or '{source_frame}' does not exist in TF tree"
        )

    def sample_messages(self, topic: str, duration_sec: float = 1.0) -> MessageSampleEvidence:
        sample_info = self._snapshot_data.get("samples", {}).get(topic, {})
        count = sample_info.get("count", 0)
        snippet = sample_info.get("snippet", None)
        dur = max(0.01, float(duration_sec))
        frequency = float(sample_info.get("frequency_hz", count / dur if count > 0 else 0.0))
        payload_valid = sample_info.get("payload_valid", True if count > 0 else False)
        valid_samples = sample_info.get("valid_samples", count if payload_valid else 0)
        invalid_samples = sample_info.get("invalid_samples", 0 if payload_valid else count)
        return MessageSampleEvidence(
            topic=topic,
            messages_received=count,
            sample_window_sec=dur,
            frequency_hz=frequency,
            latest_message_snippet=snippet,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            payload_valid=payload_valid,
            validation_detail=sample_info.get("validation_detail", f"{valid_samples}/{count} valid samples observed." if count > 0 else "No messages received during observation window.")
        )
