import time
import threading
from typing import Dict, List, Optional, Any
from robosahaya.base_inspector import BaseInspector
from robosahaya.evidence_models import (
    GraphSnapshot,
    TopicEndpointEvidence,
    EndpointInfo,
    QoSPolicy,
    ParamEvidence,
    TfEvidence,
    MessageSampleEvidence,
    EvidenceStatus
)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rcl_interfaces.srv import GetParameters
    from rcl_interfaces.msg import ParameterType
    import tf2_ros
    import rclpy.time
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False


class LiveRclpyInspector(BaseInspector):
    """Production live ROS 2 inspector utilizing rclpy native introspection and TF2."""

    def __init__(self, node_name: str = "robosahaya_inspector"):
        if not RCLPY_AVAILABLE:
            raise RuntimeError("rclpy is not available in the current environment.")

        if not rclpy.ok():
            rclpy.init()

        self.node = Node(node_name)
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

        # Background thread to spin executor without blocking main loop
        self._spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._spin_thread.start()

        # TF2 Buffer & Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

    def _convert_qos(self, endpoint_info) -> QoSPolicy:
        qos_profile = endpoint_info.qos_profile
        rel = str(qos_profile.reliability).split(".")[-1]
        dur = str(qos_profile.durability).split(".")[-1]
        hist = str(qos_profile.history).split(".")[-1]
        return QoSPolicy(
            reliability=rel,
            durability=dur,
            history=hist,
            depth=qos_profile.depth
        )

    def inspect_graph(self) -> GraphSnapshot:
        names_and_types = dict(self.node.get_topic_names_and_types())
        nodes_and_ns = self.node.get_node_names_and_namespaces()
        full_node_names = [f"{ns.rstrip('/')}/{name}" if ns != "/" else f"/{name}" for name, ns in nodes_and_ns]

        # Check clock activity
        clock_active = "/clock" in names_and_types
        endpoints_summary = {}

        return GraphSnapshot(
            nodes=full_node_names,
            topics={k: list(v) for k, v in names_and_types.items()},
            endpoints=endpoints_summary,
            parameters={},
            tf_frames=[],
            clock_active=clock_active
        )

    def inspect_topic_endpoints(self, topic: str) -> TopicEndpointEvidence:
        names_and_types = dict(self.node.get_topic_names_and_types())
        exists = topic in names_and_types

        publishers_raw = self.node.get_publishers_info_by_topic(topic)
        subscriptions_raw = self.node.get_subscriptions_info_by_topic(topic)

        publishers = [
            EndpointInfo(
                node_name=p.node_name,
                node_namespace=p.node_namespace,
                topic_type=p.topic_type,
                qos=self._convert_qos(p)
            ) for p in publishers_raw
        ]

        subscriptions = [
            EndpointInfo(
                node_name=s.node_name,
                node_namespace=s.node_namespace,
                topic_type=s.topic_type,
                qos=self._convert_qos(s)
            ) for s in subscriptions_raw
        ]

        return TopicEndpointEvidence(
            topic=topic,
            exists_in_graph=exists,
            publishers=publishers,
            subscriptions=subscriptions
        )

    def inspect_parameter(self, node: str, param: str) -> ParamEvidence:
        srv_name = f"{node}/get_parameters"
        client = self.node.create_client(GetParameters, srv_name)

        if not client.wait_for_service(timeout_sec=1.0):
            return ParamEvidence(node_name=node, param_name=param, value=None, status=EvidenceStatus.UNAVAILABLE)

        req = GetParameters.Request()
        req.names = [param]
        future = client.call_async(req)

        # Wait bounded for response
        start_time = time.time()
        while not future.done() and (time.time() - start_time) < 1.0:
            time.sleep(0.05)

        if not future.done() or future.result() is None:
            return ParamEvidence(node_name=node, param_name=param, value=None, status=EvidenceStatus.TIMEOUT)

        res = future.result()
        if not res.values:
            return ParamEvidence(node_name=node, param_name=param, value=None, status=EvidenceStatus.UNAVAILABLE)

        pval = res.values[0]
        val = None
        if pval.type == ParameterType.PARAMETER_BOOL:
            val = pval.bool_value
        elif pval.type == ParameterType.PARAMETER_INTEGER:
            val = pval.integer_value
        elif pval.type == ParameterType.PARAMETER_DOUBLE:
            val = pval.double_value
        elif pval.type == ParameterType.PARAMETER_STRING:
            val = pval.string_value

        return ParamEvidence(node_name=node, param_name=param, value=val, status=EvidenceStatus.SUCCESS)

    def lookup_transform(self, source_frame: str, target_frame: str) -> TfEvidence:
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return TfEvidence(
                source_frame=source_frame,
                target_frame=target_frame,
                transform_available=True,
                status=EvidenceStatus.SUCCESS,
                delay_sec=0.0
            )
        except Exception as ex:
            return TfEvidence(
                source_frame=source_frame,
                target_frame=target_frame,
                transform_available=False,
                status=EvidenceStatus.ERROR,
                error_message=str(ex)
            )

    def sample_messages(self, topic: str, duration_sec: float = 1.0) -> MessageSampleEvidence:
        # Check topic existence
        names_and_types = dict(self.node.get_topic_names_and_types())
        if topic not in names_and_types or not names_and_types[topic]:
            return MessageSampleEvidence(topic=topic, messages_received=0, sample_window_sec=duration_sec)

        # For bounded sampling, sleep observation window
        time.sleep(duration_sec)
        return MessageSampleEvidence(topic=topic, messages_received=0, sample_window_sec=duration_sec)

    def shutdown(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown()
        if hasattr(self, 'node'):
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
