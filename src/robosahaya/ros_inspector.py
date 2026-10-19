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
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message
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

        endpoints_summary = {
            topic: self.inspect_topic_endpoints(topic)
            for topic in names_and_types
        }
        frames = []
        try:
            frame_dump = self.tf_buffer.all_frames_as_string()
            import re
            frames = sorted(set(re.findall(r"(?:Frame|frame)\s+([A-Za-z0-9_./-]+)", frame_dump)))
        except Exception:
            pass
        clock = self.sample_messages("/clock", duration_sec=0.35) if "/clock" in names_and_types else None
        clock_active = bool(clock and clock.messages_received > 0)

        return GraphSnapshot(
            nodes=full_node_names,
            topics={k: list(v) for k, v in names_and_types.items()},
            endpoints=endpoints_summary,
            parameters={},
            tf_frames=frames,
            clock_active=clock_active,
            clock_advancing=(clock.clock_advancing if clock else False)
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
        names_and_types = dict(self.node.get_topic_names_and_types())
        try:
            if topic in names_and_types and names_and_types[topic]:
                message_type = get_message(names_and_types[topic][0])
            elif topic == "/scan":
                from sensor_msgs.msg import LaserScan
                message_type = LaserScan
            elif topic == "/camera/image_raw":
                from sensor_msgs.msg import Image
                message_type = Image
            elif topic == "/joint_states":
                from sensor_msgs.msg import JointState
                message_type = JointState
            else:
                return MessageSampleEvidence(topic=topic, messages_received=0, sample_window_sec=duration_sec,
                                             payload_valid=False, validation_detail="Topic is absent from the ROS graph.")
        except Exception as exc:
            return MessageSampleEvidence(topic=topic, sample_window_sec=duration_sec,
                                         sampling_error=f"Could not load ROS message type: {exc}")

        received = []
        lock = threading.Lock()
        def callback(message):
            valid, detail = self._validate_payload(topic, message)
            with lock:
                received.append((message, valid, detail, time.monotonic()))

        subscription = None
        observation_window = max(0.05, min(float(duration_sec), 10.0))
        try:
            subscription = self.node.create_subscription(
                message_type, topic, callback, qos_profile_sensor_data
            )
            if hasattr(self, "executor") and hasattr(self.executor, "wake"):
                self.executor.wake()

            start_t = time.monotonic()
            deadline = start_t + observation_window
            while time.monotonic() < deadline:
                time.sleep(0.05)
                if hasattr(self, "executor") and hasattr(self.executor, "wake"):
                    self.executor.wake()
                with lock:
                    if len(received) >= 10:
                        break
        except Exception as exc:
            return MessageSampleEvidence(topic=topic, sample_window_sec=duration_sec,
                                         sampling_error=f"Subscription sampling failed: {exc}")
        finally:
            if subscription is not None:
                self.node.destroy_subscription(subscription)
                if hasattr(self, "executor") and hasattr(self.executor, "wake"):
                    self.executor.wake()

        valid_count = sum(1 for _, valid, _, _ in received if valid is True)
        invalid_count = sum(1 for _, valid, _, _ in received if valid is False)
        elapsed = received[-1][3] - received[0][3] if len(received) > 1 else 0.0
        frequency = (len(received) - 1) / elapsed if elapsed > 0 else 0.0
        latest = received[-1] if received else None
        clock_values = []
        for message, _, _, _ in received:
            stamp = getattr(message, "clock", None)
            if stamp is not None:
                clock_values.append(int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))
        snippet = None
        if latest:
            message = latest[0]
            if hasattr(message, "ranges"):
                snippet = f"ranges={len(message.ranges)}, frame={message.header.frame_id!r}"
            elif hasattr(message, "data"):
                snippet = f"width={getattr(message, 'width', '?')}, height={getattr(message, 'height', '?')}, data_bytes={len(message.data)}"
            else:
                snippet = f"type={type(message).__name__}"
        return MessageSampleEvidence(
            topic=topic, messages_received=len(received), sample_window_sec=observation_window,
            frequency_hz=frequency, latest_message_snippet=snippet,
            valid_samples=valid_count, invalid_samples=invalid_count,
            payload_valid=(False if invalid_count else (True if received and valid_count == len(received) else None)),
            clock_advancing=(clock_values[-1] > clock_values[0] if len(clock_values) > 1 else None),
            validation_detail=latest[2] if latest else "No messages received during the observation window."
        )

    @staticmethod
    def _validate_payload(topic: str, message) -> tuple[Optional[bool], str]:
        if hasattr(message, "ranges"):
            import math
            ranges = list(message.ranges)
            finite = [value for value in ranges if math.isfinite(value)]
            if not ranges:
                return False, "LaserScan contains no range samples."
            if message.angle_increment <= 0 or message.range_min < 0 or message.range_max <= message.range_min:
                return False, "LaserScan angular or range metadata is invalid."
            invalid = [value for value in ranges if math.isnan(value) or
                       (math.isfinite(value) and not message.range_min <= value <= message.range_max)]
            if invalid:
                return False, f"{len(invalid)}/{len(ranges)} range values are NaN or outside declared limits."
            return True, f"{len(finite)}/{len(ranges)} finite ranges ({len(ranges) - len(finite)} no-return values); frame={message.header.frame_id!r}."
        if hasattr(message, "width") and hasattr(message, "height") and hasattr(message, "data"):
            if message.width <= 0 or message.height <= 0 or not message.data:
                return False, "Image dimensions or data payload are empty."
            if getattr(message, "step", 0) > 0 and len(message.data) < message.step * message.height:
                return False, "Image data is shorter than step multiplied by height."
            return True, f"Image payload is {message.width}x{message.height}, {len(message.data)} bytes."
        if hasattr(message, "name") and hasattr(message, "position"):
            if not message.name or len(message.position) == 0:
                return False, "JointState contains no joint names or positions."
            if len(message.name) != len(message.position):
                return False, f"JointState name count ({len(message.name)}) != position count ({len(message.position)})."
            return True, f"JointState with {len(message.name)} joints: {', '.join(message.name[:4])}."
        return None, f"Received {type(message).__name__}; semantic payload validation is not implemented for this type."

    def shutdown(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown()
        if hasattr(self, 'node'):
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
