import datetime
from typing import List, Dict, Tuple, Optional
from robosahaya.base_inspector import BaseInspector
from robosahaya.command_catalogue import CommandCatalogue
from robosahaya.evidence_models import (
    DiagnosisResponse,
    EvidenceRecord,
    EvidenceSource,
    EvidenceStatus,
    Hypothesis,
    ConfidenceLabel,
    CommandItem
)

class DiagnosticEngine:
    """Deterministic rule-based diagnostic engine for ROS 2 systems."""

    def __init__(self, inspector: BaseInspector):
        self.inspector = inspector

    def _now_iso(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def analyze(self, target_entity: str, entity_type: str = "topic", intent: str = "general_debug") -> DiagnosisResponse:
        evidence_list: List[EvidenceRecord] = []
        now = self._now_iso()

        # Route according to entity type or intent
        if intent == "sim_time_check" or target_entity == "use_sim_time":
            return self._diagnose_sim_time(evidence_list, now)
        elif entity_type == "frame" or intent == "tf_check":
            return self._diagnose_tf(target_entity, evidence_list, now)
        elif entity_type == "node":
            return self._diagnose_node(target_entity, evidence_list, now)
        else:
            # Default to topic inspection (covers Case A and Case D)
            return self._diagnose_topic(target_entity, evidence_list, now)

    def _diagnose_topic(self, topic: str, evidence: List[EvidenceRecord], now: str) -> DiagnosisResponse:
        endpoint_data = self.inspector.inspect_topic_endpoints(topic)

        # Evidence E1: Topic presence in graph
        evidence.append(EvidenceRecord(
            evidence_id="E1",
            source=EvidenceSource.GRAPH_TOPIC,
            observed_at=now,
            target=topic,
            result={"exists_in_graph": endpoint_data.exists_in_graph},
            status=EvidenceStatus.SUCCESS,
            limitations="Topic existence does not confirm active message throughput."
        ))

        # Check if topic is completely missing from graph
        if not endpoint_data.exists_in_graph:
            return DiagnosisResponse(
                case_id="case_a_topic_absent",
                issue_summary=f"Topic '{topic}' is not registered anywhere in the ROS 2 graph",
                plain_language_analogy="Like trying to tune into a radio station that has no assigned frequency or transmitter.",
                observations=["E1"],
                hypotheses=[
                    Hypothesis(
                        label="Expected publisher node is not running, or publishing under a remapped namespace",
                        evidence_ids=["E1"],
                        confidence=ConfidenceLabel.HIGH,
                        caveat="DDS network domain mismatch (ROS_DOMAIN_ID) could also prevent graph discovery."
                    )
                ],
                confidence_label=ConfidenceLabel.HIGH,
                voice_script=f"The topic {topic} is completely absent from the graph. Check if your sensor driver node was launched.",
                terminal_card=[
                    CommandCatalogue.get_command("list_topics"),
                    CommandCatalogue.get_command("list_nodes")
                ],
                next_question=f"Which node is supposed to publish to {topic}?",
                limitations=["Graph snapshot cannot distinguish between an unlaunched node and a domain ID mismatch."]
            )

        # Evidence E2: Publisher endpoints count
        pub_count = len(endpoint_data.publishers)
        sub_count = len(endpoint_data.subscriptions)
        evidence.append(EvidenceRecord(
            evidence_id="E2",
            source=EvidenceSource.ENDPOINT_QOS,
            observed_at=now,
            target=topic,
            result={"publisher_count": pub_count, "subscription_count": sub_count},
            status=EvidenceStatus.SUCCESS,
            limitations="Endpoint count reflects discovery state at the moment of inspection."
        ))

        # Check for Case A.2: Topic exists, but 0 publishers
        if pub_count == 0:
            return DiagnosisResponse(
                case_id="case_a_missing_publisher",
                issue_summary=f"Topic '{topic}' exists with {sub_count} subscriber(s), but ZERO publishers are active",
                plain_language_analogy="Like a crowd of listeners waiting in a hall with microphones on, but no speaker has taken the stage.",
                observations=["E1", "E2"],
                hypotheses=[
                    Hypothesis(
                        label="Sensor driver or publisher node has not started or failed to initialize",
                        evidence_ids=["E2"],
                        confidence=ConfidenceLabel.HIGH,
                        caveat="The publisher might be publishing to an alternate topic name due to missing remapping."
                    )
                ],
                confidence_label=ConfidenceLabel.HIGH,
                voice_script=f"The topic {topic} is registered, but no publisher endpoint was observed. Let's check active nodes.",
                terminal_card=[
                    CommandCatalogue.get_command("list_nodes"),
                    CommandCatalogue.get_command("topic_info", topic=topic)
                ],
                next_question="Did your sensor launch file start without error?",
                limitations=["A graph snapshot cannot prove whether the driver crashed or just never launched."]
            )

        # Check for Case D: QoS Incompatibility
        # Inspect publisher and subscriber QoS policies
        for pub in endpoint_data.publishers:
            for sub in endpoint_data.subscriptions:
                pub_rel = pub.qos.reliability
                sub_rel = sub.qos.reliability

                # Incompatibility: Publisher BEST_EFFORT, Subscriber RELIABLE
                if pub_rel == "BEST_EFFORT" and sub_rel == "RELIABLE":
                    evidence.append(EvidenceRecord(
                        evidence_id="E3",
                        source=EvidenceSource.ENDPOINT_QOS,
                        observed_at=now,
                        target=topic,
                        result={"publisher_reliability": pub_rel, "subscriber_reliability": sub_rel},
                        status=EvidenceStatus.SUCCESS,
                        limitations="QoS incompatibility silently drops messages without generating runtime exceptions."
                    ))
                    return DiagnosisResponse(
                        case_id="case_d_qos_reliability_mismatch",
                        issue_summary=f"Critical QoS Incompatibility on '{topic}': Publisher is {pub_rel}, Subscriber requires {sub_rel}",
                        plain_language_analogy="Like two walkie-talkies on the same frequency where one broadcasts in AM and the other listens in FM—they can never hear each other.",
                        observations=["E1", "E2", "E3"],
                        hypotheses=[
                            Hypothesis(
                                label="Subscriber drops all messages silently because its requested reliability exceeds offered reliability",
                                evidence_ids=["E3"],
                                confidence=ConfidenceLabel.HIGH,
                                caveat="Publisher is sending valid packets over the wire, but the middleware discards them at the receiver."
                            )
                        ],
                        confidence_label=ConfidenceLabel.HIGH,
                        voice_script=f"QoS mismatch detected on {topic}. The publisher offers best effort, but the subscriber demands reliable. Switch your subscription to best effort.",
                        terminal_card=[
                            CommandCatalogue.get_command("topic_info", topic=topic),
                            CommandCatalogue.get_command("topic_hz", topic=topic)
                        ],
                        next_question="Would you like the exact C++ or Python QoS profile snippet to fix your subscriber?",
                        limitations=["Fix requires editing the subscriber's QoS profile in code or launch parameters."]
                    )

        # If publishers and subscribers exist with compatible QoS, check message arrival
        sample = self.inspector.sample_messages(topic, duration_sec=1.0)
        evidence.append(EvidenceRecord(
            evidence_id="E4",
            source=EvidenceSource.MESSAGE_SAMPLE,
            observed_at=now,
            target=topic,
            result={"messages_received": sample.messages_received, "window_sec": sample.sample_window_sec},
            status=EvidenceStatus.SUCCESS,
            limitations=f"Sample observed for only {sample.sample_window_sec} seconds."
        ))

        if sample.messages_received == 0:
            return DiagnosisResponse(
                case_id="case_a_no_messages_received",
                issue_summary=f"Publisher endpoints exist on '{topic}', but 0 messages were received during observation window",
                plain_language_analogy="Like a television channel showing a test pattern with no broadcast data coming through the wire.",
                observations=["E1", "E2", "E4"],
                hypotheses=[
                    Hypothesis(
                        label="Publisher is alive but idle, waiting for hardware triggers or publishing at <0.5 Hz",
                        evidence_ids=["E4"],
                        confidence=ConfidenceLabel.MEDIUM,
                        caveat="Hardware serial connection may be hanging or stalled."
                    )
                ],
                confidence_label=ConfidenceLabel.MEDIUM,
                voice_script=f"Endpoints are connected on {topic}, but no data was received during the sample window. Check your sensor's serial connection.",
                terminal_card=[
                    CommandCatalogue.get_command("topic_echo_once", topic=topic),
                    CommandCatalogue.get_command("topic_hz", topic=topic)
                ],
                next_question="Is the physical sensor plugged into USB and powered?",
                limitations=["Cannot detect hardware baud rate or permission errors without system dmesg access."]
            )

        # Healthy Topic State
        return DiagnosisResponse(
            case_id="healthy_topic",
            issue_summary=f"Topic '{topic}' is healthy with active publishers, compatible QoS, and flowing data ({sample.messages_received} msgs/sec)",
            plain_language_analogy="Like a clear radio channel with strong signal and perfect reception.",
            observations=["E1", "E2", "E4"],
            hypotheses=[
                Hypothesis(
                    label="The communication pipeline for this topic is fully operational",
                    evidence_ids=["E1", "E2", "E4"],
                    confidence=ConfidenceLabel.HIGH,
                    caveat="Downstream logic or algorithmic consumers may still have internal errors."
                )
            ],
            confidence_label=ConfidenceLabel.HIGH,
            voice_script=f"Topic {topic} is functioning properly. Data is streaming with compatible QoS settings.",
            terminal_card=[
                CommandCatalogue.get_command("topic_hz", topic=topic)
            ],
            next_question="What downstream node is experiencing issues with this data?",
            limitations=["Verifies transport layer only, does not validate semantic correctness of data values."]
        )

    def _diagnose_tf(self, target_frame: str, evidence: List[EvidenceRecord], now: str) -> DiagnosisResponse:
        source_frame = "base_link"
        tf_res = self.inspector.lookup_transform(source_frame, target_frame)

        evidence.append(EvidenceRecord(
            evidence_id="E1",
            source=EvidenceSource.TF_LOOKUP,
            observed_at=now,
            target=f"{source_frame}->{target_frame}",
            result={"available": tf_res.transform_available, "error": tf_res.error_message},
            status=tf_res.status,
            limitations="TF2 buffer lookup was bounded with a 1.0 second timeout."
        ))

        if not tf_res.transform_available:
            return DiagnosisResponse(
                case_id="case_c_tf_transform_failure",
                issue_summary=f"Coordinate transform lookup from '{source_frame}' to '{target_frame}' failed",
                plain_language_analogy="Like a digital skeleton where the shoulder has no connection to the elbow, making it impossible to calculate where the hand is in 3D space.",
                observations=["E1"],
                hypotheses=[
                    Hypothesis(
                        label=f"Frame '{target_frame}' is missing from the TF tree or robot_state_publisher is not publishing URDF static transforms",
                        evidence_ids=["E1"],
                        confidence=ConfidenceLabel.HIGH,
                        caveat="Check if your URDF includes the joint connecting base_link to this sensor frame."
                    )
                ],
                confidence_label=ConfidenceLabel.HIGH,
                voice_script=f"TF transform lookup failed between {source_frame} and {target_frame}. Check if your robot state publisher is running with the correct URDF.",
                terminal_card=[
                    CommandCatalogue.get_command("tf_view_frames"),
                    CommandCatalogue.get_command("list_nodes")
                ],
                next_question=f"Is your URDF xacro file defining the joint for {target_frame}?",
                limitations=["TF tree view requires generating a PDF report via view_frames."]
            )

        return DiagnosisResponse(
            case_id="healthy_tf",
            issue_summary=f"Transform lookup between '{source_frame}' and '{target_frame}' is valid and connected",
            plain_language_analogy="Like a fully connected skeletal arm where every joint is accounted for.",
            observations=["E1"],
            hypotheses=[
                Hypothesis(
                    label="Spatial transform link is active and synchronized",
                    evidence_ids=["E1"],
                    confidence=ConfidenceLabel.HIGH,
                    caveat="Verify that rotational and translational offsets match physical mounting dimensions."
                )
            ],
            confidence_label=ConfidenceLabel.HIGH,
            voice_script=f"Transform from {source_frame} to {target_frame} is healthy and active.",
            terminal_card=[
                CommandCatalogue.get_command("tf_view_frames")
            ],
            limitations=[]
        )

    def _diagnose_sim_time(self, evidence: List[EvidenceRecord], now: str) -> DiagnosisResponse:
        graph = self.inspector.inspect_graph()
        evidence.append(EvidenceRecord(
            evidence_id="E1",
            source=EvidenceSource.GRAPH_TOPIC,
            observed_at=now,
            target="/clock",
            result={"clock_active": graph.clock_active},
            status=EvidenceStatus.SUCCESS,
            limitations="Clock activity indicates a simulation or rosbag playback is publishing time."
        ))

        # Check nodes for use_sim_time
        sim_divergent_nodes = []
        for node in graph.nodes:
            param = self.inspector.inspect_parameter(node, "use_sim_time")
            if param.status == EvidenceStatus.SUCCESS and param.value is not None:
                if graph.clock_active and param.value is False:
                    sim_divergent_nodes.append((node, param.value))

        if sim_divergent_nodes:
            culprit, val = sim_divergent_nodes[0]
            evidence.append(EvidenceRecord(
                evidence_id="E2",
                source=EvidenceSource.PARAM_CLIENT,
                observed_at=now,
                target=f"{culprit}/use_sim_time",
                result={"value": val, "expected": True},
                status=EvidenceStatus.SUCCESS,
                limitations="Queried via ROS 2 parameter client."
            ))

            return DiagnosisResponse(
                case_id="case_e_sim_time_divergence",
                issue_summary=f"Simulation time mismatch: /clock is active, but node '{culprit}' has use_sim_time=false",
                plain_language_analogy="Like two sprinters running a race, but one reads real wall-clock time while the other is synced to a paused stopwatch.",
                observations=["E1", "E2"],
                hypotheses=[
                    Hypothesis(
                        label=f"Node '{culprit}' is ignoring simulation clock ticks, causing time extrapolation and TF buffer errors",
                        evidence_ids=["E1", "E2"],
                        confidence=ConfidenceLabel.HIGH,
                        caveat="All nodes in a Gazebo simulation session must have use_sim_time set to true."
                    )
                ],
                confidence_label=ConfidenceLabel.HIGH,
                voice_script=f"Simulation clock is active, but node {culprit} is configured with use sim time false. This causes TF time mismatch errors.",
                terminal_card=[
                    CommandCatalogue.get_command("param_get", node=culprit, param="use_sim_time")
                ],
                next_question=f"Pass 'use_sim_time:=true' into the launch file for {culprit}.",
                limitations=["Parameters set at startup cannot always be dynamically changed at runtime without relaunching."]
            )

        return DiagnosisResponse(
            case_id="healthy_sim_time",
            issue_summary="Simulation time configuration across active nodes is consistent",
            plain_language_analogy="Like two synchronized stopwatches ticking in exact unison.",
            observations=["E1"],
            hypotheses=[
                Hypothesis(
                    label="All observed nodes share a synchronized time source",
                    evidence_ids=["E1"],
                    confidence=ConfidenceLabel.HIGH,
                    caveat="Ensure simulation does not pause while hardware nodes continue polling."
                )
            ],
            confidence_label=ConfidenceLabel.HIGH,
            voice_script="Simulation time settings are consistent across inspected nodes.",
            terminal_card=[
                CommandCatalogue.get_command("list_nodes")
            ],
            limitations=[]
        )

    def _diagnose_node(self, node: str, evidence: List[EvidenceRecord], now: str) -> DiagnosisResponse:
        graph = self.inspector.inspect_graph()
        is_running = node in graph.nodes or any(n.endswith(node) for n in graph.nodes)

        evidence.append(EvidenceRecord(
            evidence_id="E1",
            source=EvidenceSource.GRAPH_TOPIC,
            observed_at=now,
            target=node,
            result={"is_running": is_running, "total_nodes": len(graph.nodes)},
            status=EvidenceStatus.SUCCESS,
            limitations="Graph snapshot cannot prove whether the node process crashed or was never launched."
        ))

        if not is_running:
            return DiagnosisResponse(
                case_id="case_b_missing_node",
                issue_summary=f"Node '{node}' was not found in the current ROS 2 domain",
                plain_language_analogy="Like looking for a worker's desk in the office and finding it completely empty.",
                observations=["E1"],
                hypotheses=[
                    Hypothesis(
                        label="The node is not started, crashed on startup, or is running under a different namespace",
                        evidence_ids=["E1"],
                        confidence=ConfidenceLabel.HIGH,
                        caveat="Check terminal console logs for crash traceback or missing shared library error."
                    )
                ],
                confidence_label=ConfidenceLabel.HIGH,
                voice_script=f"Node {node} was not observed in the graph. Check your launch logs to see if it failed on startup.",
                terminal_card=[
                    CommandCatalogue.get_command("list_nodes")
                ],
                next_question=f"What launch command did you use to start {node}?",
                limitations=["Cannot inspect node stdout/stderr without access to terminal logs."]
            )

        return DiagnosisResponse(
            case_id="healthy_node",
            issue_summary=f"Node '{node}' is active in the graph",
            plain_language_analogy="Like an employee who is logged in and actively at their desk.",
            observations=["E1"],
            hypotheses=[
                Hypothesis(
                    label="Node is registered in the ROS 2 discovery graph",
                    evidence_ids=["E1"],
                    confidence=ConfidenceLabel.HIGH,
                    caveat="A running node may still suffer internal logic hangs."
                )
            ],
            confidence_label=ConfidenceLabel.HIGH,
            voice_script=f"Node {node} is active in the ROS graph.",
            terminal_card=[
                CommandCatalogue.get_command("list_nodes")
            ],
            limitations=[]
        )
