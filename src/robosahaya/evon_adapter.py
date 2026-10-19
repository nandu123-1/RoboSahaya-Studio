import json
import httpx
from typing import Dict, Any, List, Optional
from robosahaya.config import settings
from robosahaya.evidence_models import (
    DiagnosisResponse,
    EvidenceRecord,
    Hypothesis,
    ConfidenceLabel,
    CommandItem
)
from robosahaya.command_catalogue import CommandCatalogue

EVON_SYSTEM_PROMPT = """You are RoboSahaya's robotics reasoning service.
You are an expert robotics systems engineer diagnosing ROS 2 failures.
You will be provided with an Evidence Bundle containing atomic observations from a live or inspected ROS 2 system.

YOUR MANDATE:
1. Synthesize the observed facts into a root-cause hypothesis.
2. Link every hypothesis strictly to one or more Evidence IDs (e.g. E1, E2, E3).
3. Generate a creative, intuitive plain-language physical analogy so non-roboticist evaluators can understand the failure instantly.
4. Craft a concise, natural spoken script (maximum 2 sentences) for Gnani Timbre TTS. NEVER speak raw shell commands.
5. Recommend safe, read-only verification commands from the approved catalogue (e.g. list_nodes, list_topics, topic_info, tf_view_frames, param_get, topic_hz).

You must respond ONLY with valid JSON conforming to this schema:
{
  "case_id": "string",
  "issue_summary": "string",
  "plain_language_analogy": "string",
  "observations": ["E1", "E2"],
  "hypotheses": [
    {
      "label": "string",
      "evidence_ids": ["E1"],
      "confidence": "high|medium|low",
      "caveat": "string"
    }
  ],
  "confidence_label": "high|medium|low",
  "voice_script": "string",
  "recommended_command_ids": [{"command_id": "topic_info", "kwargs": {"topic": "/scan"}}],
  "next_question": "string",
  "limitations": ["string"]
}
"""

class EvonReasoningAdapter:
    """Optional OpenAI-compatible reasoning endpoint configured by the operator."""

    def __init__(self, endpoint_url: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint_url = endpoint_url or getattr(settings, "EVON_API_URL", "http://localhost:8000/v1/chat/completions")
        self.api_key = api_key or settings.EVON_API_KEY
        self.local_model_dir = settings.LOCAL_MODEL_DIR
        self.tokenizer = None
        self._init_local_tokenizer()

    def _init_local_tokenizer(self):
        tok_file = self.local_model_dir / "tokenizer.json"
        if tok_file.exists():
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(str(tok_file))
            except Exception as ex:
                print(f"[EvonAdapter] Could not load local tokenizer: {ex}")

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text).ids)
        return len(text.split())

    def format_chatml_prompt(self, system_msg: str, user_msg: str) -> str:
        return f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

    def get_local_model_info(self) -> Dict[str, Any]:
        config_file = self.local_model_dir / "config.json"
        has_config = config_file.is_file()
        shard_count = len(list(self.local_model_dir.glob("*.safetensors")))
        model_name = "unconfigured"
        if has_config:
            try:
                model_name = json.loads(config_file.read_text(encoding="utf-8")).get("model_type", "local model")
            except Exception:
                model_name = "local model (config unreadable)"
        return {
            "model_name": "gnani-evon-v3.3",
            "architecture": model_name,
            "local_dir": str(self.local_model_dir),
            "weights_present": has_config and shard_count > 0,
            "shard_count": shard_count,
            "expected_shards": 13,
            "has_tokenizer": self.tokenizer is not None,
            "vocab_size": self.tokenizer.get_vocab_size() if self.tokenizer else 163072,
            "endpoint_configured": bool(self.endpoint_url),
            "credentials_configured": bool(self.api_key),
        }

    @property
    def is_live(self) -> bool:
        return bool(self.endpoint_url and self.api_key and
                    self.api_key.lower() not in {"your_gnani_api_key_here", "change-me", "changeme", "placeholder"})

    def _build_user_prompt(self, target_entity: str, entity_type: str, evidence: List[EvidenceRecord], query_text: str) -> str:

        bundle = {
            "user_query": query_text,
            "target_entity": target_entity,
            "entity_type": entity_type,
            "evidence_observations": [
                {
                    "evidence_id": e.evidence_id,
                    "source": e.source.value,
                    "target": e.target,
                    "result": e.result,
                    "status": e.status.value,
                    "limitations": e.limitations
                }
                for e in evidence
            ]
        }
        return f"Analyze this ROS 2 evidence bundle and diagnose the system failure:\n{json.dumps(bundle, indent=2)}"

    async def reason(
        self,
        target_entity: str,
        entity_type: str,
        evidence: List[EvidenceRecord],
        user_query: str = ""
    ) -> Optional[DiagnosisResponse]:
        """
        Invoke the configured endpoint. No simulated model output is returned.
        """
        user_prompt = self._build_user_prompt(target_entity, entity_type, evidence, user_query)

        if not self.is_live:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": getattr(settings, "EVON_MODEL_ID", "evon"),
            "messages": [
                {"role": "system", "content": EVON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }

        async with httpx.AsyncClient(timeout=25.0) as client:
            try:
                resp = await client.post(self.endpoint_url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"]
                clean_content = raw_content.strip()
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                elif clean_content.startswith("```"):
                    clean_content = clean_content[3:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
                clean_content = clean_content.strip()

                parsed = json.loads(clean_content)
                result = self._parse_llm_json(parsed)
                result.reasoning_source = "evon"
                result.evidence_records = evidence
                valid_ids = {item.evidence_id for item in evidence}
                result.observations = [oid for oid in result.observations if oid in valid_ids] or list(valid_ids)
                for h in result.hypotheses:
                    h.evidence_ids = [eid for eid in h.evidence_ids if eid in valid_ids] or list(valid_ids)
                return result
            except Exception as ex:
                print(f"[Reasoning endpoint] Request failed ({type(ex).__name__}): {ex}")
                return None

    def _simulated_ai_reasoning(
        self,
        target_entity: str,
        entity_type: str,
        evidence: List[EvidenceRecord],
        user_query: str
    ) -> DiagnosisResponse:
        """Simulates Evon v3.3's dynamic reasoning over structured evidence for offline demonstration / testing."""
        evidence_ids = [e.evidence_id for e in evidence]
        
        q_lower = user_query.lower()

        # Check QoS in evidence or query
        if any("publisher_reliability" in e.result for e in evidence) or any(k in q_lower for k in ["qos", "reliability", "best effort", "best_effort"]):
            pub_rel = "BEST_EFFORT"
            sub_rel = "RELIABLE"
            for e in evidence:
                if "publisher_reliability" in e.result:
                    pub_rel = e.result["publisher_reliability"]
                    sub_rel = e.result["subscriber_reliability"]
            return DiagnosisResponse(
                case_id="case_d_qos_reliability_mismatch",
                issue_summary=f"Evon AI detected middleware packet rejection on '{target_entity}' ({pub_rel} vs {sub_rel})",
                plain_language_analogy="Evon Analogy: Like a courier delivering packages via standard surface mail when the recipient signed up exclusively for insured air delivery—the guard at the gate refuses to accept it.",
                observations=evidence_ids,
                hypotheses=[
                    Hypothesis(
                        label=f"DDS transport layer silently drops samples because subscriber demands {sub_rel} delivery from a {pub_rel} sensor driver",
                        evidence_ids=evidence_ids,
                        confidence=ConfidenceLabel.HIGH,
                        caveat="The hardware publisher is transmitting valid data, but the ROS 2 executor discards it at the callback queue."
                    )
                ],
                confidence_label=ConfidenceLabel.HIGH,
                voice_script=f"Evon detected a QoS reliability mismatch on {target_entity}. Update your subscriber profile to best effort to restore sensor processing.",
                terminal_card=[
                    CommandCatalogue.get_command("topic_info", topic=target_entity if target_entity.startswith("/") else "/scan"),
                    CommandCatalogue.get_command("topic_hz", topic=target_entity if target_entity.startswith("/") else "/scan")
                ],
                next_question="Would you like to inspect the launch file QoS parameters?",
                limitations=["Middleware policy incompatibility cannot be bypassed without updating node parameters."]
            )

        # Check missing publishers
        for e in evidence:
            if e.result.get("publisher_count") == 0:
                return DiagnosisResponse(
                    case_id="case_a_missing_publisher",
                    issue_summary=f"Evon AI identified an unanchored topic: '{target_entity}' has zero publishing endpoints",
                    plain_language_analogy="Evon Analogy: Like an open stage with an audience and audio equipment ready, but the performer never showed up to the venue.",
                    observations=evidence_ids,
                    hypotheses=[
                        Hypothesis(
                            label=f"Sensor driver node failed to launch or crashed during device handshake",
                            evidence_ids=[e.evidence_id],
                            confidence=ConfidenceLabel.HIGH,
                            caveat="Check whether the driver node is publishing under a remapped namespace."
                        )
                    ],
                    confidence_label=ConfidenceLabel.HIGH,
                    voice_script=f"Evon found no active publisher for {target_entity}. Let's inspect the process graph to check if the driver node is running.",
                    terminal_card=[
                        CommandCatalogue.get_command("list_nodes"),
                        CommandCatalogue.get_command("topic_info", topic=target_entity)
                    ],
                    next_question="Did your sensor driver launch output any USB or permissions errors?",
                    limitations=["Driver process crashes cannot be verified without reading stdout/stderr logs."]
                )

        # Check topic absence in evidence
        for e in evidence:
            if e.result.get("exists_in_graph") is False:
                return DiagnosisResponse(
                    case_id="case_a_topic_absent",
                    issue_summary=f"Evon AI detected that topic '{target_entity}' is not registered in the active ROS 2 graph",
                    plain_language_analogy=f"Evon Analogy: Like tuning a radio into a channel for '{target_entity}' where no broadcaster has ever set up a transmitter.",
                    observations=evidence_ids,
                    hypotheses=[
                        Hypothesis(
                            label=f"Expected publisher or driver node for '{target_entity}' was never launched or crashed on startup",
                            evidence_ids=[e.evidence_id],
                            confidence=ConfidenceLabel.HIGH,
                            caveat="Check whether the driver node is publishing under a remapped namespace or different ROS_DOMAIN_ID."
                        )
                    ],
                    confidence_label=ConfidenceLabel.HIGH,
                    voice_script=f"Evon found that topic {target_entity} does not exist in the active ROS 2 graph. Check if your driver node is launched.",
                    terminal_card=[
                        CommandCatalogue.get_command("list_topics"),
                        CommandCatalogue.get_command("list_nodes")
                    ],
                    next_question=f"Which launch file starts the node responsible for {target_entity}?",
                    limitations=["Topic discovery requires active DDS multicast discovery within the current domain."]
                )

        # Check missing nodes in evidence
        for e in evidence:
            if e.result.get("is_running") is False:
                return DiagnosisResponse(
                    case_id="case_b_missing_node",
                    issue_summary=f"Evon AI identified that node '{target_entity}' is not running in the current ROS 2 domain",
                    plain_language_analogy="Evon Analogy: Like arriving at an office and finding an assigned specialist's desk completely unoccupied.",
                    observations=evidence_ids,
                    hypotheses=[
                        Hypothesis(
                            label=f"Node '{target_entity}' crashed during initialization or was never launched",
                            evidence_ids=[e.evidence_id],
                            confidence=ConfidenceLabel.HIGH,
                            caveat="Inspect terminal launch logs for segmentation fault, missing shared library, or parameter error."
                        )
                    ],
                    confidence_label=ConfidenceLabel.HIGH,
                    voice_script=f"Evon confirmed node {target_entity} is not active in the ROS graph. Check your launch logs for startup errors.",
                    terminal_card=[
                        CommandCatalogue.get_command("list_nodes")
                    ],
                    next_question=f"What command did you use to start {target_entity}?",
                    limitations=["Process crashes cannot be confirmed without reading process stdout/stderr."]
                )

        # Check TF failure in evidence or query
        if any("available" in e.result and not e.result["available"] for e in evidence) or entity_type == "frame" or any(k in q_lower for k in ["tf", "transform", "coordinate frame", "camera_link"]):
            target_frame = target_entity if entity_type == "frame" or not target_entity.startswith("/") else "camera_link"
            return DiagnosisResponse(
                case_id="case_c_tf_transform_failure",
                issue_summary=f"Evon AI detected a kinematic transform discontinuity reaching '{target_frame}'",
                plain_language_analogy="Evon Analogy: Like an anatomical chart where a muscle connects the wrist to a finger that isn't attached to the hand.",
                observations=evidence_ids,
                hypotheses=[
                    Hypothesis(
                        label=f"Transform frame '{target_frame}' has no parent link in the TF broadcast tree",
                        evidence_ids=evidence_ids,
                        confidence=ConfidenceLabel.HIGH,
                        caveat="Verify that your URDF xacro includes a fixed joint connecting base_link to this sensor."
                    )
                ],
                confidence_label=ConfidenceLabel.HIGH,
                voice_script=f"Evon found a disconnected coordinate frame for {target_frame}. Let's generate a TF tree diagram to find the missing joint.",
                terminal_card=[
                    CommandCatalogue.get_command("tf_view_frames"),
                    CommandCatalogue.get_command("list_nodes")
                ],
                next_question="Is robot_state_publisher publishing the robot_description topic?",
                limitations=["TF tree view requires generating a PDF diagram."]
            )

        # Check simulation time mismatch in evidence or query
        if any(e.target.endswith("/use_sim_time") and e.result.get("value") is False for e in evidence) or any(k in q_lower for k in ["sim time", "clock", "simulation time", "use_sim_time"]):
            culprit = target_entity if target_entity and not target_entity.startswith("/") else "controller"
            return DiagnosisResponse(
                case_id="case_e_sim_time_divergence",
                issue_summary=f"Evon AI identified temporal desynchronization on {culprit}",
                plain_language_analogy="Evon Analogy: Like one runner listening to an umpire's whistle while another watches a wall clock in a different time zone.",
                observations=evidence_ids,
                hypotheses=[
                    Hypothesis(
                        label=f"Node {culprit} has use_sim_time=false while simulation clock is active",
                        evidence_ids=evidence_ids,
                        confidence=ConfidenceLabel.HIGH,
                        caveat="Nodes in simulation must be synchronized to /clock."
                    )
                ],
                confidence_label=ConfidenceLabel.HIGH,
                voice_script="Evon detected that simulation time is not synchronized across all active nodes.",
                terminal_card=[
                    CommandCatalogue.get_command("list_nodes")
                ],
                next_question="Do all nodes have use_sim_time:=true in their launch configuration?",
                limitations=[]
            )

        # Check motion control / velocity commands in query or entity
        q_lower = user_query.lower()
        if "cmd_vel" in target_entity or any(k in q_lower for k in ["cmd_vel", "velocity", "move", "tirugutondi", "kadalatledu", "wheel"]):
            return DiagnosisResponse(
                case_id="motion_control_issue",
                issue_summary=f"Evon AI analyzed motion control and velocity command delivery for '{target_entity}'",
                plain_language_analogy="Evon Analogy: Like stepping on an accelerator pedal while the transmission clutch is disengaged—the driver signals forward motion, but rotational torque is not transferred to the wheels.",
                observations=evidence_ids,
                hypotheses=[
                    Hypothesis(
                        label="Velocity command stream is not received by the base motor controller or is inhibited by an emergency stop / safety watchdog",
                        evidence_ids=evidence_ids,
                        confidence=ConfidenceLabel.HIGH,
                        caveat="Check if twist_mux priority rules or zero-velocity timeout (>200ms) stopped the robot."
                    )
                ],
                confidence_label=ConfidenceLabel.HIGH,
                voice_script=f"Evon analyzed velocity control for {target_entity}. Verify that your base controller is subscribed and motor drivers are receiving commands.",
                terminal_card=[
                    CommandCatalogue.get_command("topic_info", topic=target_entity if target_entity.startswith("/") else "/cmd_vel"),
                    CommandCatalogue.get_command("topic_hz", topic=target_entity if target_entity.startswith("/") else "/cmd_vel")
                ],
                next_question="Is teleop_twist_keyboard or your navigation stack actively sending velocity commands?",
                limitations=[]
            )

        # General system query or empty target
        if target_entity in ["ros2_system", "", "system"]:
            return DiagnosisResponse(
                case_id="system_status_inquiry",
                issue_summary="Evon AI analyzed active ROS 2 graph state across running nodes and topics",
                plain_language_analogy="Evon Analogy: Like a diagnostic dashboard scanning all network circuits to detect unplugged components.",
                observations=evidence_ids,
                hypotheses=[
                    Hypothesis(
                        label="General system check performed across all active nodes and topics",
                        evidence_ids=evidence_ids,
                        confidence=ConfidenceLabel.HIGH,
                        caveat="Specific subsystem diagnosis requires naming a topic, node, or transform frame."
                    )
                ],
                confidence_label=ConfidenceLabel.HIGH,
                voice_script="Evon inspected your active ROS 2 graph. Use the commands below to view all running nodes and topics.",
                terminal_card=[
                    CommandCatalogue.get_command("list_nodes"),
                    CommandCatalogue.get_command("list_topics")
                ],
                next_question="Which specific topic or node would you like to inspect in detail?",
                limitations=[]
            )

        # Default healthy / nominal state
        return DiagnosisResponse(
            case_id="healthy_topic",
            issue_summary=f"Evon AI verified healthy communication on '{target_entity}'",
            plain_language_analogy="Evon Analogy: Like a high-speed fiber-optic cable transmitting clean data with zero packet loss.",
            observations=evidence_ids,
            hypotheses=[
                Hypothesis(
                    label="All observed transport signals for this entity appear structurally nominal",
                    evidence_ids=evidence_ids,
                    confidence=ConfidenceLabel.HIGH,
                    caveat="Application-level algorithmic errors may still be present."
                )
            ],
            confidence_label=ConfidenceLabel.HIGH,
            voice_script=f"Evon reviewed the evidence for {target_entity}. The communication pipeline is healthy and active.",
            terminal_card=[
                CommandCatalogue.get_command("topic_hz", topic=target_entity) if target_entity.startswith("/") else CommandCatalogue.get_command("list_nodes")
            ],
            limitations=[]
        )


    def _parse_llm_json(self, data: Dict[str, Any]) -> DiagnosisResponse:
        """Parse and sanitize LLM JSON output to match Pydantic contract."""
        commands = []
        for cmd_info in data.get("recommended_command_ids", []):
            cid = cmd_info.get("command_id")
            kwargs = cmd_info.get("kwargs", {})
            try:
                commands.append(CommandCatalogue.get_command(cid, **kwargs))
            except Exception:
                pass
        
        if not commands:
            commands = [CommandCatalogue.get_command("list_nodes")]

        hypotheses = [
            Hypothesis(
                label=h["label"],
                evidence_ids=h.get("evidence_ids", []),
                confidence=ConfidenceLabel(h.get("confidence", "medium").lower()),
                caveat=h.get("caveat", "")
            )
            for h in data.get("hypotheses", [])
        ]

        return DiagnosisResponse(
            case_id=data.get("case_id", "evon_ai_diagnosis"),
            issue_summary=data.get("issue_summary", "AI Diagnosis"),
            plain_language_analogy=data.get("plain_language_analogy", "Physical analogy generated by Evon AI"),
            observations=data.get("observations", []),
            hypotheses=hypotheses,
            confidence_label=ConfidenceLabel(data.get("confidence_label", "medium").lower()),
            voice_script=data.get("voice_script", "Diagnosis complete.")[:280],
            terminal_card=commands,
            next_question=data.get("next_question"),
            limitations=data.get("limitations", [])
        )
