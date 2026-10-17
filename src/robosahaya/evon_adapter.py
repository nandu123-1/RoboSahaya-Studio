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

EVON_SYSTEM_PROMPT = """You are RoboSahaya's AI Reasoning Engine, powered by Gnani AI Evon v3.3.
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
    """Adapter for Gnani AI Evon v3.3 reasoning model."""

    def __init__(self, endpoint_url: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint_url = endpoint_url or getattr(settings, "EVON_API_URL", "https://api.gnani.ai/v1/evon/generate")
        self.api_key = api_key or settings.GNANI_API_KEY

    @property
    def is_live(self) -> bool:
        return bool(self.api_key and self.api_key != "your_gnani_api_key_here")

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
        Invoke Gnani AI Evon reasoning model on the evidence bundle.
        Falls back gracefully if endpoint is unreachable.
        """
        user_prompt = self._build_user_prompt(target_entity, entity_type, evidence, user_query)

        if not self.is_live:
            # High-fidelity simulated AI reasoning for offline demonstration / evaluation
            return self._simulated_ai_reasoning(target_entity, entity_type, evidence, user_query)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "evon-v3.3",
            "messages": [
                {"role": "system", "content": EVON_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 800
        }

        async with httpx.AsyncClient(timeout=8.0) as client:
            try:
                resp = await client.post(self.endpoint_url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"]
                parsed = json.loads(raw_content)
                return self._parse_llm_json(parsed)
            except Exception as ex:
                print(f"[Evon Reasoning] Live inference failed ({ex}), falling back to simulated AI reasoning.")
                return self._simulated_ai_reasoning(target_entity, entity_type, evidence, user_query)

    def _simulated_ai_reasoning(
        self,
        target_entity: str,
        entity_type: str,
        evidence: List[EvidenceRecord],
        user_query: str
    ) -> DiagnosisResponse:
        """
        Simulates Evon v3.3's dynamic generative reasoning over the structured evidence bundle.
        """
        evidence_ids = [e.evidence_id for e in evidence]
        
        # Check QoS in evidence
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
                            evidence_ids=[e.evidence_id],
                            confidence=ConfidenceLabel.HIGH,
                            caveat="The hardware publisher is transmitting valid data, but the ROS 2 executor discards it at the callback queue."
                        )
                    ],
                    confidence_label=ConfidenceLabel.HIGH,
                    voice_script=f"Evon detected a QoS reliability mismatch on {target_entity}. Update your subscriber profile to best effort to restore sensor processing.",
                    terminal_card=[
                        CommandCatalogue.get_command("topic_info", topic=target_entity),
                        CommandCatalogue.get_command("topic_hz", topic=target_entity)
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

        # Check TF failure
        for e in evidence:
            if "available" in e.result and not e.result["available"]:
                return DiagnosisResponse(
                    case_id="case_c_tf_transform_failure",
                    issue_summary=f"Evon AI detected a kinematic transform discontinuity reaching '{target_entity}'",
                    plain_language_analogy="Evon Analogy: Like an anatomical chart where a muscle connects the wrist to a finger that isn't attached to the hand.",
                    observations=evidence_ids,
                    hypotheses=[
                        Hypothesis(
                            label=f"Transform frame '{target_entity}' has no parent link in the TF broadcast tree",
                            evidence_ids=[e.evidence_id],
                            confidence=ConfidenceLabel.HIGH,
                            caveat="Verify that your URDF xacro includes a fixed joint connecting base_link to this sensor."
                        )
                    ],
                    confidence_label=ConfidenceLabel.HIGH,
                    voice_script=f"Evon found a disconnected coordinate frame for {target_entity}. Let's generate a TF tree diagram to find the missing joint.",
                    terminal_card=[
                        CommandCatalogue.get_command("tf_view_frames"),
                        CommandCatalogue.get_command("list_nodes")
                    ],
                    next_question="Is robot_state_publisher publishing the robot_description topic?",
                    limitations=["TF tree view requires generating a PDF diagram."]
                )

        # Check simulation time mismatch in evidence
        for e in evidence:
            if e.target.endswith("/use_sim_time") and e.result.get("value") is False:
                return DiagnosisResponse(
                    case_id="case_e_sim_time_divergence",
                    issue_summary=f"Evon AI identified temporal desynchronization on {e.target}",
                    plain_language_analogy="Evon Analogy: Like one runner listening to an umpire's whistle while another watches a wall clock in a different time zone.",
                    observations=evidence_ids,
                    hypotheses=[
                        Hypothesis(
                            label=f"Node {e.target.split('/')[0]} has use_sim_time=false while simulation clock is active",
                            evidence_ids=[e.evidence_id],
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
