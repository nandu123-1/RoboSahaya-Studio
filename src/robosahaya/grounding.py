import re
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

try:
    from rapidfuzz import fuzz, process
except ImportError:
    # Graceful fallback if rapidfuzz is not yet loaded
    class fuzz:
        @staticmethod
        def ratio(s1, s2):
            return 100 if s1.lower() == s2.lower() else 50
        @staticmethod
        def partial_ratio(s1, s2):
            return 100 if s2.lower() in s1.lower() else 50
    class process:
        @staticmethod
        def extractOne(query, choices, scorer=None):
            for c in choices:
                if c.lower() in query.lower():
                    return (c, 90, 0)
            return (choices[0], 60, 0) if choices else None

class GroundedEntity(BaseModel):
    raw_token: str
    canonical_name: str
    entity_type: str  # topic, node, frame, parameter
    confidence: float
    is_ambiguous: bool = False
    candidates: List[str] = Field(default_factory=list)

class GroundingResult(BaseModel):
    original_text: str
    cleaned_transcript: str
    is_code_mixed: bool
    detected_entities: List[GroundedEntity]
    primary_target: Optional[GroundedEntity] = None
    inferred_intent: str = "general_debug"

# Predefined phonetic acoustic map (ASR mistranscriptions -> canonical entities)
PHONETIC_ACOUSTIC_MAP: Dict[str, Tuple[str, str]] = {
    "slash scan": ("/scan", "topic"),
    "slash can": ("/scan", "topic"),
    "scan data": ("/scan", "topic"),
    "scan topic": ("/scan", "topic"),
    "lidar topic": ("/scan", "topic"),
    "laser data": ("/scan", "topic"),
    "laser scan": ("/scan", "topic"),
    "command well": ("/cmd_vel", "topic"),
    "cmd vel": ("/cmd_vel", "topic"),
    "cmd well": ("/cmd_vel", "topic"),
    "cmd bell": ("/cmd_vel", "topic"),
    "velocity topic": ("/cmd_vel", "topic"),
    "base link": ("base_link", "frame"),
    "base frame": ("base_link", "frame"),
    "robot frame": ("base_link", "frame"),
    "chassis frame": ("base_link", "frame"),
    "the f static": ("/tf_static", "topic"),
    "tf static": ("/tf_static", "topic"),
    "static transform": ("/tf_static", "topic"),
    "o dom": ("/odom", "topic"),
    "odometry": ("/odom", "topic"),
    "wheel odom": ("/odom", "topic"),
    "use sim time": ("use_sim_time", "parameter"),
    "sim time": ("use_sim_time", "parameter"),
    "time mismatch": ("use_sim_time", "parameter"),
    "clock": ("/clock", "topic"),
    "camera link": ("camera_link", "frame"),
    "camera frame": ("camera_link", "frame"),
    "slam toolbox": ("/slam_toolbox", "node"),
    "slam node": ("/slam_toolbox", "node"),
    "mapping node": ("/slam_toolbox", "node")
}

# Telugu code-mixed markers
TELUGU_MARKERS = [
    "ravatledu", "ravadam ledu", "tirugutondi", "move avvatledu",
    "kadalatledu", "vastunnaya", "vastondi", "running lo unda",
    "kani", "enduku", "enti", "pani cheyatledu", "tho", "levu"
]

class EntityGrounder:
    def __init__(self, known_topics: Optional[List[str]] = None,
                 known_nodes: Optional[List[str]] = None,
                 known_frames: Optional[List[str]] = None):
        self.known_topics = known_topics or ["/scan", "/cmd_vel", "/odom", "/tf", "/tf_static", "/clock"]
        self.known_nodes = known_nodes or ["/robot_state_publisher", "/slam_toolbox", "/diff_drive_controller", "/camera_driver"]
        self.known_frames = known_frames or ["base_link", "base_footprint", "laser_frame", "camera_link", "odom", "map"]


    def update_graph_knowledge(self, topics: List[str], nodes: List[str], frames: List[str]):
        """Dynamically ground against live ROS 2 graph entities."""
        self.known_topics = list(set(self.known_topics + topics))
        self.known_nodes = list(set(self.known_nodes + nodes))
        self.known_frames = list(set(self.known_frames + frames))

    def detect_code_mixed(self, text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in TELUGU_MARKERS)

    def ground(self, transcript: str) -> GroundingResult:
        lower = transcript.lower()
        is_code_mixed = self.detect_code_mixed(transcript)
        detected: List[GroundedEntity] = []

        # 1. Check known phonetic variations
        for phonetic, (canonical, etype) in PHONETIC_ACOUSTIC_MAP.items():
            if phonetic in lower:
                detected.append(GroundedEntity(
                    raw_token=phonetic,
                    canonical_name=canonical,
                    entity_type=etype,
                    confidence=0.95
                ))

        # 2. Check exact known topics/nodes/frames mentioned in transcript
        for topic in self.known_topics:
            if topic.lower() in lower and not any(d.canonical_name == topic for d in detected):
                detected.append(GroundedEntity(
                    raw_token=topic,
                    canonical_name=topic,
                    entity_type="topic",
                    confidence=1.0
                ))

        for frame in self.known_frames:
            if frame.lower() in lower and not any(d.canonical_name == frame for d in detected):
                detected.append(GroundedEntity(
                    raw_token=frame,
                    canonical_name=frame,
                    entity_type="frame",
                    confidence=0.98
                ))

        for node in self.known_nodes:
            if node.lower() in lower and not any(d.canonical_name == node for d in detected):
                detected.append(GroundedEntity(
                    raw_token=node,
                    canonical_name=node,
                    entity_type="node",
                    confidence=0.98
                ))

        # 3. Fuzzy match remaining tokens if no high-confidence match
        if not detected:
            all_known = (
                [(t, "topic") for t in self.known_topics] +
                [(n, "node") for n in self.known_nodes] +
                [(f, "frame") for f in self.known_frames]
            )
            names_only = [item[0] for item in all_known]
            match = process.extractOne(lower, names_only, scorer=fuzz.partial_ratio)
            if match and match[1] >= 65:
                matched_name = match[0]
                matched_type = next(t for n, t in all_known if n == matched_name)
                conf = round(match[1] / 100.0, 2)
                detected.append(GroundedEntity(
                    raw_token=transcript,
                    canonical_name=matched_name,
                    entity_type=matched_type,
                    confidence=conf,
                    is_ambiguous=conf < 0.85,
                    candidates=[n for n in names_only if n != matched_name][:3]
                ))

        # Select primary target
        primary = max(detected, key=lambda x: x.confidence) if detected else None

        # Infer intent
        intent = "general_debug"
        if "sim" in lower or "clock" in lower or "time mismatch" in lower:
            intent = "sim_time_check"
        elif "qos" in lower or "compatible" in lower or "reliability" in lower:
            intent = "qos_check"
        elif "tf" in lower or "frame" in lower or "transform" in lower:
            intent = "tf_check"
        elif primary and primary.entity_type == "topic":
            intent = "topic_inspection"

        return GroundingResult(
            original_text=transcript,
            cleaned_transcript=transcript.strip(),
            is_code_mixed=is_code_mixed,
            detected_entities=detected,
            primary_target=primary,
            inferred_intent=intent
        )
