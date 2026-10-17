import re
from typing import Dict, Optional
from robosahaya.evidence_models import CommandItem

# Whitelist pattern for ROS 2 entity names (alphanumeric, underscores, slashes, tildes)
SAFE_ENTITY_PATTERN = re.compile(r"^[a-zA-Z0-9_/~]+$")

COMMAND_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "list_nodes": {
        "template": "ros2 node list",
        "purpose": "Identify active nodes in the ROS domain",
        "expected": "Check if the expected driver or control node is visible."
    },
    "list_topics": {
        "template": "ros2 topic list -t",
        "purpose": "List available topics with their message types",
        "expected": "Verify if the topic is published under the expected name or remapped."
    },
    "topic_info": {
        "template": "ros2 topic info {topic} --verbose",
        "purpose": "Inspect endpoint counts and QoS policies for a topic",
        "expected": "Compare publisher and subscriber reliability and durability settings."
    },
    "param_get": {
        "template": "ros2 param get {node} {param}",
        "purpose": "Query node parameter configuration",
        "expected": "Verify parameter value (e.g. use_sim_time: true/false)."
    },
    "tf_view_frames": {
        "template": "ros2 run tf2_tools view_frames",
        "purpose": "Generate PDF diagram of coordinate transform tree",
        "expected": "Check for breaks between source and target coordinate frames."
    },
    "topic_hz": {
        "template": "ros2 topic hz {topic}",
        "purpose": "Measure publication frequency in Hz",
        "expected": "Verify that messages are arriving continuously at expected rates."
    },
    "topic_echo_once": {
        "template": "ros2 topic echo {topic} --once",
        "purpose": "Print a single sample message from topic",
        "expected": "Verify that message payload fields contain non-zero data."
    }
}

class CommandCatalogue:
    @staticmethod
    def sanitize_arg(value: str) -> str:
        """Validate and sanitize an entity argument to prevent shell injection."""
        value = value.strip()
        if not SAFE_ENTITY_PATTERN.match(value):
            raise ValueError(f"Invalid character in ROS entity name: {value}")
        return value

    @classmethod
    def get_command(cls, command_id: str, **kwargs) -> CommandItem:
        """Retrieve and format a safe command from the approved catalogue."""
        if command_id not in COMMAND_DEFINITIONS:
            raise KeyError(f"Command ID '{command_id}' not found in approved Command Catalogue.")
        
        definition = COMMAND_DEFINITIONS[command_id]
        template = definition["template"]
        
        sanitized_kwargs = {k: cls.sanitize_arg(str(v)) for k, v in kwargs.items()}
        formatted_cmd = template.format(**sanitized_kwargs)
        
        return CommandItem(
            command_id=command_id,
            raw_command=formatted_cmd,
            purpose=definition["purpose"],
            expected_observation=definition["expected"]
        )
