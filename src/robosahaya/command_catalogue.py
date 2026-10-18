"""Allowlisted tool contracts and command catalogue for RoboSahaya Studio.

This module describes tools; it does not execute shell commands. Tool adapters should
implement each registered operation directly through rclpy, filesystem APIs, or a
fixed subprocess argument vector. Never evaluate model-generated shell strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .evidence_models import CommandItem, PermissionTier, ToolRequest


class ToolCatalogueError(ValueError):
    """Raised when a tool request is unknown or malformed."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    tool_id: str
    tier: PermissionTier
    description: str
    required_arguments: frozenset[str]
    optional_arguments: frozenset[str]
    validator: Callable[[dict[str, Any]], None]

    @property
    def allowed_arguments(self) -> frozenset[str]:
        return self.required_arguments | self.optional_arguments


def _no_args(args: dict[str, Any]) -> None:
    return None


def _topic_args(args: dict[str, Any]) -> None:
    topic = args.get("topic")
    if not isinstance(topic, str) or not re.fullmatch(r"/?[A-Za-z][A-Za-z0-9_/]*", topic):
        raise ToolCatalogueError("topic must be a valid ROS-style name")
    if ".." in topic.split("/"):
        raise ToolCatalogueError("topic must not contain path-traversal components")


def _node_param_args(args: dict[str, Any]) -> None:
    node = args.get("node")
    param = args.get("parameter")
    if not isinstance(node, str) or not re.fullmatch(r"/?[A-Za-z][A-Za-z0-9_/]*", node):
        raise ToolCatalogueError("node must be a valid ROS-style name")
    if not isinstance(param, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", param):
        raise ToolCatalogueError("parameter must be a valid ROS parameter name")


def _tf_args(args: dict[str, Any]) -> None:
    for key in ("source_frame", "target_frame"):
        value = args.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_/]*", value):
            raise ToolCatalogueError(f"{key} must be a valid frame name")


def _sample_args(args: dict[str, Any]) -> None:
    _topic_args(args)
    duration = args.get("duration_sec", 1.0)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 0.1 <= duration <= 3.0:
        raise ToolCatalogueError("duration_sec must be between 0.1 and 3.0 seconds")
    max_messages = args.get("max_messages", 20)
    if isinstance(max_messages, bool) or not isinstance(max_messages, int) or not 1 <= max_messages <= 100:
        raise ToolCatalogueError("max_messages must be an integer from 1 to 100")


def _workspace_args(args: dict[str, Any]) -> None:
    profile = args.get("profile")
    if not isinstance(profile, str) or profile not in {"diff_drive_lidar_camera_v1", "differential_drive"}:
        raise ToolCatalogueError("profile is not in the approved simulation-template catalogue")
    run_id = args.get("run_id", "")
    if not isinstance(run_id, str) or (run_id and not re.fullmatch(r"[A-Za-z0-9_-]{4,80}", run_id)):
        raise ToolCatalogueError("run_id must use 4-80 letters, numbers, underscores, or hyphens")


def _package_args(args: dict[str, Any]) -> None:
    package = args.get("package")
    if not isinstance(package, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,62}", package):
        raise ToolCatalogueError("package must be an allowlisted-style ROS package name")


def _launch_args(args: dict[str, Any]) -> None:
    profile = args.get("launch_profile")
    if not isinstance(profile, str) or profile not in {"diff_drive_lidar_camera_v1", "differential_drive"}:
        raise ToolCatalogueError("launch_profile is not registered")


def _path_args(args: dict[str, Any]) -> None:
    relative = args.get("relative_path")
    if not isinstance(relative, str) or not relative or relative.startswith(("/", "\\")):
        raise ToolCatalogueError("relative_path must be a workspace-relative path")
    normalized = relative.replace("\\", "/")
    if any(part in {"..", ""} for part in normalized.split("/")) or "\x00" in relative:
        raise ToolCatalogueError("relative_path contains unsafe path components")


def _patch_args(args: dict[str, Any]) -> None:
    patch_id = args.get("patch_id")
    token = args.get("approval_token")
    if not isinstance(patch_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{4,80}", patch_id):
        raise ToolCatalogueError("patch_id has invalid format")
    if not isinstance(token, str) or len(token) < 20:
        raise ToolCatalogueError("approval_token is required")


def _test_args(args: dict[str, Any]) -> None:
    test_id = args.get("test_id")
    if not isinstance(test_id, str) or test_id not in {
        "lidar_stream_valid", "camera_stream_valid", "tf_connectivity", "simulation_clock_advancing", "test_laser_scan_payload"
    }:
        raise ToolCatalogueError("test_id is not in the named-test catalogue")


def _report_args(args: dict[str, Any]) -> None:
    fmt = args.get("format", "markdown")
    if fmt not in {"markdown", "json"}:
        raise ToolCatalogueError("format must be 'markdown' or 'json'")


_DEFINITIONS = [
    ToolDefinition("list_ros_nodes", PermissionTier.READ_ONLY, "List currently visible ROS 2 nodes.", frozenset(), frozenset(), _no_args),
    ToolDefinition("list_ros_topics", PermissionTier.READ_ONLY, "List current ROS 2 topics and types.", frozenset(), frozenset(), _no_args),
    ToolDefinition("get_topic_info", PermissionTier.READ_ONLY, "Inspect a topic's type and endpoint metadata.", frozenset({"topic"}), frozenset(), _topic_args),
    ToolDefinition("sample_topic_messages", PermissionTier.READ_ONLY, "Sample bounded topic messages and validate payloads.", frozenset({"topic"}), frozenset({"duration_sec", "max_messages"}), _sample_args),
    ToolDefinition("get_node_parameter", PermissionTier.READ_ONLY, "Read one node parameter with a bounded timeout.", frozenset({"node", "parameter"}), frozenset(), _node_param_args),
    ToolDefinition("lookup_tf_transform", PermissionTier.READ_ONLY, "Perform a bounded TF transform lookup.", frozenset({"source_frame", "target_frame"}), frozenset(), _tf_args),
    ToolDefinition("assemble_simulation_template", PermissionTier.WORKSPACE, "Create a project from an approved template in the isolated workspace.", frozenset({"profile"}), frozenset({"run_id"}), _workspace_args),
    ToolDefinition("validate_project_files", PermissionTier.WORKSPACE, "Validate known project files within the workspace.", frozenset(), frozenset(), _no_args),
    ToolDefinition("build_workspace", PermissionTier.WORKSPACE, "Build a named ROS package using a fixed build command.", frozenset({"package"}), frozenset(), _package_args),
    ToolDefinition("run_named_test", PermissionTier.WORKSPACE, "Run a predefined verification test, not arbitrary test code.", frozenset({"test_id"}), frozenset(), _test_args),
    ToolDefinition("launch_simulation_profile", PermissionTier.MANAGED_RUNTIME, "Launch one registered simulation profile under process supervision.", frozenset({"launch_profile"}), frozenset(), _launch_args),
    ToolDefinition("stop_managed_simulation", PermissionTier.MANAGED_RUNTIME, "Stop only a simulation process owned by this application.", frozenset(), frozenset(), _no_args),
    ToolDefinition("apply_reviewed_patch", PermissionTier.MANAGED_RUNTIME, "Apply a previously reviewed patch after one-time approval validation.", frozenset({"patch_id", "approval_token"}), frozenset(), _patch_args),
    ToolDefinition("generate_report", PermissionTier.WORKSPACE, "Generate a Markdown or JSON task report.", frozenset(), frozenset({"format"}), _report_args),
]

CATALOGUE: dict[str, ToolDefinition] = {item.tool_id: item for item in _DEFINITIONS}

# Deliberately not registered: arbitrary shell, Python eval, sudo, package installation,
# paths outside the workspace, and physical robot actuator commands such as /cmd_vel.
BLOCKED_TOOL_IDS = frozenset({"run_shell", "exec", "sudo", "install_system_package", "publish_cmd_vel", "control_physical_robot"})


def get_tool(tool_id: str) -> ToolDefinition:
    """Return a tool definition only when its ID is explicitly registered."""
    if tool_id in BLOCKED_TOOL_IDS:
        raise ToolCatalogueError(f"tool '{tool_id}' is permanently blocked by the MVP policy")
    try:
        return CATALOGUE[tool_id]
    except KeyError as exc:
        raise ToolCatalogueError(f"unknown tool_id: {tool_id}") from exc


def validate_tool_request(request: ToolRequest) -> ToolDefinition:
    """Validate tool identity, exact argument keys, and domain-specific values."""
    definition = get_tool(request.tool_id)
    supplied = set(request.arguments)
    missing = definition.required_arguments - supplied
    extra = supplied - definition.allowed_arguments
    if missing:
        raise ToolCatalogueError(f"missing required arguments: {', '.join(sorted(missing))}")
    if extra:
        raise ToolCatalogueError(f"unsupported arguments: {', '.join(sorted(extra))}")
    definition.validator(request.arguments)
    return definition


# =========================================================================
# Legacy Terminal Card Reference Helpers (Display Only)
# =========================================================================
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

        entry = COMMAND_DEFINITIONS[command_id]
        safe_kwargs = {k: cls.sanitize_arg(str(v)) for k, v in kwargs.items()}
        formatted_cmd = entry["template"].format(**safe_kwargs)

        return CommandItem(
            command_id=command_id,
            raw_command=formatted_cmd,
            purpose=entry["purpose"],
            expected_observation=entry["expected"]
        )
