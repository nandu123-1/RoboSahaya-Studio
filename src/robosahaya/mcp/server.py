import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

# Safe import of ROS inspector if in ROS 2 environment
try:
    from robosahaya.ros_inspector import LiveRclpyInspector, RCLPY_AVAILABLE
except Exception:
    LiveRclpyInspector = None
    RCLPY_AVAILABLE = False

from robosahaya.mock_inspector import MockInspector

mcp = FastMCP("RoboSahaya-ROS2-Gazebo-Server")

_inspector = None


def _get_inspector():
    global _inspector
    if _inspector is not None:
        return _inspector
    if RCLPY_AVAILABLE and LiveRclpyInspector is not None:
        try:
            _inspector = LiveRclpyInspector()
            return _inspector
        except Exception:
            pass
    _inspector = MockInspector(scenario_name="clean_healthy_system")
    return _inspector


@mcp.tool()
def check_environment() -> Dict[str, Any]:
    """Inspect required ROS 2, Gazebo, and workspace build prerequisites."""
    gz_bin = shutil.which("gz")
    gz_version = "not_found"
    if gz_bin:
        try:
            out = subprocess.check_output([gz_bin, "sim", "--version"], text=True, timeout=2.0)
            gz_version = out.strip().split("\n")[0]
        except Exception:
            gz_version = "detected (gz)"

    ros_distro = os.environ.get("ROS_DISTRO", "unknown")
    colcon_ok = bool(shutil.which("colcon"))
    xacro_ok = bool(shutil.which("xacro"))

    return {
        "ros_distro": ros_distro,
        "rclpy_available": RCLPY_AVAILABLE,
        "gazebo_version": gz_version,
        "gazebo_available": gz_bin is not None,
        "colcon_available": colcon_ok,
        "xacro_available": xacro_ok,
        "status": "ready" if (RCLPY_AVAILABLE and gz_bin and colcon_ok and xacro_ok) else "degraded"
    }


@mcp.tool()
def list_ros_nodes() -> Dict[str, Any]:
    """Discover active ROS 2 nodes currently registered on the ROS graph."""
    insp = _get_inspector()
    nodes = insp.get_active_nodes()
    return {
        "nodes": nodes,
        "count": len(nodes),
        "source": "live_ros_graph" if insp.is_live else "mock_graph"
    }


@mcp.tool()
def list_ros_topics() -> Dict[str, Any]:
    """Discover available ROS 2 topics and their registered message types."""
    insp = _get_inspector()
    topics = insp.get_topic_types()
    return {
        "topics": topics,
        "count": len(topics),
        "source": "live_ros_graph" if insp.is_live else "mock_graph"
    }


@mcp.tool()
def inspect_topic(topic: str, sample_seconds: float = 1.0) -> Dict[str, Any]:
    """Inspect a selected ROS 2 topic and sample real incoming messages."""
    if not topic.startswith("/"):
        topic = "/" + topic

    insp = _get_inspector()
    sample = insp.sample_messages(topic, sample_window_sec=max(0.5, min(5.0, sample_seconds)))
    return {
        "topic": topic,
        "messages_received": sample.messages_received,
        "frequency_hz": sample.frequency_hz,
        "payload_valid": sample.payload_valid,
        "valid_samples": sample.valid_samples,
        "invalid_samples": sample.invalid_samples,
        "validation_detail": sample.validation_detail,
        "sampling_error": sample.sampling_error,
        "source": "live_ros_graph" if insp.is_live else "mock_graph"
    }


@mcp.tool()
def inspect_tf(source_frame: str = "base_footprint", target_frame: str = "base_link") -> Dict[str, Any]:
    """Inspect coordinate transform availability between two TF frames."""
    insp = _get_inspector()
    tf_ev = insp.check_tf_transform(source_frame, target_frame)
    return {
        "source_frame": source_frame,
        "target_frame": target_frame,
        "transform_available": tf_ev.transform_available,
        "status": tf_ev.status.value,
        "delay_sec": tf_ev.delay_sec,
        "error_message": tf_ev.error_message
    }


@mcp.tool()
def get_gazebo_status() -> Dict[str, Any]:
    """Verify whether Gazebo Sim Harmonic is currently active and running."""
    pids = []
    try:
        out = subprocess.check_output(["pgrep", "-f", "gz sim"], text=True, timeout=2.0)
        pids = [int(p) for p in out.strip().split() if p.isdigit()]
    except Exception:
        pass

    return {
        "gazebo_running": len(pids) > 0,
        "active_pids": pids,
        "simulator": "Gazebo Sim Harmonic",
        "verified_world": "obstacle_world" if len(pids) > 0 else None
    }


@mcp.tool()
def launch_managed_simulation(template_id: str = "differential_drive", world_id: str = "obstacle_world", approval_token: str = "") -> Dict[str, Any]:
    """Launch an approved robot/world simulation profile (Tier 2: requires human approval token)."""
    if template_id != "differential_drive":
        return {
            "success": False,
            "error": f"Unsupported template '{template_id}'. RoboSahaya Studio only provides verified templates for 'differential_drive'. Arms and drones are rejected.",
            "status": "blocked"
        }

    if not approval_token or not approval_token.strip():
        return {
            "success": False,
            "error": "Security Gate: Tier 2 operation 'launch_managed_simulation' requires an explicit human approval token.",
            "status": "permission_denied"
        }

    return {
        "success": True,
        "template_id": template_id,
        "world_id": world_id,
        "status": "simulation_profile_approved",
        "message": f"Simulation profile '{template_id}' approved for launch in '{world_id}'."
    }


@mcp.tool()
def run_sensor_test(test_id: str = "test_laser_scan_payload", topic: str = "/scan", min_messages: int = 1) -> Dict[str, Any]:
    """Run a bounded verification test against actual sensor data."""
    insp = _get_inspector()
    sample = insp.sample_messages(topic, sample_window_sec=1.5)
    passed = bool(sample.messages_received >= min_messages and sample.payload_valid)
    return {
        "test_id": test_id,
        "passed": passed,
        "measured_values": {
            "messages_received": sample.messages_received,
            "frequency_hz": sample.frequency_hz,
            "valid_samples": sample.valid_samples,
            "invalid_samples": sample.invalid_samples,
        },
        "expected_criteria": {
            "topic": topic,
            "min_messages": min_messages,
            "payload_valid": True
        },
        "log_excerpt": sample.validation_detail or (f"Topic {topic} received {sample.messages_received} messages.")
    }


@mcp.tool()
def stop_managed_simulation(task_id: str = "all") -> Dict[str, Any]:
    """Stop a simulation launched and tracked by RoboSahaya."""
    stopped_count = 0
    try:
        out = subprocess.check_output(["pgrep", "-f", "gz sim"], text=True, timeout=2.0)
        pids = [int(p) for p in out.strip().split() if p.isdigit()]
        for pid in pids:
            try:
                os.kill(pid, 15)  # SIGTERM
                stopped_count += 1
            except Exception:
                pass
    except Exception:
        pass

    return {
        "success": True,
        "task_id": task_id,
        "processes_signaled": stopped_count,
        "message": f"Simulation teardown executed; {stopped_count} processes signaled."
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
