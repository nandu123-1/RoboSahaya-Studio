"""Gazebo Harmonic Simulator Adapter for RoboSahaya Studio.

Wraps Gazebo Harmonic, ros_gz_sim, and ROS 2 Jazzy workspace execution.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from robosahaya.robot_registry import RobotModel
from robosahaya.simulators.base import SimulatorAdapter


class GazeboAdapter(SimulatorAdapter):
    """Simulator adapter for Gazebo Harmonic with ROS 2 Jazzy."""

    def __init__(self, workspace_manager: Optional[Any] = None):
        self._ws_mgr = workspace_manager

    @property
    def name(self) -> str:
        return "gazebo_harmonic"

    @property
    def display_name(self) -> str:
        return "Gazebo Harmonic (ROS 2 Jazzy)"

    @property
    def supported_formats(self) -> Tuple[str, ...]:
        return ("URDF", "XACRO", "SDF")

    def probe_environment(self) -> Dict[str, Any]:
        """Probe for Gazebo Harmonic and ROS 2 tools on host."""
        gz_path = shutil.which("gz")
        ros2_path = shutil.which("ros2")
        colcon_path = shutil.which("colcon")

        rclpy_ok = False
        try:
            import rclpy
            rclpy_ok = True
        except ImportError:
            pass

        available = bool(gz_path and ros2_path)
        return {
            "available": available,
            "simulator": "gazebo_harmonic",
            "gz_binary": gz_path,
            "ros2_binary": ros2_path,
            "colcon_binary": colcon_path,
            "rclpy_available": rclpy_ok,
            "display": os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"),
        }

    def inspect_compatibility(self, robot_model: RobotModel) -> Dict[str, Any]:
        """Check compatibility with Gazebo Harmonic."""
        env = self.probe_environment()
        missing_deps = []
        if not env["gz_binary"]:
            missing_deps.append("gz (Gazebo Harmonic)")
        if not env["ros2_binary"]:
            missing_deps.append("ros2 (ROS 2 Jazzy)")

        format_match = any(fmt.upper() in self.supported_formats for fmt in robot_model.formats)
        sim_support = "gazebo_harmonic" in robot_model.simulator_support

        compatible = format_match and sim_support and len(missing_deps) == 0
        reasons = []
        if not format_match:
            reasons.append(f"Model formats {robot_model.formats} not supported by Gazebo adapter {self.supported_formats}.")
        if not sim_support:
            reasons.append(f"Model '{robot_model.id}' does not declare Gazebo Harmonic support.")
        if missing_deps:
            reasons.append(f"Missing host binaries: {', '.join(missing_deps)}.")

        return {
            "compatible": compatible,
            "format_supported": format_match,
            "adapter_supported": sim_support,
            "missing_dependencies": missing_deps,
            "reason": "; ".join(reasons) if reasons else "Compatible with Gazebo Harmonic.",
        }

    def prepare_model(self, robot_model: RobotModel, target_dir: Path) -> Dict[str, Any]:
        """Prepare model workspace using template or assets."""
        if not self._ws_mgr:
            return {"success": False, "error": "No WorkspaceManager configured."}

        template_id = robot_model.runtime_template_id or robot_model.id
        try:
            ws = self._ws_mgr.assemble_simulation_template(
                task_id=target_dir.name,
                template_id=template_id,
            )
            return {"success": True, "workspace_dir": str(ws)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def validate_model(self, model_path: Path) -> Dict[str, Any]:
        """Validate workspace files using WorkspaceManager validator."""
        if not self._ws_mgr:
            return {"valid": False, "errors": ["No WorkspaceManager configured."]}
        return self._ws_mgr.validate_workspace_files(model_path)

    def build_runtime(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build colcon workspace."""
        if not self._ws_mgr:
            return {"success": False, "error": "No WorkspaceManager configured."}
        ws_dir = Path(context["workspace_dir"])
        return self._ws_mgr.build_workspace(ws_dir, task_id=context.get("task_id"))

    def launch_headless(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Launch Gazebo headless simulation."""
        if not self._ws_mgr:
            return {"success": False, "error": "No WorkspaceManager configured."}
        ws_dir = Path(context["workspace_dir"])
        task_id = context.get("task_id", ws_dir.name)
        return self._ws_mgr.launch_workspace(task_id, ws_dir, headless=True)

    def run_verification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verification is handled by ROS 2 topic sampling."""
        return {"success": True, "message": "Gazebo topic verification ready."}

    def launch_gui(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Open Gazebo GUI."""
        if not self._ws_mgr:
            return {"success": False, "error": "No WorkspaceManager configured."}
        ws_dir = Path(context.get("workspace_dir", ""))
        return self._ws_mgr.open_gazebo_gui(ws_dir)

    def stop(self, context: Dict[str, Any]) -> None:
        if self._ws_mgr:
            task_id = context.get("task_id")
            if task_id:
                self._ws_mgr.stop_task_runtime(task_id)

    def cleanup(self, context: Dict[str, Any]) -> None:
        self.stop(context)
