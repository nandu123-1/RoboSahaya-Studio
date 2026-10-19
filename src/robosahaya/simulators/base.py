"""Base interface for Simulator Adapters in RoboSahaya Studio.

Defines the contract for simulator backends (e.g. Gazebo Harmonic, MuJoCo),
enabling decoupled multi-simulator verification and execution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from robosahaya.robot_registry import RobotModel


class SimulatorAdapter(ABC):
    """Abstract base class for all simulation runtime adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique adapter name (e.g. 'gazebo_harmonic', 'mujoco')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display name."""
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> Tuple[str, ...]:
        """Model formats supported by this adapter (e.g. ('URDF', 'XACRO'), ('MJCF', 'XML'))."""
        pass

    @abstractmethod
    def probe_environment(self) -> Dict[str, Any]:
        """Probe the host environment for binaries, packages, and hardware acceleration."""
        pass

    @abstractmethod
    def inspect_compatibility(self, robot_model: RobotModel) -> Dict[str, Any]:
        """Check if this adapter can run the given robot model."""
        pass

    @abstractmethod
    def prepare_model(self, robot_model: RobotModel, target_dir: Path) -> Dict[str, Any]:
        """Materialize or link model descriptions and asset meshes into the workspace."""
        pass

    @abstractmethod
    def validate_model(self, model_path: Path) -> Dict[str, Any]:
        """Validate model syntax, mesh references, and physical integrity."""
        pass

    @abstractmethod
    def build_runtime(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build runtime packages, configuration, or launch wrappers."""
        pass

    @abstractmethod
    def launch_headless(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Launch headless simulation for validation."""
        pass

    @abstractmethod
    def run_verification(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute physics stepping, controller verification, and sensor stream inspection."""
        pass

    @abstractmethod
    def launch_gui(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Launch user-facing graphical simulator viewer."""
        pass

    @abstractmethod
    def stop(self, context: Dict[str, Any]) -> None:
        """Stop running simulation processes."""
        pass

    @abstractmethod
    def cleanup(self, context: Dict[str, Any]) -> None:
        """Clean up temporary resources."""
        pass
