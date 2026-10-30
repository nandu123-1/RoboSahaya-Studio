"""Base interface for Simulator Adapters in RoboSahaya Studio.

Defines the contract for simulator backends (e.g. Gazebo Harmonic, MuJoCo),
enabling decoupled multi-simulator verification and execution.
Includes the SimulatorTaskController contract for runtime skill execution,
state observation, and predicate verification.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from robosahaya.robot_registry import RobotModel
from robosahaya.tasks.contracts import (
    ActionResult,
    SimulatorTaskController,
    SkillStatus,
)


class SimulatorAdapter(SimulatorTaskController, ABC):
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

    # =========================================================================
    # SimulatorTaskController implementation defaults (Agent B will specialize)
    # =========================================================================

    def execute_skill(
        self,
        skill_id: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        """Execute an atomic skill on the simulation runtime.
        
        Subclasses (Gazebo, MuJoCo) specialize this with actual control APIs.
        """
        return ActionResult(
            skill_id=skill_id,
            status=SkillStatus.BLOCKED,
            success=False,
            details=f"Runtime skill execution '{skill_id}' not implemented for {self.name}",
            error_reason=f"Adapter {self.name} has not implemented execute_skill"
        )

    def observe_state(
        self,
        query: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query simulator state (joints, objects, sensor telemetry).
        
        Subclasses specialize this with actual simulation state inspection.
        """
        return {}

    def verify_predicate(
        self,
        predicate_type: str,
        spec: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Verify an environment predicate or placement condition in simulation.
        
        Subclasses specialize this with physical verification checks.
        """
        return False, f"Predicate verification '{predicate_type}' not implemented for {self.name}"
