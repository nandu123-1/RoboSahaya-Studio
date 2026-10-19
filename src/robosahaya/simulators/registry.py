"""Simulator Adapter Registry for RoboSahaya Studio.

Manages registration and dynamic resolution of simulation backends.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from robosahaya.robot_registry import RobotModel
from robosahaya.simulators.base import SimulatorAdapter
from robosahaya.simulators.gazebo_adapter import GazeboAdapter
from robosahaya.simulators.mujoco_adapter import MujocoAdapter


class SimulatorAdapterRegistry:
    """Registry maintaining available simulator adapters."""

    def __init__(self, workspace_manager: Optional[Any] = None):
        self._adapters: Dict[str, SimulatorAdapter] = {}
        self.register_adapter(GazeboAdapter(workspace_manager=workspace_manager))
        self.register_adapter(MujocoAdapter())

    def register_adapter(self, adapter: SimulatorAdapter) -> None:
        """Register an adapter instance."""
        self._adapters[adapter.name] = adapter

    def get_adapter(self, name: str) -> Optional[SimulatorAdapter]:
        """Look up an adapter by name."""
        return self._adapters.get(name)

    def list_adapters(self) -> List[SimulatorAdapter]:
        """List all registered adapters."""
        return list(self._adapters.values())

    def select_adapter(
        self,
        robot_model: RobotModel,
        requested_simulator: Optional[str] = None,
    ) -> Tuple[Optional[SimulatorAdapter], Dict[str, Any]]:
        """
        Dynamically select the compatible simulator adapter for the given robot model.
        Returns (adapter, resolution_details).
        """
        target_sim = requested_simulator or robot_model.default_simulator

        # Check target simulator first
        adapter = self.get_adapter(target_sim)
        if adapter:
            compat = adapter.inspect_compatibility(robot_model)
            if compat["compatible"]:
                return adapter, {
                    "selected": adapter.name,
                    "display_name": adapter.display_name,
                    "reason": f"Selected default/requested simulator '{adapter.display_name}'.",
                    "compatibility": compat,
                }
            else:
                # If target simulator is not compatible, try fallback simulators
                for alt_name in robot_model.simulator_support:
                    if alt_name == target_sim:
                        continue
                    alt_adapter = self.get_adapter(alt_name)
                    if alt_adapter:
                        alt_compat = alt_adapter.inspect_compatibility(robot_model)
                        if alt_compat["compatible"]:
                            return alt_adapter, {
                                "selected": alt_adapter.name,
                                "display_name": alt_adapter.display_name,
                                "reason": f"Target simulator '{target_sim}' was incompatible ({compat['reason']}); fell back to '{alt_adapter.display_name}'.",
                                "compatibility": alt_compat,
                            }

                return None, {
                    "selected": None,
                    "reason": f"Simulator '{target_sim}' is incompatible with '{robot_model.display_name}': {compat['reason']}",
                    "compatibility": compat,
                }

        # Simulator not registered
        return None, {
            "selected": None,
            "reason": f"Requested simulator '{target_sim}' is not registered in the studio.",
            "compatibility": {"compatible": False},
        }


_GLOBAL_SIMULATOR_REGISTRY: Optional[SimulatorAdapterRegistry] = None


def get_simulator_registry(workspace_manager: Optional[Any] = None) -> SimulatorAdapterRegistry:
    global _GLOBAL_SIMULATOR_REGISTRY
    if _GLOBAL_SIMULATOR_REGISTRY is None:
        _GLOBAL_SIMULATOR_REGISTRY = SimulatorAdapterRegistry(workspace_manager=workspace_manager)
    return _GLOBAL_SIMULATOR_REGISTRY
