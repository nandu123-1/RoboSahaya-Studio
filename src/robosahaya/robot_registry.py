"""Universal Robot Registry for RoboSahaya Studio.

Single source of truth for robot models, physical capabilities, formats,
and simulator compatibility.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class RobotModel:
    """Canonical model descriptor supporting multi-simulator architecture."""
    id: str
    display_name: str
    family: str
    aliases: Tuple[str, ...] = ()
    description_module: Optional[str] = None
    formats: Tuple[str, ...] = ("URDF",)
    license: str = "Check upstream source"
    source_url: str = "https://github.com/robot-descriptions/robot_descriptions.py"
    runtime_template_id: Optional[str] = None
    runtime_status: str = "description_only"
    notes: str = ""

    # Multi-simulator and capability extensions
    default_simulator: str = "gazebo_harmonic"
    simulator_support: Tuple[str, ...] = ("gazebo_harmonic",)
    support_status: str = "description_only"
    source_info: Dict[str, Any] = field(default_factory=dict)
    description_info: Dict[str, Any] = field(default_factory=dict)
    controllers: Tuple[str, ...] = ()
    sensors: Tuple[str, ...] = ()
    compatible_worlds: Tuple[str, ...] = ("obstacle_world", "tabletop_world")
    required_dependencies: Tuple[str, ...] = ()
    verification_evidence: Dict[str, Any] = field(default_factory=dict)
    known_limitations: str = ""

    @property
    def runtime_ready(self) -> bool:
        return self.runtime_status == "runtime_ready" or self.support_status in (
            "simulation_ready", "control_ready", "task_verified"
        )

    def is_simulator_compatible(self, simulator_name: str) -> bool:
        return simulator_name in self.simulator_support


def _normalize_alias(alias: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (alias or "").lower()).strip()


class RobotRegistry:
    """Central registry maintaining canonical robot definitions."""

    def __init__(self):
        self._models: Dict[str, RobotModel] = {}
        self._alias_map: Dict[str, str] = {}
        self._register_default_models()

    def register_model(self, model: RobotModel) -> None:
        """Register a robot model with strict validation."""
        if not model.id or not isinstance(model.id, str):
            raise ValueError("RobotModel must have a non-empty string 'id'.")
        if model.id in self._models:
            raise ValueError(f"Duplicate robot model canonical ID: '{model.id}'")

        self._models[model.id] = model

        # Register canonical ID as primary alias
        norm_id = _normalize_alias(model.id)
        if norm_id:
            self._alias_map[norm_id] = model.id

        # Register display name
        norm_disp = _normalize_alias(model.display_name)
        if norm_disp:
            self._alias_map[norm_disp] = model.id

        # Register aliases
        for alias in model.aliases:
            norm_alias = _normalize_alias(alias)
            if norm_alias and norm_alias not in self._alias_map:
                self._alias_map[norm_alias] = model.id

    def get_model(self, identifier_or_alias: str) -> Optional[RobotModel]:
        """Look up by canonical ID or registered alias."""
        if not identifier_or_alias:
            return None
        # Direct lookup by ID
        if identifier_or_alias in self._models:
            return self._models[identifier_or_alias]

        # Normalized lookup
        norm = _normalize_alias(identifier_or_alias)
        if norm in self._alias_map:
            return self._models[self._alias_map[norm]]

        # Runtime template ID lookup
        for model in self._models.values():
            if model.runtime_template_id == identifier_or_alias:
                return model

        # Substring alias search
        for alias_norm, model_id in self._alias_map.items():
            if norm == alias_norm:
                return self._models[model_id]

        return None

    def list_models(self, family: str = "All") -> List[RobotModel]:
        models = list(self._models.values())
        if family and family != "All":
            models = [m for m in models if m.family.lower() == family.lower()]
        return sorted(models, key=lambda m: m.display_name)

    def _register_default_models(self) -> None:
        """Register working and catalogued models."""
        # 1. Differential Drive Robot
        self.register_model(RobotModel(
            id="differential_drive",
            display_name="RoboSahaya Differential Drive",
            family="wheeled/mobile",
            aliases=("differential drive", "diff drive", "wheeled base", "sahaya bot", "sahaya_bot"),
            formats=("URDF", "XACRO"),
            license="Apache-2.0",
            source_url="internal",
            runtime_template_id="differential_drive",
            runtime_status="runtime_ready",
            support_status="simulation_ready",
            default_simulator="gazebo_harmonic",
            simulator_support=("gazebo_harmonic",),
            controllers=("diff_drive_controller",),
            sensors=("2d_lidar", "rgb_camera", "odometry"),
            compatible_worlds=("obstacle_world",),
            required_dependencies=("ros_gz_sim", "ros_gz_bridge"),
            notes="Bundled verified mobile robot template.",
        ))

        # 2. Franka Emika Panda
        self.register_model(RobotModel(
            id="franka_panda",
            display_name="Franka Emika Panda",
            family="robotic arm",
            aliases=("franka panda", "franka emika panda", "panda arm", "panda robot", "franka"),
            description_module="panda_description",
            formats=("URDF", "XACRO", "DAE"),
            license="Apache-2.0",
            runtime_template_id="robotic_arm_6dof",
            runtime_status="runtime_ready",
            support_status="simulation_ready",
            default_simulator="gazebo_harmonic",
            simulator_support=("gazebo_harmonic",),
            controllers=("joint_state_broadcaster", "joint_trajectory_controller"),
            sensors=("joint_encoders", "wrist_camera"),
            compatible_worlds=("tabletop_world",),
            required_dependencies=("ros_gz_sim", "ros_gz_bridge"),
            notes="Bundled verified Franka Panda arm template on tabletop.",
        ))

        # 3. SO-101 6-DOF Arm (LeRobot)
        self.register_model(RobotModel(
            id="so_arm101",
            display_name="SO-ARM101 / SO-101",
            family="robotic arm",
            aliases=("so-101", "so101", "so arm 101", "so-arm101", "lerobot so101", "so 101"),
            description_module="so_arm101_description",
            formats=("URDF", "XACRO", "STL"),
            license="Apache-2.0",
            runtime_template_id="so_arm101",
            runtime_status="runtime_ready",
            support_status="simulation_ready",
            default_simulator="gazebo_harmonic",
            simulator_support=("gazebo_harmonic",),
            controllers=("feetech_bus_joint_controller",),
            sensors=("joint_encoders", "wrist_camera"),
            compatible_worlds=("tabletop_world",),
            required_dependencies=("ros_gz_sim", "ros_gz_bridge"),
            notes="Bundled LeRobot / The Robot Studio 3D-printable follower arm template.",
        ))

        # 4. Robot Soccer Kit (RSK) — MuJoCo Target
        self.register_model(RobotModel(
            id="rsk",
            display_name="Robot Soccer Kit (RSK)",
            family="wheeled/mobile",
            aliases=("rsk", "robot soccer kit", "robot_soccer_kit", "rsk robot", "soccer robot"),
            description_module="rsk_mj_description",
            formats=("MJCF", "XML", "STL"),
            license="Apache-2.0",
            source_url="https://github.com/google-deepmind/mujoco_menagerie/tree/main/robot_soccer_kit",
            runtime_template_id=None,
            runtime_status="runtime_ready",
            support_status="simulation_ready",
            default_simulator="mujoco",
            simulator_support=("mujoco",),
            controllers=("omnidirectional_wheel_velocity",),
            sensors=("wheel_encoders",),
            compatible_worlds=("soccer_pitch", "obstacle_world"),
            required_dependencies=("mujoco",),
            source_info={
                "repository": "https://github.com/google-deepmind/mujoco_menagerie.git",
                "package": "robot_soccer_kit",
                "commit": "feadf76d42f8a2162426f7d226a3b539556b3bf5",
            },
            description_info={
                "format": "MJCF",
                "entrypoint": "robot_soccer_kit.xml",
                "relative_path": "robot_soccer_kit/robot_soccer_kit.xml",
            },
            notes="MuJoCo Menagerie Robot Soccer Kit omnidirectional mobile robot.",
        ))

        # 5. Skydio X2 Drone (Catalogued)
        self.register_model(RobotModel(
            id="skydio_x2",
            display_name="Skydio X2",
            family="drone/aerial",
            aliases=("skydio x2", "skydio", "sky dio x2"),
            description_module="skydio_x2_description",
            formats=("URDF", "MJCF"),
            license="Apache-2.0",
            support_status="description_only",
            runtime_status="description_only",
            default_simulator="gazebo_harmonic",
            simulator_support=("gazebo_harmonic", "mujoco"),
            notes="Aerial drone template; flight dynamics adapter not yet runtime verified.",
        ))

        # 6. SO-ARM100 (Catalogued description-only)
        self.register_model(RobotModel(
            id="so_arm100",
            display_name="SO-ARM100 / SO-100",
            family="robotic arm",
            aliases=("so-100", "so100", "so arm 100", "so-arm100"),
            description_module="so_arm100_description",
            formats=("URDF",),
            license="Apache-2.0",
            support_status="description_only",
            runtime_status="description_only",
            notes="Description available in library; use so_arm101 for verified simulation.",
        ))

        # 7. Universal Robots UR5e (Catalogued)
        self.register_model(RobotModel(
            id="ur5e",
            display_name="Universal Robots UR5e",
            family="robotic arm",
            aliases=("ur5e", "ur 5e", "universal robots ur5e"),
            description_module="ur5e_description",
            formats=("URDF", "MJCF"),
            license="BSD-3-Clause",
            support_status="description_only",
            runtime_status="description_only",
            default_simulator="gazebo_harmonic",
            simulator_support=("gazebo_harmonic", "mujoco"),
            notes="Industrial robotic arm; Gazebo ros2_control adapter not yet bundled.",
        ))

        # 8. Unitree Go2 Quadruped (Catalogued)
        self.register_model(RobotModel(
            id="unitree_go2",
            display_name="Unitree Go2",
            family="quadruped",
            aliases=("unitree go2", "go2", "go 2"),
            description_module="go2_description",
            formats=("URDF", "MJCF"),
            license="BSD-3-Clause",
            support_status="description_only",
            runtime_status="description_only",
            default_simulator="mujoco",
            simulator_support=("mujoco", "gazebo_harmonic"),
            notes="Quadruped robot; legged locomotion controller not yet runtime verified.",
        ))


# Global Singleton Registry
_GLOBAL_REGISTRY: Optional[RobotRegistry] = None


def get_robot_registry() -> RobotRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = RobotRegistry()
    return _GLOBAL_REGISTRY
