"""Robot Capability Definition and Resolution System for RoboSahaya Studio.

Determines what physical and algorithmic operations a robot can genuinely perform,
preventing false claims and unauthorized robot substitutions.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from robosahaya.robot_registry import RobotModel, RobotRegistry, get_robot_registry


class CapabilityType(str, Enum):
    """Atomic robot physical and algorithmic capabilities."""
    MANIPULATION_ARM = "manipulation_arm"
    MANIPULATION_GRIPPER = "manipulation_gripper"
    MOBILE_WHEEL_VELOCITY = "mobile_wheel_velocity"
    MOBILE_NAVIGATION = "mobile_navigation"
    DRONE_FLIGHT = "drone_flight"
    LEGGED_LOCOMOTION = "legged_locomotion"
    PERCEPTION_RGB = "perception_rgb"
    PERCEPTION_DEPTH = "perception_depth"
    PERCEPTION_LIDAR = "perception_lidar"
    FORCE_TORQUE_SENSING = "force_torque_sensing"


class RobotCapability(BaseModel):
    """Specification of an individual capability for a robot."""
    model_config = ConfigDict(extra="ignore")

    capability_type: CapabilityType
    supported: bool
    prerequisites: List[str] = Field(default_factory=list)
    required_controllers: List[str] = Field(default_factory=list)
    supported_adapters: List[str] = Field(default_factory=list)
    control_interface: str = ""
    verification_tests: List[str] = Field(default_factory=list)
    limitation_reason: Optional[str] = None


class RobotCapabilityResolver:
    """Evaluates and resolves verified physical capabilities for robot models."""

    def __init__(self, registry: Optional[RobotRegistry] = None):
        self.registry = registry or get_robot_registry()

    def resolve_capabilities(self, robot_model_or_id: RobotModel | str) -> Dict[CapabilityType, RobotCapability]:
        """Resolve full capability mapping for a robot model."""
        if isinstance(robot_model_or_id, str):
            model = self.registry.get_model(robot_model_or_id)
            if not model:
                return self._unknown_capabilities(robot_model_or_id)
        else:
            model = robot_model_or_id

        family = (model.family or "").lower()
        model_id = model.id.lower()

        caps: Dict[CapabilityType, RobotCapability] = {}

        # 1. Robotic Arms (Franka Panda, SO-101, UR5e, etc.)
        if family == "robotic arm":
            caps[CapabilityType.MANIPULATION_ARM] = RobotCapability(
                capability_type=CapabilityType.MANIPULATION_ARM,
                supported=True,
                prerequisites=["Multi-DOF kinematic chain", "Joint position or trajectory controller"],
                required_controllers=["joint_trajectory_controller"],
                supported_adapters=["gazebo_harmonic"],
                control_interface="ros2_joint_trajectory",
                verification_tests=["test_joint_states_payload"],
            )
            # Parallel gripper
            has_gripper = model_id in ("franka_panda", "so_arm101", "robotic_arm_6dof", "so_arm100")
            caps[CapabilityType.MANIPULATION_GRIPPER] = RobotCapability(
                capability_type=CapabilityType.MANIPULATION_GRIPPER,
                supported=has_gripper,
                prerequisites=["Finger link geometries", "Gripper prismatic/revolute actuator"],
                required_controllers=["gripper_controller"] if has_gripper else [],
                supported_adapters=["gazebo_harmonic"],
                control_interface="ros2_gripper_action" if has_gripper else "",
                verification_tests=["test_gripper_actuation"] if has_gripper else [],
                limitation_reason=None if has_gripper else "Robot description does not configure an integrated gripper.",
            )
            caps[CapabilityType.PERCEPTION_RGB] = RobotCapability(
                capability_type=CapabilityType.PERCEPTION_RGB,
                supported=True,
                prerequisites=["Wrist or overhead RGB camera sensor plugin"],
                required_controllers=[],
                supported_adapters=["gazebo_harmonic"],
                control_interface="ros2_image_topic",
                verification_tests=["test_camera_stream_payload"],
            )
            # Mobile capabilities unsupported on fixed-base arms
            caps[CapabilityType.MOBILE_WHEEL_VELOCITY] = RobotCapability(
                capability_type=CapabilityType.MOBILE_WHEEL_VELOCITY,
                supported=False,
                limitation_reason="Fixed-base manipulator has no drive wheels.",
            )
            caps[CapabilityType.MOBILE_NAVIGATION] = RobotCapability(
                capability_type=CapabilityType.MOBILE_NAVIGATION,
                supported=False,
                limitation_reason="Fixed-base manipulator cannot execute mobile base navigation.",
            )
            caps[CapabilityType.DRONE_FLIGHT] = RobotCapability(
                capability_type=CapabilityType.DRONE_FLIGHT,
                supported=False,
                limitation_reason="Robotic arm is not an aerial vehicle.",
            )
            caps[CapabilityType.LEGGED_LOCOMOTION] = RobotCapability(
                capability_type=CapabilityType.LEGGED_LOCOMOTION,
                supported=False,
                limitation_reason="Robotic arm has no legged gait locomotion subsystem.",
            )

        # 2. Wheeled Robots (Robot Soccer Kit RSK, Differential Drive)
        elif family == "wheeled/mobile":
            is_rsk = model_id == "rsk"
            caps[CapabilityType.MOBILE_WHEEL_VELOCITY] = RobotCapability(
                capability_type=CapabilityType.MOBILE_WHEEL_VELOCITY,
                supported=True,
                prerequisites=["Wheel joints and velocity actuators"],
                required_controllers=["omnidirectional_wheel_velocity"] if is_rsk else ["diff_drive_controller"],
                supported_adapters=["mujoco"] if is_rsk else ["gazebo_harmonic"],
                control_interface="mujoco_ctrl_actuator" if is_rsk else "ros2_cmd_vel",
                verification_tests=["test_rsk_wheel_actuation"] if is_rsk else ["test_diff_drive_cmd_vel"],
            )
            caps[CapabilityType.MOBILE_NAVIGATION] = RobotCapability(
                capability_type=CapabilityType.MOBILE_NAVIGATION,
                supported=not is_rsk,  # Diff-drive has Nav2 / odom; RSK is low-level omni velocity
                prerequisites=["Odometry feedback", "Costmap LiDAR sensors", "Path follower"],
                required_controllers=["diff_drive_controller"],
                supported_adapters=["gazebo_harmonic"],
                control_interface="ros2_nav2_navigate_to_pose",
                verification_tests=["test_laser_scan_payload"],
                limitation_reason="RSK is configured for direct velocity control; Nav2 stack not active." if is_rsk else None,
            )
            caps[CapabilityType.PERCEPTION_LIDAR] = RobotCapability(
                capability_type=CapabilityType.PERCEPTION_LIDAR,
                supported=not is_rsk,
                prerequisites=["2D planar LiDAR sensor"],
                supported_adapters=["gazebo_harmonic"],
                control_interface="ros2_laser_scan_topic",
                verification_tests=["test_laser_scan_payload"],
                limitation_reason="RSK does not bundle on-board LiDAR (uses overhead vision)." if is_rsk else None,
            )
            caps[CapabilityType.MANIPULATION_ARM] = RobotCapability(
                capability_type=CapabilityType.MANIPULATION_ARM,
                supported=False,
                limitation_reason=f"Wheeled mobile robot '{model.display_name}' has no manipulator arm.",
            )
            caps[CapabilityType.MANIPULATION_GRIPPER] = RobotCapability(
                capability_type=CapabilityType.MANIPULATION_GRIPPER,
                supported=False,
                limitation_reason=f"Wheeled mobile robot '{model.display_name}' has no object gripper.",
            )
            caps[CapabilityType.DRONE_FLIGHT] = RobotCapability(
                capability_type=CapabilityType.DRONE_FLIGHT,
                supported=False,
                limitation_reason="Ground vehicle cannot fly.",
            )
            caps[CapabilityType.LEGGED_LOCOMOTION] = RobotCapability(
                capability_type=CapabilityType.LEGGED_LOCOMOTION,
                supported=False,
                limitation_reason="Wheeled robot does not use legged locomotion.",
            )

        # 3. Drones / Aerial
        elif family == "drone/aerial":
            caps[CapabilityType.DRONE_FLIGHT] = RobotCapability(
                capability_type=CapabilityType.DRONE_FLIGHT,
                supported=False,
                prerequisites=["Rotor dynamics plugin", "PX4 / SITL flight controller interface"],
                limitation_reason="Drone flight controller and SITL bridge are not integrated in this studio installation. Blocked.",
            )
            caps[CapabilityType.MANIPULATION_ARM] = RobotCapability(
                capability_type=CapabilityType.MANIPULATION_ARM,
                supported=False,
                limitation_reason="Aerial drone has no physical manipulation arm.",
            )
            caps[CapabilityType.MANIPULATION_GRIPPER] = RobotCapability(
                capability_type=CapabilityType.MANIPULATION_GRIPPER,
                supported=False,
                limitation_reason="Aerial drone has no object gripper.",
            )
            caps[CapabilityType.MOBILE_WHEEL_VELOCITY] = RobotCapability(
                capability_type=CapabilityType.MOBILE_WHEEL_VELOCITY,
                supported=False,
                limitation_reason="Drone has no ground wheels.",
            )

        # 4. Quadrupeds and Humanoids
        elif family in ("quadruped", "humanoid/biped"):
            caps[CapabilityType.LEGGED_LOCOMOTION] = RobotCapability(
                capability_type=CapabilityType.LEGGED_LOCOMOTION,
                supported=False,
                prerequisites=["WBC / MPC locomotion controller", "Ground contact sensors"],
                limitation_reason=f"Legged gait controller for '{model.display_name}' is not integrated in this studio runtime. Blocked.",
            )
            caps[CapabilityType.MANIPULATION_ARM] = RobotCapability(
                capability_type=CapabilityType.MANIPULATION_ARM,
                supported=False,
                limitation_reason=f"Locomotion model '{model.display_name}' does not configure arm manipulation in this studio profile.",
            )
            caps[CapabilityType.MOBILE_WHEEL_VELOCITY] = RobotCapability(
                capability_type=CapabilityType.MOBILE_WHEEL_VELOCITY,
                supported=False,
                limitation_reason="Legged robot has no drive wheels.",
            )

        # 5. Gripper / End-effector alone
        elif family == "gripper/end effector":
            caps[CapabilityType.MANIPULATION_GRIPPER] = RobotCapability(
                capability_type=CapabilityType.MANIPULATION_GRIPPER,
                supported=True,
                prerequisites=["Prismatic finger joints", "Gripper command topic"],
                supported_adapters=["gazebo_harmonic"],
                control_interface="ros2_gripper_action",
            )
            caps[CapabilityType.MANIPULATION_ARM] = RobotCapability(
                capability_type=CapabilityType.MANIPULATION_ARM,
                supported=False,
                limitation_reason="Standalone gripper has no multi-axis arm.",
            )

        else:
            return self._unknown_capabilities(model.id)

        return caps

    def check_capabilities(
        self,
        robot_id: str,
        required_caps: List[CapabilityType | str]
    ) -> Tuple[bool, List[str], List[str]]:
        """Verify if a robot satisfies all requested capabilities."""
        caps_map = self.resolve_capabilities(robot_id)
        satisfied: List[str] = []
        missing_reasons: List[str] = []

        for req in required_caps:
            c_type = req if isinstance(req, CapabilityType) else CapabilityType(str(req))
            cap = caps_map.get(c_type)
            if cap and cap.supported:
                satisfied.append(c_type.value)
            else:
                reason = cap.limitation_reason if cap else f"Capability '{c_type.value}' is unknown or not configured."
                missing_reasons.append(reason)

        return (len(missing_reasons) == 0, satisfied, missing_reasons)

    get_capabilities = resolve_capabilities

    def _unknown_capabilities(self, robot_id: str) -> Dict[CapabilityType, RobotCapability]:
        res = {}
        for ct in CapabilityType:
            res[ct] = RobotCapability(
                capability_type=ct,
                supported=False,
                limitation_reason=f"Robot '{robot_id}' has no registered capability verification in the studio.",
            )
        return res
