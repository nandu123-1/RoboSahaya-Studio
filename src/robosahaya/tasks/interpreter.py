"""Natural-Language Task Interpreter and Planner for RoboSahaya Studio.

Transforms unstructured engineering instructions into validated, capability-checked
Structured Task Specifications and action sequences.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from robosahaya.robot_registry import RobotModel, RobotRegistry, get_robot_registry
from .capabilities import CapabilityType, RobotCapabilityResolver
from .scene import SceneEntity, WorldScene, get_world_scene
from .specification import (
    ActionStep,
    PlacementPredicate,
    TaskSpecification,
    TaskType,
    VerificationCondition,
)


class TaskInterpreter:
    """Interprets natural language into structured multi-robot task specifications."""

    def __init__(self, registry: Optional[RobotRegistry] = None):
        self.registry = registry or get_robot_registry()
        self.cap_resolver = RobotCapabilityResolver(self.registry)

    def interpret_instruction(
        self,
        instruction: str,
        robot_id_override: Optional[str] = None,
        world_id_override: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> TaskSpecification:
        """Parse natural language instruction and construct a complete TaskSpecification."""
        task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        q_clean = (instruction or "").strip()
        q_lower = q_clean.lower()

        # 1. Infer Task Type
        task_type = self._classify_task_type(q_lower)

        # 2. Resolve or Infer Robot
        resolved_robot = self._resolve_robot(q_lower, task_type, robot_id_override)

        # 3. Resolve or Infer World
        world_id = self._resolve_world(q_lower, task_type, resolved_robot, world_id_override)
        scene = get_world_scene(world_id)

        # 4. Extract Specific Target Objects and Destinations
        target_entities, destination_entity, extra_params = self._ground_entities(q_clean, task_type, scene)

        # 5. Determine Required Capabilities, Sensors, Controllers
        req_caps, req_sensors, req_controllers = self._determine_prerequisites(task_type, extra_params)

        # 6. Validate Robot Compatibility and Capabilities
        is_supported = True
        rejection_reason = None
        known_limitations: List[str] = []

        if not resolved_robot:
            is_supported = False
            rejection_reason = "No compatible robot could be identified for this instruction."
        else:
            can_do, satisfied, missing = self.cap_resolver.check_capabilities(resolved_robot.id, req_caps)
            if not can_do:
                is_supported = False
                rejection_reason = f"Robot '{resolved_robot.display_name}' cannot execute this task: " + "; ".join(missing)
                known_limitations.extend(missing)

        # 7. Check Scene Prerequisites
        if is_supported and task_type == TaskType.MANIPULATION_PICK_PLACE:
            if not target_entities:
                is_supported = False
                rejection_reason = f"Target object to pick up could not be found in '{world_id}'."
            elif not destination_entity:
                is_supported = False
                rejection_reason = f"Destination plate or placement area could not be found in '{world_id}'."

        # 8. Synthesize Action Steps and Verification Predicates
        action_sequence: List[ActionStep] = []
        placement_predicate: Optional[PlacementPredicate] = None
        verif_conditions: List[VerificationCondition] = []

        if is_supported and resolved_robot:
            action_sequence, placement_predicate, verif_conditions = self._generate_action_sequence(
                task_type=task_type,
                robot=resolved_robot,
                scene=scene,
                target_entities=target_entities,
                destination_entity=destination_entity,
                extra_params=extra_params,
            )

        sim_adapter = resolved_robot.default_simulator if resolved_robot else "gazebo_harmonic"

        return TaskSpecification(
            task_id=task_id,
            original_instruction=q_clean,
            task_type=task_type,
            canonical_robot_id=resolved_robot.id if resolved_robot else "unknown",
            robot_model_version="v1.0.0",
            simulator_adapter=sim_adapter,
            world_id=world_id,
            target_objects=[e.id for e in target_entities],
            destination=destination_entity.id if destination_entity else None,
            required_capabilities=[c.value for c in req_caps],
            required_sensors=req_sensors,
            required_controllers=req_controllers,
            action_sequence=action_sequence,
            preconditions=[f"Robot {resolved_robot.id if resolved_robot else ''} model loaded", f"World {world_id} materialized"],
            intermediate_criteria={"min_steps": 10},
            placement_predicate=placement_predicate,
            verification_conditions=verif_conditions,
            final_criteria={"task_success": True},
            safety_constraints=["Keep joint velocities within safety limits", "Prevent ungrounded collision"],
            is_supported=is_supported,
            rejection_reason=rejection_reason,
            known_limitations=known_limitations,
        )

    def _classify_task_type(self, q: str) -> TaskType:
        if any(w in q for w in ("pick and place", "pick up", "place it", "grasp", "lift", "place on")):
            return TaskType.MANIPULATION_PICK_PLACE
        if any(w in q for w in ("wheel_speed", "wheel speed", "set wheel", "wheel1", "wheel2", "wheel3")):
            return TaskType.MOBILE_WHEEL_CONTROL
        if any(w in q for w in ("move to", "drive to", "navigate to", "stop 0.", "stop 1.")):
            return TaskType.MOBILE_NAVIGATION
        if any(w in q for w in ("fly", "takeoff", "land", "waypoint", "drone", "altitude", "quadcopter")):
            return TaskType.DRONE_FLIGHT
        if any(w in q for w in ("walk", "gait", "stand", "trot", "quadruped", "biped")):
            return TaskType.LEGGED_LOCOMOTION
        if any(w in q for w in ("camera", "picture", "capture", "scan", "inspect topic")):
            return TaskType.PERCEPTION_INSPECTION
        return TaskType.SIMULATION_LIFECYCLE

    def _resolve_robot(self, q: str, task_type: TaskType, override_id: Optional[str]) -> Optional[RobotModel]:
        if override_id:
            return self.registry.get_model(override_id)

        # Check for exact named robot references first
        for model in self.registry.list_models():
            # Check ID
            if re.search(r"\b" + re.escape(model.id) + r"\b", q):
                return model
            # Check aliases
            for alias in model.aliases:
                if len(alias) >= 3 and re.search(r"\b" + re.escape(alias) + r"\b", q):
                    return model

        # Fallback by task type default canonical robots
        if task_type == TaskType.MANIPULATION_PICK_PLACE:
            return self.registry.get_model("franka_panda")
        elif task_type == TaskType.MOBILE_WHEEL_CONTROL:
            # If RSK-specific wheel name is mentioned
            if "wheel" in q or "rsk" in q:
                return self.registry.get_model("rsk")
            return self.registry.get_model("differential_drive")
        elif task_type == TaskType.MOBILE_NAVIGATION:
            return self.registry.get_model("differential_drive")
        elif task_type == TaskType.DRONE_FLIGHT:
            return self.registry.get_model("skydio_x2")
        elif task_type == TaskType.LEGGED_LOCOMOTION:
            return self.registry.get_model("unitree_go2")

        return self.registry.get_model("differential_drive")

    def _resolve_world(
        self,
        q: str,
        task_type: TaskType,
        robot: Optional[RobotModel],
        override_world: Optional[str]
    ) -> str:
        if override_world:
            return override_world

        if task_type == TaskType.MANIPULATION_PICK_PLACE or (robot and robot.family == "robotic arm"):
            return "tabletop_world"
        if robot and robot.id == "rsk":
            return "soccer_pitch"
        if task_type == TaskType.MOBILE_NAVIGATION:
            return "obstacle_world"

        return "obstacle_world"

    def _ground_entities(
        self,
        instruction: str,
        task_type: TaskType,
        scene: WorldScene
    ) -> Tuple[List[SceneEntity], Optional[SceneEntity], Dict[str, Any]]:
        target_entities: List[SceneEntity] = []
        destination_entity: Optional[SceneEntity] = None
        extra_params: Dict[str, Any] = {}

        q_lower = instruction.lower()

        # 1. Manipulation Pick & Place Entity Grounding
        if task_type == TaskType.MANIPULATION_PICK_PLACE:
            # Find target object to pick
            target = scene.find_target_object(instruction)
            if not target:
                target = scene.find_target_object("")
            if target:
                target_entities.append(target)

            # Find destination
            dest = scene.find_destination(instruction)
            if not dest:
                dest = scene.find_destination("")
            if dest:
                destination_entity = dest

        # 2. Wheel Speed Extraction (e.g. "Set wheel1_speed to 5.44")
        elif task_type == TaskType.MOBILE_WHEEL_CONTROL:
            # Extract wheel name
            wheel_match = re.search(r"(wheel[1-4](?:_speed)?)", q_lower)
            wheel_name = wheel_match.group(1) if wheel_match else "wheel1_speed"
            if not wheel_name.endswith("_speed"):
                wheel_name = f"{wheel_name}_speed"
            extra_params["wheel_name"] = wheel_name

            # Extract numeric velocity (positive or negative float), ignoring wheel index
            cleaned_instruction = re.sub(r"wheel\s*\d(?:_speed)?", "", instruction, flags=re.IGNORECASE)
            vel_match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned_instruction)
            velocity = float(vel_match.group(0)) if vel_match else 5.44
            extra_params["velocity"] = velocity

        # 3. Mobile Navigation Grounding (e.g. "Move to the red box and stop 0.5 metres away")
        elif task_type == TaskType.MOBILE_NAVIGATION:
            target = scene.find_target_object(instruction)
            if target:
                target_entities.append(target)
            dist_match = re.search(r"stop\s+([0-9.]+)\s*(?:m|meter|metres)?", q_lower)
            extra_params["stop_distance_m"] = float(dist_match.group(1)) if dist_match else 0.5

        return target_entities, destination_entity, extra_params

    def _determine_prerequisites(
        self,
        task_type: TaskType,
        extra_params: Dict[str, Any]
    ) -> Tuple[List[CapabilityType], List[str], List[str]]:
        if task_type == TaskType.MANIPULATION_PICK_PLACE:
            return (
                [CapabilityType.MANIPULATION_ARM, CapabilityType.MANIPULATION_GRIPPER],
                ["wrist_camera", "joint_encoders"],
                ["joint_trajectory_controller", "gripper_controller"],
            )
        elif task_type == TaskType.MOBILE_WHEEL_CONTROL:
            return (
                [CapabilityType.MOBILE_WHEEL_VELOCITY],
                ["wheel_encoders"],
                ["omnidirectional_wheel_velocity"],
            )
        elif task_type == TaskType.MOBILE_NAVIGATION:
            return (
                [CapabilityType.MOBILE_NAVIGATION],
                ["2d_lidar", "odometry"],
                ["diff_drive_controller"],
            )
        elif task_type == TaskType.DRONE_FLIGHT:
            return (
                [CapabilityType.DRONE_FLIGHT],
                ["imu", "altimeter", "camera"],
                ["px4_flight_controller"],
            )
        elif task_type == TaskType.LEGGED_LOCOMOTION:
            return (
                [CapabilityType.LEGGED_LOCOMOTION],
                ["joint_encoders", "imu", "foot_contact_sensors"],
                ["wbc_gait_controller"],
            )

        return ([], [], [])

    def _generate_action_sequence(
        self,
        task_type: TaskType,
        robot: RobotModel,
        scene: WorldScene,
        target_entities: List[SceneEntity],
        destination_entity: Optional[SceneEntity],
        extra_params: Dict[str, Any]
    ) -> Tuple[List[ActionStep], Optional[PlacementPredicate], List[VerificationCondition]]:
        steps: List[ActionStep] = []
        predicate: Optional[PlacementPredicate] = None
        conditions: List[VerificationCondition] = []

        # --- A. Manipulation Pick and Place ---
        if task_type == TaskType.MANIPULATION_PICK_PLACE and target_entities and destination_entity:
            target = target_entities[0]
            dest = destination_entity

            # Construct Geometric Placement Predicate on Plate
            predicate = PlacementPredicate(
                target_entity_id=target.id,
                destination_entity_id=dest.id,
                target_center_xyz=(dest.pose[0], dest.pose[1], dest.pose[2]),
                allowed_radius=0.08,  # Plate surface radius
                min_height=dest.pose[2] - 0.02,
                max_height=dest.pose[2] + 0.10,
                require_upright=True,
            )

            # Planned Action Sequence
            steps = [
                ActionStep(
                    step_index=1,
                    skill_id="move_joints",
                    description=f"Move arm to pre-grasp approach pose above {target.name}",
                    parameters={"pose": "pre_grasp", "target_xyz": (target.pose[0], target.pose[1], target.pose[2] + 0.15)},
                    preconditions=["arm_homed", f"object_{target.id}_visible"],
                    expected_observation={"joint_state": "reached"},
                    timeout_sec=5.0,
                ),
                ActionStep(
                    step_index=2,
                    skill_id="gripper_control",
                    description="Open gripper fingers to maximum aperture",
                    parameters={"action": "open", "aperture_m": 0.08},
                    preconditions=["at_pre_grasp_pose"],
                    expected_observation={"gripper_stroke": 0.08},
                    timeout_sec=3.0,
                ),
                ActionStep(
                    step_index=3,
                    skill_id="move_joints",
                    description=f"Descend end-effector to grasp height at {target.name}",
                    parameters={"pose": "grasp", "target_xyz": (target.pose[0], target.pose[1], target.pose[2])},
                    preconditions=["gripper_open"],
                    expected_observation={"end_effector_z": target.pose[2]},
                    timeout_sec=4.0,
                ),
                ActionStep(
                    step_index=4,
                    skill_id="gripper_control",
                    description=f"Close gripper fingers firmly on {target.name}",
                    parameters={"action": "close", "force_n": 20.0},
                    preconditions=["at_grasp_pose"],
                    expected_observation={"grasp_contact": True},
                    timeout_sec=4.0,
                    is_critical=True,
                ),
                ActionStep(
                    step_index=5,
                    skill_id="move_joints",
                    description=f"Lift {target.name} vertically by 0.12m above table",
                    parameters={"pose": "lift", "dz": 0.12},
                    preconditions=["grasp_secured"],
                    expected_observation={"object_attached": True},
                    timeout_sec=5.0,
                    is_critical=True,
                ),
                ActionStep(
                    step_index=6,
                    skill_id="move_joints",
                    description=f"Transport arm trajectory above {dest.name}",
                    parameters={"pose": "above_destination", "target_xyz": (dest.pose[0], dest.pose[1], dest.pose[2] + 0.15)},
                    preconditions=["object_lifted"],
                    expected_observation={"at_destination_xy": True},
                    timeout_sec=6.0,
                ),
                ActionStep(
                    step_index=7,
                    skill_id="move_joints",
                    description=f"Lower {target.name} onto surface of {dest.name}",
                    parameters={"pose": "placement_surface", "target_xyz": (dest.pose[0], dest.pose[1], dest.pose[2] + 0.02)},
                    preconditions=["above_destination"],
                    expected_observation={"surface_contact": True},
                    timeout_sec=4.0,
                ),
                ActionStep(
                    step_index=8,
                    skill_id="gripper_control",
                    description=f"Open gripper fingers to release {target.name}",
                    parameters={"action": "open", "aperture_m": 0.08},
                    preconditions=["on_destination_surface"],
                    expected_observation={"object_released": True},
                    timeout_sec=3.0,
                ),
                ActionStep(
                    step_index=9,
                    skill_id="pick_and_place",
                    description=f"Verify {target.name} resting settlement and evaluate placement predicate",
                    parameters={"target_object_id": target.id, "destination_id": dest.id},
                    preconditions=["gripper_released"],
                    expected_observation={"predicate_satisfied": True},
                    timeout_sec=5.0,
                    is_critical=True,
                ),
                ActionStep(
                    step_index=10,
                    skill_id="move_joints",
                    description="Retreat arm upward and return to safe home configuration",
                    parameters={"pose": "home"},
                    preconditions=["placement_verified"],
                    expected_observation={"arm_homed": True},
                    timeout_sec=5.0,
                    is_critical=False,
                ),
            ]

            conditions.append(VerificationCondition(
                name="Object Placement on Plate",
                metric="placement_predicate_passed",
                comparator="is_true",
                expected_value=True,
                is_mandatory=True,
            ))

        # --- B. Wheel Speed Control (e.g. RSK wheel1_speed = 5.44) ---
        elif task_type == TaskType.MOBILE_WHEEL_CONTROL:
            wheel_name = extra_params.get("wheel_name", "wheel1_speed")
            target_vel = extra_params.get("velocity", 5.44)

            steps = [
                ActionStep(
                    step_index=1,
                    skill_id="set_wheel_velocity",
                    description=f"Command angular velocity {target_vel} rad/s to actuator '{wheel_name}'",
                    parameters={"wheel_name": wheel_name, "velocity": target_vel, "duration_sec": 0.5},
                    preconditions=["model_physics_initialized", f"actuator_{wheel_name}_ready"],
                    expected_observation={"actuator_commanded": True},
                    timeout_sec=5.0,
                    is_critical=True,
                ),
            ]

            conditions.append(VerificationCondition(
                name=f"Wheel Velocity Response ({wheel_name})",
                metric="measured_joint_velocity",
                comparator="within_tolerance",
                expected_value=target_vel,
                tolerance=1.5,
                is_mandatory=True,
            ))

        # --- C. Mobile Navigation ---
        elif task_type == TaskType.MOBILE_NAVIGATION:
            stop_dist = extra_params.get("stop_distance_m", 0.5)
            target = target_entities[0] if target_entities else None
            target_name = target.name if target else "target destination"

            steps = [
                ActionStep(
                    step_index=1,
                    skill_id="drive_to_pose",
                    description=f"Navigate base toward {target_name} stopping {stop_dist}m away",
                    parameters={"target_xyz": target.xyz if target else (2.0, 0.0, 0.0), "stop_distance_m": stop_dist},
                    preconditions=["odom_active", "costmap_clear"],
                    expected_observation={"distance_to_target": stop_dist},
                    timeout_sec=15.0,
                    is_critical=True,
                )
            ]

            conditions.append(VerificationCondition(
                name="Navigation Final Distance",
                metric="stopped_at_distance",
                comparator="within_tolerance",
                expected_value=stop_dist,
                tolerance=0.10,
                is_mandatory=True,
            ))

        return steps, predicate, conditions


_GLOBAL_TASK_INTERPRETER = TaskInterpreter()


def get_task_interpreter() -> TaskInterpreter:
    """Get the global task interpreter instance."""
    return _GLOBAL_TASK_INTERPRETER
