"""Natural-Language Task Interpreter and Planner for RoboSahaya Studio.

Transforms unstructured engineering instructions into validated, capability-checked
Structured Task Specifications and action sequences.
Enforces honest capability gating, detects ambiguous prompts, rejects unsupported actions,
and computes deterministic plan fingerprints for invalidation tracking.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from robosahaya.robot_registry import RobotModel, RobotRegistry, get_robot_registry
from .capabilities import CapabilityType, RobotCapabilityResolver
from .contracts import (
    ActionStep,
    PlacementPredicate,
    SkillAction,
    TaskSpecification,
    TaskType,
    VerificationCondition,
    compute_task_fingerprint,
)
from .scene import SceneEntity, WorldScene, get_world_scene
from .skills import SkillRegistry, get_skill_registry


class TaskInterpreter:
    """Interprets natural language into structured multi-robot task specifications."""

    def __init__(
        self,
        registry: Optional[RobotRegistry] = None,
        skill_registry: Optional[SkillRegistry] = None,
    ):
        self.registry = registry or get_robot_registry()
        self.cap_resolver = RobotCapabilityResolver(self.registry)
        self.skill_registry = skill_registry or get_skill_registry()

    def interpret_instruction(
        self,
        instruction: str,
        robot_id_override: Optional[str] = None,
        world_id_override: Optional[str] = None,
        task_id: Optional[str] = None,
        simulator_adapter_override: Optional[str] = None,
    ) -> TaskSpecification:
        """Parse natural language instruction and construct a complete TaskSpecification."""
        task_id = task_id or f"task_{uuid.uuid4().hex[:8]}"
        q_clean = (instruction or "").strip()
        q_lower = q_clean.lower()

        # 0. Reject Empty Instruction
        if not q_clean:
            spec = TaskSpecification(
                task_id=task_id,
                original_instruction="",
                task_type=TaskType.CUSTOM,
                canonical_robot_id=robot_id_override or "differential_drive",
                robot_model_version="v1.0.0",
                simulator_adapter="gazebo_harmonic",
                world_id=world_id_override or "obstacle_world",
                is_supported=False,
                rejection_reason="No instruction text was supplied. Enter a robotics goal or engineering instruction.",
                plan_status="rejected",
            )
            spec.compute_fingerprint()
            return spec

        # 1. Infer Task Type
        task_type = self._classify_task_type(q_lower)

        # 2. Ambiguity Detection
        is_ambiguous, ambig_reason = self._check_ambiguous_instruction(q_clean, q_lower, task_type)
        if is_ambiguous:
            spec = TaskSpecification(
                task_id=task_id,
                original_instruction=q_clean,
                task_type=task_type,
                canonical_robot_id=robot_id_override or "unknown",
                robot_model_version="v1.0.0",
                simulator_adapter="gazebo_harmonic",
                world_id=world_id_override or "obstacle_world",
                is_supported=False,
                rejection_reason=ambig_reason,
                plan_status="rejected",
            )
            spec.compute_fingerprint()
            return spec

        # 3. Resolve or Infer Robot
        resolved_robot = self._resolve_robot(q_lower, task_type, robot_id_override)

        # 4. Resolve or Infer World
        world_id = self._resolve_world(q_lower, task_type, resolved_robot, world_id_override)
        scene = get_world_scene(world_id)

        # 5. Extract Specific Target Objects and Destinations
        target_entities, destination_entity, extra_params = self._ground_entities(q_clean, task_type, scene)

        # 6. Determine Required Capabilities, Sensors, Controllers
        req_caps, req_sensors, req_controllers = self._determine_prerequisites(task_type, extra_params)

        # 7. Validate Robot Compatibility and Capabilities
        is_supported = True
        rejection_reason = None
        known_limitations: List[str] = []

        if not resolved_robot:
            is_supported = False
            rejection_reason = "No compatible robot could be identified for this instruction."
        else:
            can_do, satisfied, missing = self.cap_resolver.check_capabilities(resolved_robot.id, req_caps)
            if not can_do:
                if task_type == TaskType.DRONE_FLIGHT and any(kw in q_lower for kw in ("flight controller", "px4", "sitl", "autopilot")):
                    # Explicit flight controller interface requested/configured
                    can_do = True
                    missing = [m for m in missing if "drone flight controller" not in m.lower()]
                if not can_do:
                    is_supported = False
                    rejection_reason = f"Robot '{resolved_robot.display_name}' cannot execute this task: " + "; ".join(missing)
                    known_limitations.extend(missing)

        # 8. Check Scene Prerequisites for Manipulation, Perception, and Navigation
        if is_supported and task_type == TaskType.MANIPULATION_PICK_PLACE:
            if not target_entities:
                is_supported = False
                rejection_reason = f"Target object to pick up could not be found in '{world_id}'."
            elif not destination_entity:
                is_supported = False
                rejection_reason = f"Destination plate or placement area could not be found in '{world_id}'."
        elif is_supported and task_type == TaskType.PERCEPTION_INSPECTION:
            if any(w in q_lower for w in ("find", "detect", "locate", "track")) and not target_entities:
                is_supported = False
                rejection_reason = f"Target object requested in '{q_clean}' could not be resolved in world scene '{world_id}'."
        elif is_supported and task_type == TaskType.MOBILE_NAVIGATION:
            if not target_entities and not any(w in q_lower for w in ("stop", "halt", "all")):
                is_supported = False
                rejection_reason = f"Target destination requested in '{q_clean}' could not be resolved in world scene '{world_id}'."

        # 9. Synthesize Action Steps and Verification Predicates
        action_sequence: List[SkillAction] = []
        placement_predicate: Optional[PlacementPredicate] = None
        verif_conditions: List[VerificationCondition] = []

        if simulator_adapter_override:
            sim_adapter = simulator_adapter_override
        elif task_type in (TaskType.DRONE_FLIGHT, TaskType.LEGGED_LOCOMOTION):
            sim_adapter = "mujoco"
        elif resolved_robot and resolved_robot.default_simulator:
            sim_adapter = resolved_robot.default_simulator
        else:
            sim_adapter = "gazebo_harmonic"

        if is_supported and resolved_robot:
            action_sequence, placement_predicate, verif_conditions = self._generate_action_sequence(
                task_type=task_type,
                robot=resolved_robot,
                scene=scene,
                target_entities=target_entities,
                destination_entity=destination_entity,
                extra_params=extra_params,
            )

            # 10. Capability-Based Skill Compatibility Verification
            has_flight_ctrl = any(kw in q_lower for kw in ("flight controller", "px4", "sitl", "autopilot"))
            compat_err = self._validate_skill_compatibility(action_sequence, resolved_robot, sim_adapter, flight_controller_enabled=has_flight_ctrl)
            if compat_err:
                is_supported = False
                rejection_reason = compat_err
                action_sequence = []

        spec = TaskSpecification(
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
            preconditions=[
                f"Robot {resolved_robot.id if resolved_robot else ''} model loaded",
                f"World {world_id} materialized",
            ],
            intermediate_criteria={"min_steps": 10},
            placement_predicate=placement_predicate,
            verification_conditions=verif_conditions,
            final_criteria={"task_success": True},
            safety_constraints=[
                "Keep joint velocities within safety limits",
                "Prevent ungrounded collision",
            ],
            is_supported=is_supported,
            rejection_reason=rejection_reason,
            known_limitations=known_limitations,
            plan_status="draft" if is_supported else "rejected",
        )
        spec.compute_fingerprint()
        return spec

    def _check_ambiguous_instruction(
        self,
        q_clean: str,
        q_lower: str,
        task_type: TaskType
    ) -> Tuple[bool, Optional[str]]:
        """Detect underspecified or ambiguous prompts requiring clarification."""
        tokens = [t.strip(",.?!;:") for t in q_lower.split() if t.strip(",.?!;:")]

        # Very brief vague single-word commands
        vague_single_tokens = {
            "move", "drive", "pick", "fly", "walk", "trot", "run", "start",
            "stop", "go", "robot", "task", "grasp", "lift", "operate", "test"
        }
        if len(tokens) <= 1 and (not tokens or tokens[0] in vague_single_tokens):
            return (
                True,
                f"Ambiguous instruction: '{q_clean}' is too brief. Please specify parameters such as target object, destination, speed, or direction."
            )

        # Common underspecified phrases
        vague_phrases = {
            "move forward", "drive forward", "move", "drive", "pick something",
            "pick up something", "pick an object", "grab something", "do something",
            "do task", "run fast", "fly around", "fly drone", "walk around",
            "make it move", "pick and place", "pick it up", "lift something",
            "just move", "test robot", "execute task", "pick object", "put it somewhere",
        }
        if q_lower in vague_phrases:
            if task_type == TaskType.MANIPULATION_PICK_PLACE:
                return (
                    True,
                    f"Ambiguous manipulation instruction: '{q_clean}'. Specify the target object (e.g. 'red cube' or 'object in front of arm') and the destination (e.g. 'plate')."
                )
            elif task_type in (TaskType.MOBILE_WHEEL_CONTROL, TaskType.MOBILE_NAVIGATION):
                return (
                    True,
                    f"Ambiguous mobile motion instruction: '{q_clean}'. Specify velocity, distance, or target destination (e.g. 'move forward at 1.0 m/s')."
                )
            elif task_type == TaskType.DRONE_FLIGHT:
                return (
                    True,
                    f"Ambiguous drone instruction: '{q_clean}'. Specify target altitude, waypoint coordinates, or maneuver."
                )
            elif task_type == TaskType.LEGGED_LOCOMOTION:
                return (
                    True,
                    f"Ambiguous legged locomotion instruction: '{q_clean}'. Specify gait pattern, speed, or distance."
                )
            return (
                True,
                f"Ambiguous instruction: '{q_clean}'. Please provide specific target entities and parameters."
            )

        # Manipulation specifics: checking for vague "something" or "somewhere"
        if task_type == TaskType.MANIPULATION_PICK_PLACE:
            has_vague_term = any(v in q_lower for v in ("pick something", "put somewhere", "put it somewhere", "pick anything"))
            if has_vague_term:
                return (
                    True,
                    "Ambiguous manipulation instruction: missing specific target object or destination. Specify what to pick and where to place it."
                )

        # Wheel control specifics: "move forward" or "set wheel" without numeric velocity
        if task_type == TaskType.MOBILE_WHEEL_CONTROL:
            has_number = bool(re.search(r"[-+]?\d+(?:\.\d+)?", q_clean))
            is_wheel_motion = any(w in q_lower for w in ("move forward", "drive forward", "set wheel", "wheel_speed", "wheel speed", "spin wheel"))
            if not has_number and is_wheel_motion:
                return (
                    True,
                    "Ambiguous wheel control instruction: missing target speed or velocity. Specify a numeric speed (e.g. 'move forward at 5.44 rad/s')."
                )

        # Conflicting / contradictory instructions
        if ("open gripper" in q_lower or "open the gripper" in q_lower) and ("close gripper" in q_lower or "close the gripper" in q_lower):
            return (
                True,
                f"Conflicting instruction: '{q_clean}' contains mutually contradictory actions ('open gripper' and 'close gripper')."
            )
        if ("move forward" in q_lower or "drive forward" in q_lower) and ("move backward" in q_lower or "drive backward" in q_lower):
            return (
                True,
                f"Conflicting instruction: '{q_clean}' commands both forward and backward drive simultaneously."
            )
        if "altitude" in q_lower:
            alt_matches = re.findall(r"altitude\s+([-+]?\d+(?:\.\d+)?)", q_lower)
            if alt_matches and float(alt_matches[0]) < 0:
                return (
                    True,
                    f"Conflicting instruction: '{q_clean}' specifies negative altitude ({alt_matches[0]}m) which violates physical floor constraints."
                )

        # Ambiguous marker target without spatial disambiguation
        # E.g. "move toward the marker" without specifying "left", "right", or "goal"
        is_generic_marker = bool(re.search(r"\b(?:the\s+)?marker\b", q_lower))
        if is_generic_marker and not any(k in q_lower for k in ("left", "right", "goal", "red", "green", "blue", "yellow", "start", "visible", "course", "waypoint")):
            return (
                True,
                f"Ambiguous marker target: '{q_clean}' references a marker without spatial disambiguation. Specify 'marker on the left' or 'marker on the right'."
            )

        return False, None

    def _classify_task_type(self, q: str) -> TaskType:
        if (("pick" in q or "grasp" in q or "lift" in q) and ("place" in q or "bin" in q or "plate" in q or "put" in q)) or any(w in q for w in ("pick and place", "pick up", "place it", "grasp", "lift", "place on", "put it", "pick something", "put somewhere")):
            return TaskType.MANIPULATION_PICK_PLACE
        if any(w in q for w in (
            "move arm", "move joint", "set joint", "reach pose", "reach target", "reach [",
            "open gripper", "close gripper", "arm to", "panda joint", "ur5 joint", "home arm",
            "joint_positions", "joint position", "joint error", "home pose", "nominal pose",
        )) or (("reach" in q or "target" in q) and any(c in q for c in ("pose", "xyz", "meter", "position", "arm"))) \
           or (("gripper" in q) and any(c in q for c in ("open", "close", "aperture", "stroke"))) \
           or (("joint" in q) and any(c in q for c in ("rad", "deg", "move", "set", "angle", "home", "arm"))) \
           or ("home" in q and any(c in q for c in ("pose", "config", "configuration", "arm", "position", "nominal"))):
            return TaskType.MANIPULATION_TRAJECTORY
        if any(w in q for w in (
            "wheel_speed", "wheel speed", "set wheel", "wheel1", "wheel2", "wheel3", "wheel4",
            "move forward at", "drive forward at", "forward at", "move at", "drive at", "velocity at",
            "turn in place", "rotate in place", "turn by", "rotate by", "stop the robot", "stop the soccer robot",
            "stop mobile", "stop base", "stop robot", "halt",
        )) or (("turn" in q or "rotate" in q) and any(r in q for r in ("robot", "base", "soccer", "degrees", "deg", "rsk", "in place"))):
            return TaskType.MOBILE_WHEEL_CONTROL
        if any(w in q for w in ("move to", "drive to", "navigate to", "stop 0.", "stop 1.", "avoid", "stop before", "move toward", "drive toward", "go to the waypoint", "go to waypoint")):
            return TaskType.MOBILE_NAVIGATION
        if any(w in q for w in ("fly", "takeoff", "land", "waypoint", "altitude", "quadcopter")):
            return TaskType.DRONE_FLIGHT
        if any(w in q for w in ("walk", "gait", "stand", "trot", "quadruped", "biped")):
            return TaskType.LEGGED_LOCOMOTION
        if any(w in q for w in ("create", "build", "assemble", "launch", "setup", "spawn", "repair", "fix")):
            return TaskType.SIMULATION_LIFECYCLE
        if any(w in q for w in ("camera", "picture", "capture", "scan", "inspect topic", "find", "detect", "locate")):
            return TaskType.PERCEPTION_INSPECTION
        if any(w in q for w in ("inspect", "test")):
            return TaskType.SIMULATION_LIFECYCLE
        if any(w in q for w in ("move forward", "drive forward", "spin wheel")):
            return TaskType.MOBILE_WHEEL_CONTROL
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
        if task_type in (TaskType.MANIPULATION_PICK_PLACE, TaskType.MANIPULATION_TRAJECTORY):
            if "ur5" in q:
                return self.registry.get_model("ur5e")
            if "so_arm101" in q or "so101" in q:
                return self.registry.get_model("so_arm101")
            if "so_arm100" in q or "so100" in q or "so_arm" in q:
                return self.registry.get_model("so_arm100")
            return self.registry.get_model("franka_panda")
        elif task_type == TaskType.MOBILE_WHEEL_CONTROL:
            if "wheel" in q or "rsk" in q or "soccer" in q:
                return self.registry.get_model("rsk")
            return self.registry.get_model("differential_drive")
        elif task_type == TaskType.MOBILE_NAVIGATION:
            return self.registry.get_model("differential_drive")
        elif task_type == TaskType.DRONE_FLIGHT:
            return self.registry.get_model("skydio_x2")
        elif task_type == TaskType.LEGGED_LOCOMOTION:
            return self.registry.get_model("unitree_go2")
        elif task_type == TaskType.PERCEPTION_INSPECTION:
            if any(w in q for w in ("cube", "table", "arm", "panda", "ur5", "work")):
                return self.registry.get_model("franka_panda")
            if any(w in q for w in ("ball", "soccer", "goal")):
                return self.registry.get_model("rsk")
            return self.registry.get_model("differential_drive")

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

        if task_type in (TaskType.MANIPULATION_PICK_PLACE, TaskType.MANIPULATION_TRAJECTORY) or (robot and robot.family == "robotic arm") or any(w in q for w in ("cube", "table", "bin", "plate")):
            return "tabletop_world"
        if (robot and robot.id == "rsk") or any(w in q for w in ("ball", "soccer")):
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
            target = scene.find_target_object(instruction)
            if not target:
                has_specific_unknown_desc = any(
                    w in q_lower for w in ("purple", "yellow", "orange", "cylinder", "pyramid", "triangle", "cup", "can", "bottle", "nonexistent", "unknown")
                )
                if not has_specific_unknown_desc:
                    target = scene.find_target_object("")
            if target:
                target_entities.append(target)

            dest = scene.find_destination(instruction)
            if not dest:
                dest = scene.find_destination("")
            if dest:
                destination_entity = dest

        # 2. Wheel Speed & Mobile Base Control Extraction
        elif task_type == TaskType.MOBILE_WHEEL_CONTROL:
            dur_match = re.search(r"(?:for|duration)\s+([0-9.]+)\s*(?:s|sec|seconds)?", q_lower)
            duration_sec = float(dur_match.group(1)) if dur_match else 0.5
            extra_params["duration_sec"] = duration_sec

            is_turn = any(t in q_lower for t in ("turn", "rotate")) and not any(a in q_lower for a in ("arm", "wrist", "joint"))
            is_drive_forward = ("drive" in q_lower and "forward" in q_lower) and not ("move forward at" in q_lower)
            is_stop = any(s in q_lower for s in ("stop mobile", "stop base", "stop robot", "halt")) or ("stop" in q_lower and any(r in q_lower for r in ("robot", "soccer", "base", "rsk", "wheels", "motion")))

            has_drive = any(d in q_lower for d in ("drive", "move forward", "forward"))
            if (has_drive and is_turn) or (has_drive and ("stop" in q_lower and ("turn" in q_lower or "and stop" in q_lower))):
                extra_params["action_subtype"] = "multistep_mobile"
                dist_match = re.search(r"(?:forward|move|drive)\s+([0-9.]+)\s*(?:m|meter|meters)?", q_lower)
                dist = float(dist_match.group(1)) if dist_match else 0.5
                extra_params["distance_m"] = dist
                extra_params["speed_m_s"] = 0.2
                extra_params["drive_duration_sec"] = max(0.5, min(2.0, dist / 0.2))

                angle_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)", q_lower)
                angle_deg = float(angle_match.group(1)) if angle_match else 90.0
                extra_params["angle_deg"] = angle_deg
                extra_params["turn_duration_sec"] = 0.5
                extra_params["stop_duration_sec"] = 0.2

            elif is_turn:
                angle_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)", q_lower)
                angle_deg = float(angle_match.group(1)) if angle_match else 90.0
                extra_params["action_subtype"] = "turn_in_place"
                extra_params["angle_deg"] = angle_deg
                if not dur_match:
                    extra_params["duration_sec"] = 1.0

            elif is_drive_forward:
                cleaned_instruction = re.sub(r"wheel\s*\d(?:_speed)?", "", instruction, flags=re.IGNORECASE)
                vel_match = re.search(r"at\s+([-+]?\d+(?:\.\d+)?)\s*(?:m/s|meter/s|rad/s)?", cleaned_instruction)
                if not vel_match:
                    vel_match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:m/s|meter/s)", cleaned_instruction)
                if not vel_match:
                    vel_match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned_instruction)
                speed = float(vel_match.group(1) if vel_match.lastindex else vel_match.group(0)) if vel_match else 0.2
                extra_params["action_subtype"] = "drive_forward"
                extra_params["speed_m_s"] = speed
                extra_params["velocity"] = speed
                if not dur_match:
                    extra_params["duration_sec"] = 1.0

            elif is_stop:
                extra_params["action_subtype"] = "stop_mobile_base"

            else:
                wheel_match = re.search(r"(wheel[1-4](?:_speed)?)", q_lower)
                wheel_name = wheel_match.group(1) if wheel_match else "wheel1_speed"
                if not wheel_name.endswith("_speed"):
                    wheel_name = f"{wheel_name}_speed"
                extra_params["wheel_name"] = wheel_name

                cleaned_instruction = re.sub(r"wheel\s*\d(?:_speed)?", "", instruction, flags=re.IGNORECASE)
                vel_match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned_instruction)
                velocity = float(vel_match.group(0)) if vel_match else 5.44
                extra_params["velocity"] = velocity
                extra_params["action_subtype"] = "set_wheel_velocity"

        # 3. Mobile Navigation Grounding
        elif task_type == TaskType.MOBILE_NAVIGATION:
            target = scene.find_target_object(instruction)
            if target:
                target_entities.append(target)
                # For navigation the target entity is also the destination
                destination_entity = target

            # Check for obstacle avoidance directive (e.g. "Avoid the obstacle and go to the waypoint")
            if "avoid" in q_lower and "obstacle" in q_lower:
                for ent in scene.list_entities():
                    if ent.entity_type == "obstacle" or "obstacle" in ent.id:
                        extra_params["avoid_obstacle"] = ent.id
                        extra_params["obstacle_entity"] = ent
                        extra_params["has_obstacle_avoidance"] = True
                        break

            # Check for stopping before obstacle directive (e.g. "Stop before reaching the obstacle")
            if "stop before" in q_lower or ("stop" in q_lower and "obstacle" in q_lower):
                extra_params["action_subtype"] = "stop_mobile_base"
                extra_params["stop_before_obstacle"] = True
                for ent in scene.list_entities():
                    if ent.entity_type == "obstacle" or "obstacle" in ent.id:
                        if not target:
                            target_entities.append(ent)
                            destination_entity = ent
                        extra_params["obstacle_entity"] = ent
                        break

            dist_match = re.search(r"stop\s+([0-9.]+)\s*(?:m|meter|metres)?", q_lower)
            extra_params["stop_distance_m"] = float(dist_match.group(1)) if dist_match else 0.5

        # 4. Arm Manipulation Trajectory Grounding
        elif task_type == TaskType.MANIPULATION_TRAJECTORY:
            dur_match = re.search(r"(?:for|duration)\s+([0-9.]+)\s*(?:s|sec|seconds)?", q_lower)
            extra_params["duration_sec"] = float(dur_match.group(1)) if dur_match else 1.0

            if "gripper" in q_lower or "finger" in q_lower or "jaw" in q_lower:
                action = "close" if "close" in q_lower else "open"
                extra_params["action_subtype"] = "gripper_control"
                extra_params["action"] = action
            elif any(k in q_lower for k in ("reach", "target pose", "target position", "target_xyz", "reach pose")):
                extra_params["action_subtype"] = "reach_target_pose"
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?", instruction)
                if len(nums) >= 3:
                    extra_params["target_xyz"] = [float(nums[0]), float(nums[1]), float(nums[2])]
                else:
                    extra_params["target_xyz"] = [0.4, 0.0, 0.4]
            elif "home" in q_lower or "nominal" in q_lower:
                extra_params["action_subtype"] = "move_joints"
                extra_params["pose"] = "home"
            else:
                extra_params["action_subtype"] = "move_joints"
                j_match = re.search(r"joint\s*([0-9a-zA-Z_]+)", q_lower)
                j_name = j_match.group(1) if j_match else "1"
                val_match = re.search(r"(?:to|as|at)\s+([-+]?\d+(?:\.\d+)?)", q_lower)
                if not val_match:
                    cleaned = re.sub(r"(?:so_arm100|so_arm101|so100|so101|ur5e|ur5|fr3|panda|joint\s*[0-9a-zA-Z_]+)", "", instruction, flags=re.IGNORECASE)
                    val_match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
                target_val = float(val_match.group(1) if val_match and val_match.lastindex else (val_match.group(0) if val_match else 0.3))
                extra_params["joint_name"] = j_name
                extra_params["target_val"] = target_val

        # 5. Legged Locomotion / Posture Grounding
        elif task_type == TaskType.LEGGED_LOCOMOTION:
            has_motion = any(w in q_lower for w in ("walk", "trot", "step", "forward", "turn", "run", "gait"))
            has_posture = any(p in q_lower for p in ("stand", "sit", "lie", "posture"))
            has_stop = any(s in q_lower for s in ("stop", "halt"))
            if has_motion and (has_posture or has_stop):
                extra_params["action_subtype"] = "compound_legged"
            elif "sit" in q_lower:
                extra_params["action_subtype"] = "stand_sit_posture"
                extra_params["posture"] = "sit"
            elif "stand" in q_lower:
                extra_params["action_subtype"] = "stand_sit_posture"
                extra_params["posture"] = "stand"
            else:
                extra_params["action_subtype"] = "legged_gait"

        # 6. Perception / Inspection Entity Grounding
        elif task_type == TaskType.PERCEPTION_INSPECTION:
            target = scene.find_target_object(instruction)
            if target:
                target_entities.append(target)

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
        elif task_type == TaskType.MANIPULATION_TRAJECTORY:
            subtype = extra_params.get("action_subtype", "move_joints")
            req_caps = [CapabilityType.MANIPULATION_ARM]
            if subtype == "gripper_control":
                req_caps.append(CapabilityType.MANIPULATION_GRIPPER)
            return (
                req_caps,
                ["joint_encoders"],
                ["joint_trajectory_controller"],
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

    def _validate_skill_compatibility(
        self,
        steps: List[SkillAction],
        robot: RobotModel,
        sim_adapter: str,
        flight_controller_enabled: bool = False,
    ) -> Optional[str]:
        """Verify that every skill in the planned action sequence is registered and compatible."""
        compat_skills = self.skill_registry.get_compatible_skills(robot, sim_adapter)
        compat_ids = {s.skill_id for s in compat_skills}
        if flight_controller_enabled and robot.family == "drone/aerial":
            compat_ids.add("drone_flight")
        compat_ids.add("perceive_target")

        for step in steps:
            skill = self.skill_registry.get_skill(step.skill_id)
            if not skill:
                return f"Skill '{step.skill_id}' is not registered in the execution engine."
            if sim_adapter not in skill.supported_adapters:
                return f"Skill '{step.skill_id}' does not support simulator '{sim_adapter}'."
            if step.skill_id not in compat_ids:
                return f"Skill '{step.skill_id}' is not supported by robot '{robot.display_name}' on simulator '{sim_adapter}'."

        return None

    def _generate_action_sequence(
        self,
        task_type: TaskType,
        robot: RobotModel,
        scene: WorldScene,
        target_entities: List[SceneEntity],
        destination_entity: Optional[SceneEntity],
        extra_params: Dict[str, Any]
    ) -> Tuple[List[SkillAction], Optional[PlacementPredicate], List[VerificationCondition]]:
        steps: List[SkillAction] = []
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
                allowed_radius=0.08,
                min_height=dest.pose[2] - 0.02,
                max_height=dest.pose[2] + 0.10,
                require_upright=True,
            )

            # Planned Action Sequence with Bounded Retries
            steps = [
                SkillAction(
                    step_index=1,
                    skill_id="move_joints",
                    description=f"Move arm to pre-grasp approach pose above {target.name}",
                    parameters={"pose": "pre_grasp", "target_xyz": (target.pose[0], target.pose[1], target.pose[2] + 0.15)},
                    preconditions=["arm_homed", f"object_{target.id}_visible"],
                    expected_observation={"joint_state": "reached"},
                    timeout_sec=5.0,
                    can_retry=True,
                    max_retries=2,
                ),
                SkillAction(
                    step_index=2,
                    skill_id="gripper_control",
                    description="Open gripper fingers to maximum aperture",
                    parameters={"action": "open", "aperture_m": 0.08},
                    preconditions=["at_pre_grasp_pose"],
                    expected_observation={"gripper_stroke": 0.08},
                    timeout_sec=3.0,
                    can_retry=True,
                    max_retries=2,
                ),
                SkillAction(
                    step_index=3,
                    skill_id="move_joints",
                    description=f"Descend end-effector to grasp height at {target.name}",
                    parameters={"pose": "grasp", "target_xyz": (target.pose[0], target.pose[1], target.pose[2])},
                    preconditions=["gripper_open"],
                    expected_observation={"end_effector_z": target.pose[2]},
                    timeout_sec=4.0,
                    can_retry=True,
                    max_retries=2,
                ),
                SkillAction(
                    step_index=4,
                    skill_id="gripper_control",
                    description=f"Close gripper fingers firmly on {target.name}",
                    parameters={"action": "close", "force_n": 20.0},
                    preconditions=["at_grasp_pose"],
                    expected_observation={"grasp_contact": True},
                    timeout_sec=4.0,
                    is_critical=True,
                    can_retry=True,
                    max_retries=2,
                ),
                SkillAction(
                    step_index=5,
                    skill_id="move_joints",
                    description=f"Lift {target.name} vertically by 0.12m above table",
                    parameters={"pose": "lift", "dz": 0.12},
                    preconditions=["grasp_secured"],
                    expected_observation={"object_attached": True},
                    timeout_sec=5.0,
                    is_critical=True,
                    can_retry=True,
                    max_retries=2,
                ),
                SkillAction(
                    step_index=6,
                    skill_id="move_joints",
                    description=f"Transport arm trajectory above {dest.name}",
                    parameters={"pose": "above_destination", "target_xyz": (dest.pose[0], dest.pose[1], dest.pose[2] + 0.15)},
                    preconditions=["object_lifted"],
                    expected_observation={"at_destination_xy": True},
                    timeout_sec=6.0,
                    can_retry=True,
                    max_retries=2,
                ),
                SkillAction(
                    step_index=7,
                    skill_id="move_joints",
                    description=f"Lower {target.name} onto surface of {dest.name}",
                    parameters={"pose": "placement_surface", "target_xyz": (dest.pose[0], dest.pose[1], dest.pose[2] + 0.02)},
                    preconditions=["above_destination"],
                    expected_observation={"surface_contact": True},
                    timeout_sec=4.0,
                    can_retry=True,
                    max_retries=2,
                ),
                SkillAction(
                    step_index=8,
                    skill_id="gripper_control",
                    description=f"Open gripper fingers to release {target.name}",
                    parameters={"action": "open", "aperture_m": 0.08},
                    preconditions=["on_destination_surface"],
                    expected_observation={"object_released": True},
                    timeout_sec=3.0,
                    can_retry=True,
                    max_retries=2,
                ),
                SkillAction(
                    step_index=9,
                    skill_id="pick_and_place",
                    description=f"Verify {target.name} resting settlement and evaluate placement predicate",
                    parameters={"target_object_id": target.id, "destination_id": dest.id},
                    preconditions=["gripper_released"],
                    expected_observation={"predicate_satisfied": True},
                    timeout_sec=5.0,
                    is_critical=True,
                    can_retry=True,
                    max_retries=2,
                ),
                SkillAction(
                    step_index=10,
                    skill_id="move_joints",
                    description="Retreat arm upward and return to safe home configuration",
                    parameters={"pose": "home"},
                    preconditions=["placement_verified"],
                    expected_observation={"arm_homed": True},
                    timeout_sec=5.0,
                    is_critical=False,
                    can_retry=False,
                ),
            ]

            conditions.append(VerificationCondition(
                name="Object Placement on Plate",
                metric="placement_predicate_passed",
                comparator="is_true",
                expected_value=True,
                is_mandatory=True,
            ))

        # --- B. Mobile Wheel & Motion Control ---
        elif task_type == TaskType.MOBILE_WHEEL_CONTROL:
            subtype = extra_params.get("action_subtype", "set_wheel_velocity")
            dur = float(extra_params.get("duration_sec", 0.5))

            if subtype == "multistep_mobile":
                speed = float(extra_params.get("speed_m_s", 0.2))
                drive_dur = float(extra_params.get("drive_duration_sec", 0.5))
                dist = float(extra_params.get("distance_m", speed * drive_dur))
                angle_deg = float(extra_params.get("angle_deg", 90.0))
                turn_dur = float(extra_params.get("turn_duration_sec", 0.5))
                stop_dur = float(extra_params.get("stop_duration_sec", 0.2))

                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="drive_forward",
                        description=f"Drive mobile base forward at {speed} m/s",
                        parameters={"speed_m_s": speed, "velocity": speed, "duration_sec": drive_dur, "distance_m": dist},
                        preconditions=["model_physics_initialized"],
                        expected_observation={"motion_verified": True},
                        timeout_sec=drive_dur + 3.0,
                        is_critical=True,
                    ),
                    SkillAction(
                        step_index=2,
                        skill_id="turn_in_place",
                        description=f"Turn mobile base in place by {angle_deg} degrees",
                        parameters={"angle_deg": angle_deg, "angular_velocity": 1.0, "duration_sec": turn_dur},
                        preconditions=["model_physics_initialized"],
                        expected_observation={"motion_verified": True},
                        timeout_sec=turn_dur + 3.0,
                        is_critical=True,
                    ),
                    SkillAction(
                        step_index=3,
                        skill_id="stop_mobile_base",
                        description="Stop mobile base and settle velocities",
                        parameters={"duration_sec": stop_dur},
                        preconditions=["model_physics_initialized"],
                        expected_observation={"is_stopped": True},
                        timeout_sec=stop_dur + 3.0,
                        is_critical=True,
                    ),
                ]
                conditions.append(VerificationCondition(
                    name="Forward Drive Displacement",
                    metric="measured_displacement_m",
                    comparator="gt",
                    expected_value=0.005,
                    is_mandatory=True,
                ))
                conditions.append(VerificationCondition(
                    name="Turn in Place Response",
                    metric="measured_yaw_change_rad",
                    comparator="gt",
                    expected_value=0.01,
                    is_mandatory=True,
                ))

            elif subtype == "turn_in_place":
                angle_deg = float(extra_params.get("angle_deg", 90.0))
                ang_vel = float(extra_params.get("angular_velocity", 1.0))
                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="turn_in_place",
                        description=f"Turn mobile base in place by {angle_deg} degrees",
                        parameters={"angle_deg": angle_deg, "angular_velocity": ang_vel, "duration_sec": dur},
                        preconditions=["model_physics_initialized", "actuator_wheel1_speed_ready"],
                        expected_observation={"motion_verified": True},
                        timeout_sec=max(5.0, dur + 3.0),
                        is_critical=True,
                        can_retry=True,
                        max_retries=2,
                    )
                ]
                conditions.append(VerificationCondition(
                    name="Turn in Place Response",
                    metric="measured_yaw_change_rad",
                    comparator="gt",
                    expected_value=0.02,
                    is_mandatory=True,
                ))

            elif subtype == "drive_forward":
                speed = float(extra_params.get("speed_m_s", extra_params.get("velocity", 0.2)))
                dist = speed * dur
                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="drive_forward",
                        description=f"Drive mobile base forward at {speed} m/s for {dur}s",
                        parameters={"speed_m_s": speed, "velocity": speed, "duration_sec": dur, "distance_m": dist},
                        preconditions=["model_physics_initialized", "actuator_wheel1_speed_ready"],
                        expected_observation={"motion_verified": True},
                        timeout_sec=max(5.0, dur + 3.0),
                        is_critical=True,
                        can_retry=True,
                        max_retries=2,
                    )
                ]
                conditions.append(VerificationCondition(
                    name="Forward Drive Displacement",
                    metric="measured_displacement_m",
                    comparator="gt",
                    expected_value=0.005,
                    is_mandatory=True,
                ))

            elif subtype == "stop_mobile_base":
                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="stop_mobile_base",
                        description="Stop mobile base and halt all wheel actuators",
                        parameters={},
                        preconditions=["model_physics_initialized"],
                        expected_observation={"is_stopped": True},
                        timeout_sec=5.0,
                        is_critical=True,
                        can_retry=False,
                    )
                ]
                conditions.append(VerificationCondition(
                    name="Mobile Base Stopped",
                    metric="is_stopped",
                    comparator="is_true",
                    expected_value=True,
                    is_mandatory=True,
                ))

            else:
                wheel_name = extra_params.get("wheel_name", "wheel1_speed")
                target_vel = extra_params.get("velocity", 5.44)

                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="set_wheel_velocity",
                        description=f"Command angular velocity {target_vel} rad/s to actuator '{wheel_name}'",
                        parameters={"wheel_name": wheel_name, "velocity": target_vel, "duration_sec": dur},
                        preconditions=["model_physics_initialized", f"actuator_{wheel_name}_ready"],
                        expected_observation={"actuator_commanded": True},
                        timeout_sec=5.0,
                        is_critical=True,
                        can_retry=True,
                        max_retries=2,
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
                SkillAction(
                    step_index=1,
                    skill_id="drive_to_pose",
                    description=f"Navigate base toward {target_name} stopping {stop_dist}m away",
                    parameters={"target_xyz": target.xyz if target else (2.0, 0.0, 0.0), "stop_distance_m": stop_dist},
                    preconditions=["odom_active", "costmap_clear"],
                    expected_observation={"distance_to_target": stop_dist},
                    timeout_sec=15.0,
                    is_critical=True,
                    can_retry=True,
                    max_retries=2,
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

        # --- D. Arm Manipulation Trajectory ---
        elif task_type == TaskType.MANIPULATION_TRAJECTORY:
            subtype = extra_params.get("action_subtype", "move_joints")
            dur = float(extra_params.get("duration_sec", 1.0))

            if subtype == "gripper_control":
                action = extra_params.get("action", "open")
                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="gripper_control",
                        description=f"Command gripper {action}",
                        parameters={"action": action, "aperture_m": 0.08 if action == "open" else 0.0, "duration_sec": dur},
                        preconditions=["model_physics_initialized", "arm_manipulator_ready"],
                        expected_observation={"motion_verified": True},
                        timeout_sec=max(5.0, dur + 3.0),
                        is_critical=True,
                        can_retry=False,
                    )
                ]
                conditions.append(VerificationCondition(
                    name=f"Gripper {action.capitalize()} State",
                    metric="gripper_action",
                    comparator="eq",
                    expected_value=action,
                    is_mandatory=True,
                ))

            elif subtype == "reach_target_pose":
                target_xyz = extra_params.get("target_xyz", [0.4, 0.0, 0.4])
                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="reach_target_pose",
                        description=f"Move end-effector toward target position {target_xyz}",
                        parameters={"target_xyz": target_xyz, "tolerance_m": 0.15, "duration_sec": dur},
                        preconditions=["model_physics_initialized", "arm_manipulator_ready"],
                        expected_observation={"motion_verified": True},
                        timeout_sec=max(5.0, dur + 4.0),
                        is_critical=True,
                        can_retry=False,
                    )
                ]
                conditions.append(VerificationCondition(
                    name="End-Effector Reach Motion",
                    metric="displacement_m",
                    comparator="gt",
                    expected_value=0.01,
                    is_mandatory=True,
                ))

            elif extra_params.get("pose") == "home":
                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="move_joints",
                        description="Command arm to safe home configuration",
                        parameters={"pose": "home", "duration_sec": dur},
                        preconditions=["model_physics_initialized", "arm_manipulator_ready"],
                        expected_observation={"motion_verified": True},
                        timeout_sec=max(5.0, dur + 3.0),
                        is_critical=True,
                        can_retry=False,
                    )
                ]
                conditions.append(VerificationCondition(
                    name="Arm Home Configuration",
                    metric="motion_verified",
                    comparator="is_true",
                    expected_value=True,
                    is_mandatory=True,
                ))

            else:
                j_name = extra_params.get("joint_name", "1")
                target_val = float(extra_params.get("target_val", 0.3))
                j_dict = {f"panda_joint{j_name}": target_val} if "panda" in robot.id else {str(j_name): target_val}
                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="move_joints",
                        description=f"Command joint {j_name} to {target_val} rad",
                        parameters={"joint_positions": j_dict, "duration_sec": dur},
                        preconditions=["model_physics_initialized", "arm_manipulator_ready"],
                        expected_observation={"motion_verified": True},
                        timeout_sec=max(5.0, dur + 3.0),
                        is_critical=True,
                        can_retry=False,
                    )
                ]
                conditions.append(VerificationCondition(
                    name="Joint Trajectory Execution",
                    metric="max_joint_error_rad",
                    comparator="lte",
                    expected_value=0.20,
                    is_mandatory=True,
                ))

        # --- E. Legged Locomotion / Posture ---
        elif task_type == TaskType.LEGGED_LOCOMOTION:
            subtype = extra_params.get("action_subtype", "stand_sit_posture")
            dur = float(extra_params.get("duration_sec", 1.0))
            if subtype == "compound_legged":
                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="stand_sit_posture",
                        description="Transition quadruped to standing posture",
                        parameters={"posture": "stand", "duration_sec": 0.8},
                        preconditions=["model_physics_initialized", "quadruped_ready"],
                        expected_observation={"stable": True},
                        timeout_sec=5.0,
                        is_critical=True,
                    ),
                    SkillAction(
                        step_index=2,
                        skill_id="legged_gait",
                        description="Walk/trot quadruped forward",
                        parameters={"gait": "trot", "speed_m_s": 0.4, "duration_sec": 1.2},
                        preconditions=["model_physics_initialized", "quadruped_ready"],
                        expected_observation={"motion_verified": True},
                        timeout_sec=6.0,
                        is_critical=True,
                    ),
                    SkillAction(
                        step_index=3,
                        skill_id="stand_sit_posture",
                        description="Halt gait and stabilize standing posture",
                        parameters={"posture": "stand", "duration_sec": 0.5},
                        preconditions=["model_physics_initialized", "quadruped_ready"],
                        expected_observation={"stable": True},
                        timeout_sec=4.0,
                        is_critical=True,
                    ),
                ]
                conditions.append(VerificationCondition(
                    name="Quadruped Forward Locomotion & Stability",
                    metric="motion_verified",
                    comparator="is_true",
                    expected_value=True,
                    is_mandatory=True,
                ))
            elif subtype == "stand_sit_posture":
                posture = extra_params.get("posture", "stand")
                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="stand_sit_posture",
                        description=f"Command quadruped to {posture} posture",
                        parameters={"posture": posture, "duration_sec": dur},
                        preconditions=["model_physics_initialized", "quadruped_ready"],
                        expected_observation={"stable": True},
                        timeout_sec=max(5.0, dur + 3.0),
                        is_critical=True,
                        can_retry=False,
                    )
                ]
                conditions.append(VerificationCondition(
                    name=f"Quadruped {posture.capitalize()} Posture Stability",
                    metric="stable",
                    comparator="is_true",
                    expected_value=True,
                    is_mandatory=True,
                ))
            else:
                steps = [
                    SkillAction(
                        step_index=1,
                        skill_id="legged_gait",
                        description="Execute dynamic quadruped trotting gait",
                        parameters={"gait": "trot", "speed_m_s": 0.5},
                        preconditions=["model_physics_initialized", "quadruped_ready"],
                        expected_observation={"motion_verified": True},
                        timeout_sec=10.0,
                        is_critical=True,
                        can_retry=False,
                    )
                ]

        # --- F. Drone Flight ---
        elif task_type == TaskType.DRONE_FLIGHT:
            steps = [
                SkillAction(
                    step_index=1,
                    skill_id="drone_flight",
                    description="Execute multirotor flight maneuver",
                    parameters={"altitude_m": 1.5},
                    preconditions=["model_physics_initialized"],
                    expected_observation={"motion_verified": True},
                    timeout_sec=10.0,
                    is_critical=True,
                    can_retry=False,
                )
            ]

        # --- G. Perception & Inspection ---
        elif task_type == TaskType.PERCEPTION_INSPECTION:
            target = target_entities[0] if target_entities else None
            target_id = target.id if target else "target_object"
            target_name = target.name if target else "target object"
            steps = [
                SkillAction(
                    step_index=1,
                    skill_id="perceive_target",
                    description=f"Perceive and detect {target_name} using simulated sensor pipeline",
                    parameters={"target_object_id": target_id, "world_id": scene.world_id},
                    preconditions=["model_physics_initialized"],
                    expected_observation={"perception_verified": True},
                    timeout_sec=5.0,
                    is_critical=True,
                    can_retry=False,
                )
            ]
            conditions.append(VerificationCondition(
                name=f"Perception Target Detection ({target_name})",
                metric="perception_verified",
                comparator="is_true",
                expected_value=True,
                is_mandatory=True,
            ))

        return steps, predicate, conditions


_GLOBAL_TASK_INTERPRETER = TaskInterpreter()


def get_task_interpreter() -> TaskInterpreter:
    """Get the global task interpreter instance."""
    return _GLOBAL_TASK_INTERPRETER
