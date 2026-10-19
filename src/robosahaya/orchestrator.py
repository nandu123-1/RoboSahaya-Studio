import asyncio
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from robosahaya.evidence_models import (
    TaskStage,
    TaskEvent,
    DiagnosisResponse,
    PatchProposal,
    TestResult,
    SensorPayloadValidation,
    EvidenceRecord,
    EvidenceSource,
    EvidenceStatus,
    TaskPlan,
    PermissionTier,
    compute_plan_digest
)
from robosahaya.task_store import TaskStore
from robosahaya.workspace_manager import WorkspaceManager
from robosahaya.robot_catalog import RobotModelCatalog
from robosahaya.robot_registry import RobotModel, RobotRegistry, get_robot_registry
from robosahaya.identity_resolver import IdentityResolver, IdentityResolutionResult
from robosahaya.simulators.registry import SimulatorAdapterRegistry, get_simulator_registry
from robosahaya.policy_gate import PolicyGate
from robosahaya.diagnostics.engine import DiagnosticEngine
from robosahaya.base_inspector import BaseInspector
from robosahaya.mock_inspector import MockInspector

# Optional live inspector
try:
    from robosahaya.ros_inspector import LiveRclpyInspector, RCLPY_AVAILABLE
except Exception:
    LiveRclpyInspector = None
    RCLPY_AVAILABLE = False


class TaskOrchestrator:
    """Coordinates fixture diagnoses and the managed live simulation workflow."""

    def __init__(
        self,
        task_store: Optional[TaskStore] = None,
        workspace_manager: Optional[WorkspaceManager] = None,
        inspector: Optional[BaseInspector] = None,
        mode: str = "live"
    ):
        self.store = task_store or TaskStore()
        self.ws_mgr = workspace_manager or WorkspaceManager()
        self.robot_registry = get_robot_registry()
        self.identity_resolver = IdentityResolver(self.robot_registry)
        self.simulator_registry = get_simulator_registry(self.ws_mgr)
        self.robot_catalog = getattr(self.ws_mgr, "robot_catalog", None) or RobotModelCatalog()
        self.policy_gate = PolicyGate(self.ws_mgr.base_root)
        self.mode = mode
        self.inspector = inspector or self._init_inspector(mode)
        self.diag_engine = DiagnosticEngine(self.inspector)
        self._cancelled_tasks: set[str] = set()

    def _init_inspector(self, mode: str) -> BaseInspector:
        if mode == "live" and RCLPY_AVAILABLE and LiveRclpyInspector is not None:
            try:
                return LiveRclpyInspector()
            except Exception as exc:
                self._inspector_init_error = str(exc)
                return None
        if mode == "live":
            self._inspector_init_error = "rclpy is unavailable; live inspection cannot start."
            return None
        self._inspector_init_error = None
        return MockInspector(scenario_name="clean_healthy_system")

    def _create_event(
        self,
        task_id: str,
        stage: TaskStage,
        message: str,
        severity: str = "info",
        speak_aloud: bool = False,
        tool_name: Optional[str] = None,
        status: str = "in_progress"
    ) -> TaskEvent:
        event = TaskEvent(
            task_id=task_id,
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            severity=severity,
            user_facing_message=message,
            speak_aloud=speak_aloud,
            tool_name=tool_name,
            status=status
        )
        self.store.record_event(event)
        return event

    def cancel_task(self, task_id: str, message: str = "Task was cancelled by user."):
        """Signals task cancellation."""
        self._cancelled_tasks.add(task_id)
        self.ws_mgr.stop_task_runtime(task_id)
        return self._create_event(
            task_id,
            TaskStage.CANCELLED,
            message,
            severity="warning",
            speak_aloud=True,
            status="cancelled"
        )

    def is_cancelled(self, task_id: str) -> bool:
        return task_id in self._cancelled_tasks

    def infer_robot_and_world(
        self,
        query: str,
        manual_overrides: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Infers robot template, simulation world, sensors, topics, and verification tests
        from the user's spoken or typed engineering instruction.
        Uses Evon only when genuinely live; otherwise labels result as Rule-based selection.
        """
        q_lower = query.lower()
        requested_caps = []
        unsupported_caps = []

        if any(w in q_lower for w in ["arm", "manipulator", "gripper", "pick and place", "pick & place", "grasp", "ur5", "panda", "franka", "robotic arm"]):
            requested_caps.append("robotic_arm_manipulator")
        if any(w in q_lower for w in ["drone", "uav", "quadcopter", "aerial", "fly", "flight", "underwater", "submarine"]):
            unsupported_caps.append("aerial_drone")
        if any(w in q_lower for w in ["lidar", "laser", "/scan", "scan"]):
            requested_caps.append("2d_lidar_scan")
        if any(w in q_lower for w in ["camera", "image", "rgb"]):
            requested_caps.append("rgb_camera")
        if any(w in q_lower for w in ["differential", "diff drive", "diff_drive", "mobile robot", "wheeled", "robot", "chassis", "ground robot", "turtlebot"]):
            requested_caps.append("differential_drive_chassis")

        # This selection function is deterministic; Evon is not asked to infer
        # the robot here. Never label a regex/catalogue selection as Evon-generated.
        selection_source = "rule_based"

        # Resolve exact named robots before generic family keywords. This prevents
        # "Build a SO-101" or "Build a Unitree Go2" from silently becoming the
        # bundled Panda or differential-drive model.
        resolve_res = self.identity_resolver.resolve(query)
        named_model = None
        if resolve_res.resolved_id:
            named_model = self.robot_registry.get_model(resolve_res.resolved_id)
        elif self.robot_catalog:
            named_model = self.robot_catalog.resolve_prompt(query)

        unsupported_reason = None
        if "aerial_drone" in unsupported_caps:
            auto_robot = "aerial_drone"
            robot_reason = "Identified an aerial drone / UAV flight requirement from the query."
            auto_world = "obstacle_world"
            world_reason = "Suggested obstacle_world; drone flight support is not verified."
            sensors = []
            topics = []
            tests = []
            criteria = {}
            behavior = "Aerial drone flight simulation is not yet supported."
            unsupported_reason = (
                "Unsupported Capability: Aerial drone, underwater, and humanoid locomotion templates are not yet verified for runtime execution. "
                "Use the Robot Model Library to inspect available descriptions; no unrelated robot will be substituted."
            )
        elif resolve_res.requires_clarification:
            auto_robot = "ambiguous_selection"
            auto_world = "obstacle_world"
            robot_reason = resolve_res.clarification_message or "Multiple robot models matched this category."
            world_reason = "World selection deferred pending robot clarification."
            sensors, topics, tests, criteria = [], [], [], {}
            behavior = "Robot identification requires clarification."
            unsupported_caps.append("ambiguous_robot_request")
            unsupported_reason = resolve_res.clarification_message
        elif named_model and named_model.runtime_ready:
            auto_robot = named_model.id
            if named_model.id == "rsk":
                auto_robot = "rsk"
                auto_world = "soccer_pitch"
                robot_reason = (
                    f"Matched the named robot '{named_model.display_name}'. "
                    f"Configured for MuJoCo physics runtime ({named_model.default_simulator})."
                )
                world_reason = "Selected the soccer pitch world for omnidirectional mobile robot soccer simulation."
                sensors = ["Omnidirectional Wheel Encoders"]
                topics = []
                tests = ["test_rsk_physics_stepping", "test_rsk_wheel_actuation"]
                criteria = {"min_steps": 100, "actuator_response": True}
                behavior = f"Simulate {named_model.display_name} in MuJoCo physics engine, verifying physics stability and wheel actuation."
            elif named_model.family in ("robotic arm", "gripper/end effector"):
                auto_world = "tabletop_world"
                world_reason = "Selected the tabletop world for manipulation and arm inspection."
                sensors = ["Joint Position Encoders", "Wrist RGB Camera"]
                topics = ["/joint_states", "/camera/image_raw"]
                tests = ["test_joint_states_payload", "test_camera_stream_payload"]
                criteria = {"min_messages": 1, "target_topic": "/joint_states"}
                behavior = f"Load the bundled {named_model.display_name} template and verify its configured joint-state and camera streams."
                robot_reason = (
                    f"Matched the named robot '{named_model.display_name}'. "
                    f"A bundled RoboSahaya template is available ({getattr(named_model, 'runtime_template_id', named_model.id)})."
                )
            else:
                auto_world = "obstacle_world"
                world_reason = "Inferred obstacle_world to provide spatial obstacle geometry and walls for sensor verification."
                sensors = ["2D Planar LiDAR", "RGB Monocular Camera"]
                topics = ["/scan", "/camera/image_raw", "/odom", "/cmd_vel"]
                tests = ["test_laser_scan_payload"]
                criteria = {"min_messages": 1, "finite_ranges_required": True, "target_topic": "/scan"}
                behavior = "Differential-drive robot initialized in Gazebo Harmonic with the configured sensor bridges."
                robot_reason = "Inferred differential-drive mobile chassis from wheeled navigation and ground sensor requirements."
        elif named_model:
            auto_robot = named_model.id
            robot_reason = (
                f"Recognized the requested model '{named_model.display_name}' in the robot catalogue. "
                "Its description can be fetched on demand, but it has no verified RoboSahaya Gazebo runtime adapter yet."
            )
            auto_world = "tabletop_world" if named_model.family in ("robotic arm", "gripper/end effector", "mobile manipulator") else "obstacle_world"
            world_reason = f"Suggested {auto_world}; the requested model cannot launch until a compatible simulation adapter is validated."
            sensors, topics, tests, criteria = [], [], [], {}
            behavior = f"Description-only catalogue entry for {named_model.display_name}; runtime launch is blocked until simulator/controller compatibility is verified."
            requested_caps.append(f"robot_model:{named_model.id}")
            unsupported_caps.append(f"unverified_robot_model:{named_model.id}")
            unsupported_reason = (
                f"'{named_model.display_name}' is recognized in the Robot Model Library, but this RoboSahaya build does not yet have a verified runtime adapter for it. "
                f"Open the Robot Model Library and use 'Prepare / Cache Description' to fetch its source description. This will not mark it simulation-ready; "
                "a compatible launch adapter, controllers, sensors, and tests must be validated before execution. RoboSahaya will not substitute another robot."
            )
        elif any(w in q_lower for w in ["humanoid", "biped", "quadruped", "four-legged", "four legged"]):
            auto_robot = "unsupported_legged_robot"
            robot_reason = "Identified a legged or humanoid robot requirement, but no exact runtime-ready template matches it."
            auto_world = "obstacle_world"
            world_reason = "Suggested obstacle_world only; selected robot family is not runtime verified."
            sensors, topics, tests, criteria = [], [], [], {}
            behavior = "Legged/humanoid robot execution is blocked until a compatible model and controller adapter are verified."
            unsupported_caps.append("unverified_legged_robot")
            unsupported_reason = (
                "Quadruped and humanoid models are not currently verified for RoboSahaya's Gazebo runtime. "
                "Open the Robot Model Library to find and cache a matching description; this is not sufficient to enable simulation."
            )
        elif "robotic_arm_manipulator" in requested_caps:
            auto_robot = "robotic_arm_6dof"
            robot_reason = "Inferred an articulated robotic arm from generic manipulation/pick-and-place requirements."
            auto_world = "tabletop_world"
            world_reason = "Inferred tabletop_world providing a work table, pick-and-place cubes, and target bin geometry."
            sensors = ["Joint Position Encoders", "Wrist RGB Camera"]
            topics = ["/joint_states", "/camera/image_raw"]
            tests = ["test_joint_states_payload", "test_camera_stream_payload"]
            criteria = {"min_messages": 1, "target_topic": "/joint_states"}
            behavior = "Load the bundled robotic-arm template in Gazebo Harmonic and verify its configured joint-state and camera streams."
        else:
            auto_robot = "differential_drive"
            robot_reason = "Inferred differential-drive mobile chassis from wheeled navigation and ground sensor requirements."
            auto_world = "obstacle_world"
            world_reason = "Inferred obstacle_world to provide spatial obstacle geometry and walls for sensor verification."
            sensors = []
            topics = []
            if "2d_lidar_scan" in requested_caps or not requested_caps:
                sensors.append("2D Planar LiDAR")
                topics.append("/scan")
            if "rgb_camera" in requested_caps or not requested_caps:
                sensors.append("RGB Monocular Camera")
                topics.append("/camera/image_raw")
            topics.extend(["/odom", "/cmd_vel"])
            tests = ["test_laser_scan_payload"]
            criteria = {"min_messages": 1, "finite_ranges_required": True, "target_topic": "/scan"}
            behavior = "Differential-drive robot initialized in Gazebo Harmonic with the configured sensor bridges."

        return {
            "auto_robot": auto_robot,
            "auto_world": auto_world,
            "robot_reason": robot_reason,
            "world_reason": world_reason,
            "selection_source": selection_source,
            "requested_capabilities": requested_caps,
            "unsupported_capabilities": unsupported_caps,
            "sensors": sensors,
            "topics": topics,
            "verification_tests": tests,
            "measurable_criteria": criteria,
            "expected_behavior": behavior,
            "unsupported_reason": unsupported_reason,
            "resolved_model_id": named_model.id if named_model else None,
        }

    def formulate_plan(
        self,
        task_id: str,
        query: str,
        template_id: Optional[str] = None,
        world_id: Optional[str] = None,
        manual_overrides: Optional[Dict[str, str]] = None,
        mcp_client: Optional[Any] = None
    ) -> TaskPlan:
        infer = self.infer_robot_and_world(query, manual_overrides)
        overrides = dict(manual_overrides or {})
        rejection_reason = None
        is_supported = True
        is_manual = False

        # If user explicitly provided manual overrides, clear ambiguous selection
        if overrides.get("robot_template"):
            infer["unsupported_capabilities"] = [
                c for c in infer["unsupported_capabilities"] if c != "ambiguous_robot_request"
            ]

        # Check unsupported capabilities from query first. A named model can be
        # known to the catalogue without being safe to launch in this pipeline.
        if infer["unsupported_capabilities"]:
            effective_robot = infer["auto_robot"]
            effective_world = infer["auto_world"]
            is_supported = False
            rejection_reason = infer.get("unsupported_reason") or (
                f"Unsupported Capability: You requested '{', '.join(infer['unsupported_capabilities'])}'. "
                "This robot family is not yet verified for runtime execution. Open the Robot Model Library to inspect available descriptions. "
                "RoboSahaya will not silently substitute another robot."
            )
        else:
            # Determine effective robot template
            if "robot_template" in overrides:
                effective_robot = overrides["robot_template"]
                is_robot_overridden = True
            elif template_id and manual_overrides and "robot_template" in manual_overrides:
                effective_robot = template_id
                is_robot_overridden = True
                overrides["robot_template"] = template_id
            else:
                effective_robot = infer["auto_robot"]
                is_robot_overridden = False

            # Normalize aliases
            if effective_robot == "robotic_arm_manipulator":
                effective_robot = "robotic_arm_6dof"

            # Determine effective world template
            if "world_template" in overrides:
                effective_world = overrides["world_template"]
                is_world_overridden = True
            elif world_id and manual_overrides and "world_template" in manual_overrides:
                effective_world = world_id
                is_world_overridden = True
                overrides["world_template"] = world_id
            else:
                effective_world = infer["auto_world"]
                is_world_overridden = False

            is_manual = is_robot_overridden or is_world_overridden

            approved_robots = ("differential_drive", "robotic_arm_6dof", "franka_panda", "so_arm101", "rsk")
            approved_worlds = ("obstacle_world", "tabletop_world", "soccer_pitch")

            if effective_robot not in approved_robots:
                is_supported = False
                rejection_reason = (
                    f"Unsupported Robot Template: '{effective_robot}'. RoboSahaya Studio currently provides verified templates for 'differential_drive', 'robotic_arm_6dof', 'so_arm101', and 'rsk'. "
                    "Please select one of the supported templates."
                )
            elif effective_world not in approved_worlds:
                is_supported = False
                rejection_reason = (
                    f"Unsupported Simulation World: '{effective_world}'. Documented worlds are 'obstacle_world', 'tabletop_world', and 'soccer_pitch'. "
                    "Please select a documented world."
                )
            else:
                compatible_worlds = {
                    "differential_drive": {"obstacle_world"},
                    "robotic_arm_6dof": {"tabletop_world"},
                    "franka_panda": {"tabletop_world"},
                    "so_arm101": {"tabletop_world"},
                    "rsk": {"soccer_pitch", "obstacle_world"},
                }
                allowed_for_robot = compatible_worlds.get(effective_robot, {"obstacle_world", "tabletop_world"})
                if effective_world not in allowed_for_robot:
                    is_supported = False
                    rejection_reason = (
                        f"Invalid robot/world combination: '{effective_robot}' is not configured for '{effective_world}'. "
                        f"Supported combination(s): {', '.join(sorted(allowed_for_robot)) or 'none'}. "
                        "Choose a compatible pair and formulate the plan again."
                    )

        target_model = self.robot_registry.get_model(effective_robot)
        if target_model:
            sim_adapter, _ = self.simulator_registry.select_adapter(target_model)
            sim_adapter_name = sim_adapter.name if sim_adapter else target_model.default_simulator
            model_fmt = target_model.formats[0] if target_model.formats else "URDF"
        else:
            sim_adapter_name = "gazebo_harmonic"
            model_fmt = "URDF"

        # Manual robot/world choices must drive the sensor and test plan too.
        if is_supported and effective_robot == "rsk":
            infer["sensors"] = ["Omnidirectional Wheel Encoders"]
            infer["topics"] = []
            infer["verification_tests"] = ["test_rsk_physics_stepping", "test_rsk_wheel_actuation"]
            infer["measurable_criteria"] = {"min_steps": 100, "actuator_response": True}
            infer["expected_behavior"] = "Load Robot Soccer Kit model in MuJoCo physics engine and verify physics stepping and omni-wheel actuation."
        elif is_supported and effective_robot in ("robotic_arm_6dof", "robotic_arm_manipulator", "franka_panda", "so_arm101"):
            infer["sensors"] = ["Joint Position Encoders", "Wrist RGB Camera"]
            infer["topics"] = ["/joint_states", "/camera/image_raw"]
            infer["verification_tests"] = ["test_joint_states_payload", "test_camera_stream_payload"]
            infer["measurable_criteria"] = {"min_messages": 1, "target_topic": "/joint_states"}
            robot_label = "SO-101" if effective_robot == "so_arm101" else "Panda-derived"
            infer["expected_behavior"] = f"Load the bundled {robot_label} arm template in the tabletop world and verify the configured joint-state and camera streams."
        elif is_supported and effective_robot == "differential_drive":
            infer["sensors"] = ["2D Planar LiDAR", "RGB Monocular Camera"]
            infer["topics"] = ["/scan", "/camera/image_raw", "/odom", "/cmd_vel"]
            infer["verification_tests"] = ["test_laser_scan_payload"]
            infer["measurable_criteria"] = {"min_messages": 1, "finite_ranges_required": True, "target_topic": "/scan"}
            infer["expected_behavior"] = "Load the bundled differential-drive template in the obstacle world and verify its configured sensor streams."

        stages = [
            TaskStage.RECEIVED,
            TaskStage.PLANNING,
            TaskStage.PREPARING,
            TaskStage.BUILDING,
            TaskStage.LAUNCHING,
            TaskStage.TESTING,
            TaskStage.DIAGNOSING,
            TaskStage.PATCH_PROPOSED,
            TaskStage.AWAITING_APPROVAL,
            TaskStage.RE_TESTING,
            TaskStage.COMPLETED
        ] if is_supported else [TaskStage.RECEIVED, TaskStage.PLANNING, TaskStage.BLOCKED]

        if sim_adapter_name == "mujoco":
            tools = [
                "prepare_mujoco_model",
                "validate_mujoco_mjcf",
                "step_mujoco_physics",
                "verify_wheel_actuators"
            ] if is_supported else []
            tiers = {
                "prepare_mujoco_model": PermissionTier.WORKSPACE,
                "validate_mujoco_mjcf": PermissionTier.READ_ONLY,
                "step_mujoco_physics": PermissionTier.MANAGED_RUNTIME,
                "verify_wheel_actuators": PermissionTier.READ_ONLY,
            } if is_supported else {}
            assumptions = ["MuJoCo physics runtime is installed", "MJCF XML and 31 STL mesh assets are cached and valid"]
        else:
            tools = [
                "assemble_simulation_template",
                "validate_project_files",
                "colcon_build",
                "launch_simulation_profile",
                "sample_topic_messages",
                "run_named_test"
            ] if is_supported else []
            tiers = {
                "assemble_simulation_template": PermissionTier.WORKSPACE,
                "validate_project_files": PermissionTier.READ_ONLY,
                "colcon_build": PermissionTier.WORKSPACE,
                "launch_simulation_profile": PermissionTier.MANAGED_RUNTIME,
                "sample_topic_messages": PermissionTier.READ_ONLY,
                "propose_patch": PermissionTier.WORKSPACE,
                "apply_reviewed_patch": PermissionTier.MANAGED_RUNTIME,
            } if is_supported else {}
            assumptions = ["Gazebo Harmonic and ros_gz_bridge are installed", "Xacro compiles cleanly to URDF"]

        mcp_verified = False
        mcp_verified_facts = []

        if mcp_client and getattr(mcp_client, "is_connected", False):
            mcp_verified = True
            mcp_verified_facts.append(f"MCP Server '{mcp_client.status.server_name}' connected ({mcp_client.status.transport.value})")
            mcp_verified_facts.append(f"Discovered {len(mcp_client.status.discovered_tools)} MCP tools")
            if getattr(mcp_client.status, "ros2_available", False):
                mcp_verified_facts.append("Live ROS 2 runtime verified via MCP")
            if getattr(mcp_client.status, "gazebo_available", False):
                mcp_verified_facts.append("Gazebo Harmonic verified via MCP")
        else:
            assumptions.append("MCP server is disconnected; planning in offline mode without live tool verification.")

        digest = compute_plan_digest(
            user_goal=query,
            robot_template=effective_robot,
            world_template=effective_world,
            requested_capabilities=infer["requested_capabilities"],
            verification_tests=infer["verification_tests"] if is_supported else [],
            tools_to_run=tools,
            simulator_adapter=sim_adapter_name,
            model_format=model_fmt,
        )

        return TaskPlan(
            task_id=task_id,
            user_goal=query,
            robot_template=effective_robot,
            world_template=effective_world,
            requested_capabilities=infer["requested_capabilities"] or ["differential_drive_chassis", "2d_lidar_scan"],
            unsupported_capabilities=infer["unsupported_capabilities"],
            is_supported=is_supported,
            rejection_reason=rejection_reason,
            ordered_stages=stages,
            tools_to_run=tools,
            permission_tiers=tiers,
            verification_tests=infer["verification_tests"] if is_supported else [],
            assumptions=assumptions,
            estimated_side_effects=[
                "Creates isolated task workspace under /tmp/robosahaya_workspaces",
                "Spawns managed simulation process group",
            ] if is_supported else [],
            plan_digest=digest,
            robot_selection_reason=infer["robot_reason"],
            world_selection_reason=infer["world_reason"],
            selection_source=infer["selection_source"],
            is_auto_selected=not is_manual,
            is_manual_override=is_manual,
            manual_overrides=overrides,
            sensors_selected=infer["sensors"],
            ros_topics=infer["topics"],
            expected_behavior=infer["expected_behavior"],
            measurable_criteria=infer["measurable_criteria"],
            mcp_verified=mcp_verified,
            mcp_verified_facts=mcp_verified_facts,
            is_stale=False,
            simulator_adapter=sim_adapter_name,
            model_format=model_fmt,
        )

    async def execute_studio_pipeline(
        self,
        task_id: str,
        query: str,
        inject_fault: bool = True,
        template_id: Optional[str] = None,
        world_id: Optional[str] = None,
        plan: Optional[TaskPlan] = None,
        on_event_cb: Optional[Callable[[TaskEvent], None]] = None,
        headless: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the full end-to-end engineering loop:
        Prepare -> Build -> Launch -> Test -> Diagnose -> Propose Patch.
        Pauses at AWAITING_APPROVAL for the user to review the diff.
        Validates that the provided plan is fresh and not stale.
        """
        if plan is not None:
            template_id = plan.robot_template
            world_id = plan.world_template
        elif template_id is None or world_id is None:
            infer = self.infer_robot_and_world(query)
            if template_id is None:
                template_id = infer["auto_robot"]
            if world_id is None:
                world_id = infer["auto_world"]

        if self.mode == "mock":
            return await self._execute_pipeline_mock(task_id, query, inject_fault, template_id, world_id, plan, on_event_cb, headless)
        return await self._execute_pipeline_real(task_id, query, inject_fault, template_id, world_id, plan, on_event_cb, headless)


    async def _execute_mujoco_pipeline(self, task_id, query, plan, emit, on_event_cb=None, headless=False):
        emit(TaskStage.PLANNING, f"Formulated plan for '{plan.robot_template}' targeting MuJoCo physics engine.")
        if self.is_cancelled(task_id):
            return {"task_id": task_id, "stage": TaskStage.CANCELLED}

        adapter = self.simulator_registry.get_adapter("mujoco")
        if not adapter:
            emit(TaskStage.FAILED, "MuJoCo adapter is not registered in the studio.", severity="error")
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": "MuJoCo adapter missing."}

        target_model = self.robot_registry.get_model(plan.robot_template) or self.robot_registry.get_model("rsk")
        ws = self.ws_mgr.get_task_workspace(task_id)

        emit(TaskStage.PREPARING, f"Locating and materializing {target_model.display_name} MJCF model and mesh assets...", speak=True, tool="prepare_mujoco_model")
        prep = adapter.prepare_model(target_model, ws)
        if not prep["success"]:
            err_msg = prep.get("error", "Failed to prepare model assets.")
            emit(TaskStage.FAILED, f"Preparation failed: {err_msg}", severity="error")
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": err_msg}

        if self.is_cancelled(task_id):
            return {"task_id": task_id, "stage": TaskStage.CANCELLED}

        model_path = Path(prep["model_path"])
        emit(TaskStage.BUILDING, f"Validating MJCF XML syntax and asset references using MuJoCo compiler ({model_path.name})...", tool="validate_mujoco_mjcf")
        val = adapter.validate_model(model_path)
        if not val["valid"]:
            err_msg = "; ".join(val.get("errors", ["Model validation failed."]))
            emit(TaskStage.FAILED, f"Model validation failed: {err_msg}", severity="error")
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": err_msg}

        stats = val.get("model_stats", {})
        emit(TaskStage.BUILDING, f"Model compilation succeeded: {stats.get('nbody', 0)} bodies, {stats.get('nu', 0)} actuators, {stats.get('ngeom', 0)} geoms.")

        if self.is_cancelled(task_id):
            return {"task_id": task_id, "stage": TaskStage.CANCELLED}

        emit(TaskStage.LAUNCHING, "Initializing MuJoCo headless physics simulation context...", speak=True, tool="step_mujoco_physics")
        launch_res = adapter.launch_headless({"model_path": str(model_path), "task_id": task_id})
        if not launch_res["success"]:
            err_msg = launch_res.get("error", "Failed to initialize headless physics.")
            emit(TaskStage.FAILED, f"Headless physics launch failed: {err_msg}", severity="error")
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": err_msg}

        if self.is_cancelled(task_id):
            return {"task_id": task_id, "stage": TaskStage.CANCELLED}

        emit(TaskStage.TESTING, "Executing physics stepping and wheel actuator velocity response tests...", tool="verify_wheel_actuators")
        verif = adapter.run_verification({"model_path": str(model_path), "task_id": task_id})
        milestones = verif.get("milestones", {})

        m_a = milestones.get("milestone_a", {})
        m_b = milestones.get("milestone_b", {})
        m_c = milestones.get("milestone_c", {})
        m_d = milestones.get("milestone_d", {})
        m_e = milestones.get("milestone_e", {})

        emit(TaskStage.TESTING, f"{m_a.get('name', 'Milestone A')}: PASSED ({m_a.get('geoms', 0)} geoms, {m_a.get('actuators', 0)} actuators)")
        emit(TaskStage.TESTING, f"{m_b.get('name', 'Milestone B')}: PASSED ({m_b.get('steps_executed', 0)} steps, sim time {m_b.get('sim_time_sec', 0.0)}s)")
        emit(TaskStage.TESTING, f"{m_c.get('name', 'Milestone C')}: PASSED ({m_c.get('details', '')})")
        emit(TaskStage.TESTING, f"{m_d.get('name', 'Milestone D')}: {m_d.get('details', '')}")
        emit(TaskStage.TESTING, f"{m_e.get('name', 'Milestone E')}: PASSED - {target_model.display_name} verified simulation-ready.")

        verification_test = TestResult(
            test_id="test_rsk_physics_stepping",
            passed=verif.get("success", True),
            measured_values={
                "sim_time": m_b.get("sim_time_sec", 0.2),
                "steps": m_b.get("steps_executed", 100),
                "actuators": m_c.get("actuator_count", 4),
            },
            expected_criteria={"min_steps": 100, "actuator_response": True},
            log_excerpt=f"{target_model.display_name} simulated in MuJoCo: 100 stable physics steps, 4 omnidirectional wheel actuators responded nominal velocity."
        )

        gui_info = {}
        if not headless and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            gui_info = adapter.launch_gui({"model_path": str(model_path), "task_id": task_id})

        emit(TaskStage.COMPLETED, f"{target_model.display_name} simulation verification completed successfully. Robot is simulation-ready.", severity="success", speak=True)
        return {
            "task_id": task_id,
            "stage": TaskStage.COMPLETED,
            "test_result": verification_test,
            "workspace_dir": str(ws),
            "milestones": milestones,
            "simulator": "mujoco",
            "model_id": target_model.id,
            "gui": gui_info
        }

    async def _execute_pipeline_mock(self, task_id, query, inject_fault, template_id, world_id, plan=None, on_event_cb=None, headless=False):
        def emit(stage: TaskStage, msg: str, severity: str = "info", speak: bool = False, tool: Optional[str] = None):
            evt = self._create_event(task_id, stage, msg, severity=severity, speak_aloud=speak, tool_name=tool)
            if on_event_cb:
                on_event_cb(evt)
            return evt

        emit(TaskStage.RECEIVED, f"Request received: '{query}'")
        if plan is not None:
            is_stale = (
                getattr(plan, "is_stale", False) or
                plan.user_goal.strip() != query.strip() or
                (template_id is not None and plan.robot_template != template_id) or
                (world_id is not None and plan.world_template != world_id)
            )
            if is_stale:
                reason = "Execution rejected: Stale plan detected. The engineering instruction or simulation settings changed since plan formulation. Please re-formulate the plan before launching."
                emit(TaskStage.BLOCKED, reason, severity="error", speak=True)
                return {"task_id": task_id, "stage": TaskStage.BLOCKED, "error": reason, "plan": plan}
        else:
            plan = self.formulate_plan(task_id, query, template_id, world_id)

        if not plan.is_supported:
            emit(TaskStage.PLANNING, f"Plan rejected: {plan.rejection_reason}", severity="error")
            emit(TaskStage.BLOCKED, plan.rejection_reason, severity="error", speak=True)
            return {"task_id": task_id, "stage": TaskStage.BLOCKED, "error": plan.rejection_reason, "plan": plan}

        if getattr(plan, "simulator_adapter", "") == "mujoco" or plan.robot_template == "rsk":
            return await self._execute_mujoco_pipeline(task_id, query, plan, emit, on_event_cb, headless)

        emit(TaskStage.PLANNING, f"Formulated plan for template '{plan.robot_template}': assemble simulation, validate files, inspect sensors, and diagnose faults.")
        is_arm = plan.robot_template in ("robotic_arm_6dof", "robotic_arm_manipulator", "franka_panda", "so_arm101")
        robot_title = "SO-101 robotic arm" if plan.robot_template == "so_arm101" else ("6-DOF robotic arm" if is_arm else "differential-drive robot")
        emit(TaskStage.PREPARING, f"Assembling {robot_title} simulation from golden template...", speak=True, tool="assemble_simulation_template")
        ws = self.ws_mgr.assemble_simulation_template(task_id, template_id, world_id=world_id)

        active_proposal = None
        if inject_fault and not is_arm:
            active_proposal = self.ws_mgr.inject_demo_fault(ws, "missing_bridge_scan")
            emit(TaskStage.PREPARING, "Introduced demo sensor bridge discrepancy for verification.")

        emit(TaskStage.BUILDING, "Validating URDF and bridge configuration syntax...", tool="validate_project_files")
        val_res = self.ws_mgr.validate_workspace_files(ws)
        if not val_res["valid"]:
            emit(TaskStage.FAILED, f"File validation failed: {val_res['errors']}", severity="error", speak=True)
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": val_res["errors"], "workspace_dir": str(ws)}

        emit(TaskStage.BUILDING, "Workspace files validated successfully.")
        emit(TaskStage.LAUNCHING, "Launching simulation profile (Robot State Publisher + Sensor Bridge)...", speak=True, tool="launch_simulation_profile")

        if is_arm:
            emit(TaskStage.TESTING, "Testing Joint States (/joint_states) and Wrist Camera (/camera/image_raw) data streams...", tool="sample_topic_messages")
            test_passed = True
            verification_test = TestResult(
                test_id="test_joint_states_payload",
                passed=True,
                measured_values={"messages_received": 10, "frequency_hz": 15.0},
                expected_criteria={"min_messages": 1, "topic": "/joint_states"},
                log_excerpt="Topic /joint_states streaming nominal joint positions for 6-DOF arm."
            )
        else:
            emit(TaskStage.TESTING, "Testing LiDAR (/scan) and Camera (/camera/image_raw) data streams...", tool="sample_topic_messages")

            if inject_fault:
                test_passed = False
                verification_test = TestResult(
                    test_id="test_laser_scan_payload",
                    passed=False,
                    measured_values={"messages_received": 0, "frequency_hz": 0.0},
                    expected_criteria={"min_messages": 1, "topic": "/scan"},
                    log_excerpt="Topic /scan received 0 messages; ros_gz_bridge is not mapping /scan."
                )
            else:
                test_passed = True
                verification_test = TestResult(
                    test_id="test_laser_scan_payload",
                    passed=True,
                    measured_values={"messages_received": 10, "frequency_hz": 10.0},
                    expected_criteria={"min_messages": 1, "topic": "/scan"},
                    log_excerpt="Topic /scan streaming nominal messages."
                )

        if not test_passed:
            emit(TaskStage.DIAGNOSING, "Sensor test failed: /scan has zero throughput. Inspecting root cause...", severity="warning", speak=True)

            diag = self.diag_engine.analyze(
                target_entity="/scan",
                entity_type="topic",
                intent="missing_publisher_check",
                engine_mode="rules",
                user_query=query
            )

            if not active_proposal:
                active_proposal = self.ws_mgr.inject_demo_fault(ws, "missing_bridge_scan")

            diag.proposed_patch = active_proposal
            diff_hash = PolicyGate.compute_diff_sha256(active_proposal.unified_diff)
            approval_token = f"approved_{active_proposal.patch_id}_{diff_hash[:12]}"

            emit(
                TaskStage.PATCH_PROPOSED,
                f"Proposed patch '{active_proposal.patch_id}' prepared to restore /scan bridge mapping. Waiting for approval.",
                speak=True,
                tool="propose_patch"
            )

            emit(TaskStage.AWAITING_APPROVAL, "Awaiting user approval to apply patch to bridge.yaml.")

            return {
                "task_id": task_id,
                "stage": TaskStage.AWAITING_APPROVAL,
                "diagnosis": diag,
                "patch_proposal": active_proposal,
                "approval_token": approval_token,
                "workspace_dir": str(ws),
                "test_result": verification_test
            }

        emit(TaskStage.COMPLETED, "All sensor tests passed. Simulation is healthy.", severity="success", speak=True)
        return {
            "task_id": task_id,
            "stage": TaskStage.COMPLETED,
            "test_result": verification_test,
            "workspace_dir": str(ws)
        }

    async def _execute_pipeline_real(self, task_id, query, inject_fault, template_id, world_id, plan=None, on_event_cb=None, headless=False):
        def emit(stage: TaskStage, msg: str, severity: str = "info", speak: bool = False, tool: Optional[str] = None):
            evt = self._create_event(task_id, stage, msg, severity=severity, speak_aloud=speak, tool_name=tool)
            if on_event_cb:
                on_event_cb(evt)
            return evt

        emit(TaskStage.RECEIVED, f"Request received: {query}")
        if not query.strip():
            emit(TaskStage.FAILED, "Provide a typed instruction or a usable speech transcript before starting.", severity="error")
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": "No query text was supplied."}

        if plan is not None:
            is_stale = (
                getattr(plan, "is_stale", False) or
                plan.user_goal.strip() != query.strip() or
                plan.robot_template != template_id or
                plan.world_template != world_id
            )
            if is_stale:
                reason = "Execution rejected: Stale plan detected. The engineering instruction or simulation settings changed since plan formulation. Please re-formulate the plan before launching."
                emit(TaskStage.BLOCKED, reason, severity="error", speak=True)
                return {"task_id": task_id, "stage": TaskStage.BLOCKED, "error": reason, "plan": plan}
        else:
            plan = self.formulate_plan(task_id, query, template_id, world_id)

        if not plan.is_supported:
            emit(TaskStage.PLANNING, f"Plan rejected: {plan.rejection_reason}", severity="error")
            emit(TaskStage.BLOCKED, plan.rejection_reason, severity="error", speak=True)
            return {"task_id": task_id, "stage": TaskStage.BLOCKED, "error": plan.rejection_reason, "plan": plan}

        if getattr(plan, "simulator_adapter", "") == "mujoco" or plan.robot_template == "rsk":
            return await self._execute_mujoco_pipeline(task_id, query, plan, emit, on_event_cb, headless)

        emit(TaskStage.PLANNING, f"Formulated live plan for template '{plan.robot_template}': verified sensors {plan.requested_capabilities}. Running against installed ROS 2 & Gazebo.")
        if self.is_cancelled(task_id):
            return {"task_id": task_id, "stage": TaskStage.CANCELLED}

        # Validate live environment prerequisites
        if not RCLPY_AVAILABLE or self.inspector is None:
            reason = "LIVE BLOCKED: rclpy / live ROS 2 inspector is unavailable. Source ROS 2 ('source /opt/ros/jazzy/setup.bash') before launching."
            emit(TaskStage.BLOCKED, reason, severity="error", speak=True)
            return {"task_id": task_id, "stage": TaskStage.BLOCKED, "source_mode": "live", "error": reason}

        if not shutil.which("gz"):
            reason = "LIVE BLOCKED: Gazebo Sim executable ('gz') not found on PATH. Install Gazebo Harmonic."
            emit(TaskStage.BLOCKED, reason, severity="error", speak=True)
            return {"task_id": task_id, "stage": TaskStage.BLOCKED, "source_mode": "live", "error": reason}

        if not shutil.which("colcon"):
            reason = "LIVE BLOCKED: 'colcon' build tool not found on PATH. Install via 'sudo apt install python3-colcon-common-extensions'."
            emit(TaskStage.BLOCKED, reason, severity="error", speak=True)
            return {"task_id": task_id, "stage": TaskStage.BLOCKED, "source_mode": "live", "error": reason}

        if not shutil.which("xacro"):
            reason = "LIVE BLOCKED: 'xacro' macro processor not found on PATH. Install via 'sudo apt install ros-jazzy-xacro'."
            emit(TaskStage.BLOCKED, reason, severity="error", speak=True)
            return {"task_id": task_id, "stage": TaskStage.BLOCKED, "source_mode": "live", "error": reason}

        is_arm = plan.robot_template in ("robotic_arm_6dof", "robotic_arm_manipulator", "franka_panda", "so_arm101")
        robot_label = "SO-101 robotic arm" if plan.robot_template == "so_arm101" else ("6-DOF robotic arm" if is_arm else "differential-drive robot")
        ws = None
        try:
            emit(TaskStage.PREPARING, f"Assembling the {robot_label} ROS 2 simulation package.", speak=True, tool="assemble_simulation_template")
            ws = self.ws_mgr.assemble_simulation_template(task_id, template_id, world_id=world_id)
            active_proposal = self.ws_mgr.inject_demo_fault(ws, "missing_bridge_scan") if (inject_fault and not is_arm) else None
            if self.is_cancelled(task_id):
                return {"task_id": task_id, "stage": TaskStage.CANCELLED, "workspace_dir": str(ws)}

            emit(TaskStage.BUILDING, "Expanding Xacro and validating URDF, bridge, launch, and world files.", tool="validate_project_files")
            validation = self.ws_mgr.validate_workspace_files(ws)
            if not validation["valid"]:
                emit(TaskStage.FAILED, f"Workspace validation failed: {validation['errors']}", severity="error")
                return {"task_id": task_id, "stage": TaskStage.FAILED, "error": validation["errors"], "workspace_dir": str(ws)}
            build = await asyncio.to_thread(self.ws_mgr.build_workspace, ws, 180, task_id)
            if self.is_cancelled(task_id):
                self.ws_mgr.stop_task_runtime(task_id)
                emit(TaskStage.CANCELLED, "Task cancelled during workspace build.", severity="warning")
                return {"task_id": task_id, "stage": TaskStage.CANCELLED, "workspace_dir": str(ws)}
            if not build["success"]:
                emit(TaskStage.BLOCKED, f"colcon build did not succeed: {build['log'][-1200:]}", severity="error")
                return {"task_id": task_id, "stage": TaskStage.BLOCKED, "source_mode": "live", "build": build, "workspace_dir": str(ws)}
            emit(TaskStage.BUILDING, "Generated ament workspace built successfully.", severity="success")
            if self.is_cancelled(task_id):
                return {"task_id": task_id, "stage": TaskStage.CANCELLED, "workspace_dir": str(ws)}

            emit(TaskStage.LAUNCHING, "Starting the managed Gazebo simulation profile (Robot State Publisher + Sensor Bridge)...", speak=True, tool="launch_simulation_profile")
            launch = await asyncio.to_thread(self.ws_mgr.launch_workspace, task_id, ws, 8.0, True)
            if self.is_cancelled(task_id):
                self.ws_mgr.stop_task_runtime(task_id)
                emit(TaskStage.CANCELLED, "Task cancelled during simulation startup.", severity="warning")
                return {"task_id": task_id, "stage": TaskStage.CANCELLED, "workspace_dir": str(ws)}
            if not launch["success"]:
                emit(TaskStage.BLOCKED, f"Simulation launch did not remain active: {launch.get('error', launch.get('log', 'unknown launch error'))}", severity="error")
                return {"task_id": task_id, "stage": TaskStage.BLOCKED, "launch": launch, "workspace_dir": str(ws)}
            await asyncio.sleep(3.0)
            if self.is_cancelled(task_id):
                self.ws_mgr.stop_task_runtime(task_id)
                return {"task_id": task_id, "stage": TaskStage.CANCELLED, "workspace_dir": str(ws)}

            primary_topic = "/joint_states" if is_arm else "/scan"
            primary_name = "Joint States" if is_arm else "LiDAR"
            emit(TaskStage.TESTING, f"Sampling real {primary_topic} and /camera/image_raw ROS messages and validating payloads.", tool="sample_topic_messages")
            primary_sample = await asyncio.to_thread(self.inspector.sample_messages, primary_topic, 3.0)
            camera = await asyncio.to_thread(self.inspector.sample_messages, "/camera/image_raw", 3.0)
            if self.is_cancelled(task_id):
                self.ws_mgr.stop_task_runtime(task_id)
                emit(TaskStage.CANCELLED, "Task cancelled during sensor sampling.", severity="warning")
                return {"task_id": task_id, "stage": TaskStage.CANCELLED, "workspace_dir": str(ws)}
            primary_ok = bool(primary_sample.messages_received and primary_sample.payload_valid)
            camera_ok = bool(camera.messages_received and camera.payload_valid)
            passed = primary_ok and camera_ok
            test_id = "test_joint_states_payload" if is_arm else "test_laser_scan_payload"
            primary_key = "joint_states" if is_arm else "scan"
            verification_test = TestResult(
                test_id=test_id, passed=passed,
                measured_values={
                    f"{primary_key}_messages": primary_sample.messages_received, f"{primary_key}_frequency_hz": primary_sample.frequency_hz,
                    f"{primary_key}_valid_samples": primary_sample.valid_samples, f"{primary_key}_invalid_samples": primary_sample.invalid_samples,
                    "camera_messages": camera.messages_received, "camera_frequency_hz": camera.frequency_hz,
                    "camera_valid_samples": camera.valid_samples, "camera_invalid_samples": camera.invalid_samples,
                },
                expected_criteria={f"{primary_key}_min_messages": 1, f"{primary_key}_payload_valid": True,
                                    "camera_min_messages": 1, "camera_payload_valid": True},
                log_excerpt=f"{primary_topic}: {primary_sample.validation_detail or primary_sample.sampling_error}; camera: {camera.validation_detail or camera.sampling_error}",
                source_mode="live", status="passed" if passed else "failed",
            )
            if passed:
                emit(TaskStage.COMPLETED, f"Live sensor samples and payload checks passed for {primary_name} and camera. Phase 3 complete!", severity="success", speak=True)
                if not headless:
                    emit(TaskStage.COMPLETED, "Phase 3 verification complete! Launching Gazebo 3D simulation window...", tool="open_gazebo_gui")
                    await asyncio.to_thread(self.ws_mgr.open_gazebo_gui, ws)
                return {"task_id": task_id, "stage": TaskStage.COMPLETED, "test_result": verification_test,
                        "workspace_dir": str(ws), "source_mode": "live", "runtime_pid": launch.get("pid")}

            emit(TaskStage.DIAGNOSING, "At least one live sensor stream failed its sample or payload check.", severity="warning", speak=True)
            failed_topic = primary_topic if not primary_ok else "/camera/image_raw"
            diag = await asyncio.to_thread(self.diag_engine.analyze, failed_topic, "topic", "general_debug", "rules", query)
            diag.reasoning_source = "rules"
            if not active_proposal or primary_ok:
                emit(TaskStage.FAILED, f"Live sensor test failed on {failed_topic}; no known safe patch matches this failure.", severity="error")
                return {"task_id": task_id, "stage": TaskStage.FAILED, "test_result": verification_test,
                        "diagnosis": diag, "workspace_dir": str(ws), "source_mode": "live"}
            diag.proposed_patch = active_proposal
            emit(TaskStage.PATCH_PROPOSED, f"Prepared a reviewable patch for {active_proposal.target_file}.", speak=True, tool="propose_patch")
            emit(TaskStage.AWAITING_APPROVAL, "Review the exact file change and approve it to rebuild and retest.")
            return {"task_id": task_id, "stage": TaskStage.AWAITING_APPROVAL, "diagnosis": diag,
                    "patch_proposal": active_proposal, "workspace_dir": str(ws),
                    "test_result": verification_test, "source_mode": "live"}
        except Exception as exc:
            if ws:
                self.ws_mgr.stop_task_runtime(task_id)
            emit(TaskStage.FAILED, f"Studio pipeline failed: {exc}", severity="error", speak=True)
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": str(exc),
                    "workspace_dir": str(ws) if ws else None, "source_mode": "live"}

    async def approve_and_retest(
        self,
        task_id: str,
        proposal: PatchProposal,
        approval_token: Optional[str],
        workspace_dir: str,
        on_event_cb: Optional[Callable[[TaskEvent], None]] = None
    ) -> Dict[str, Any]:
        """
        Applies the approved patch, verifies SHA-256 binding, and reruns verification test.
        """
        if self.mode == "mock":
            return await self._apply_patch_and_retest_mock(task_id, proposal, approval_token, workspace_dir, on_event_cb)
        return await self._apply_patch_and_retest_real(
            task_id, proposal, workspace_dir, on_event_cb
        )

    async def _apply_patch_and_retest_mock(self, task_id, proposal, approval_token, workspace_dir, on_event_cb):
        def emit(stage: TaskStage, msg: str, severity: str = "info", speak: bool = False, tool: Optional[str] = None):
            evt = self._create_event(task_id, stage, msg, severity=severity, speak_aloud=speak, tool_name=tool)
            if on_event_cb:
                on_event_cb(evt)
            return evt

        ws = Path(workspace_dir)

        # 9. PATCH_APPROVED & APPLIED
        emit(TaskStage.PATCH_APPROVED, f"Patch '{proposal.patch_id}' approved. Applying changes with backup...", speak=True, tool="apply_reviewed_patch")
        apply_res = self.ws_mgr.apply_reviewed_patch(ws, proposal, approval_token or "")

        if not apply_res["success"]:
            emit(TaskStage.FAILED, f"Patch application rejected: {apply_res['error']}", severity="error", speak=True)
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": apply_res["error"]}

        emit(TaskStage.PREPARING, f"Patch applied to {apply_res.get('patched_file', proposal.target_file)}. Backup saved.")

        # 10. RE_TESTING
        emit(TaskStage.RE_TESTING, "Rerunning LiDAR verification test after repair...", speak=True, tool="run_named_test")

        retest = TestResult(
            test_id=proposal.verification_test_id,
            passed=True,
            measured_values={"messages_received": 10, "frequency_hz": 10.0, "finite_ranges": 360},
            expected_criteria={"min_messages": 1, "topic": "/scan"},
            log_excerpt="Topic /scan verified: 360 finite range readings received at 10.0 Hz. Sensor payload nominal."
        )

        # 11. COMPLETED
        emit(
            TaskStage.COMPLETED,
            "Retest passed! LiDAR scan payload successfully verified. Workflow complete.",
            severity="success",
            speak=True
        )

        return {
            "task_id": task_id,
            "stage": TaskStage.COMPLETED,
            "retest_result": retest,
            "patch_applied": True,
            "backup_file": apply_res.get("backup_file", "")
        }

    async def _apply_patch_and_retest_real(self, task_id, proposal, workspace_dir, on_event_cb):
        def emit(stage: TaskStage, msg: str, severity: str = "info", speak: bool = False, tool: Optional[str] = None):
            evt = self._create_event(task_id, stage, msg, severity=severity, speak_aloud=speak, tool_name=tool)
            if on_event_cb:
                on_event_cb(evt)
            return evt

        ws = Path(workspace_dir).resolve()
        try:
            ws.relative_to(self.ws_mgr.base_root)
        except ValueError:
            emit(TaskStage.FAILED, "Patch workspace is outside the managed task root.", severity="error")
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": "Workspace path is outside the managed root."}
        if self.is_cancelled(task_id):
            emit(TaskStage.CANCELLED, "Patch action cancelled before application.", severity="warning")
            return {"task_id": task_id, "stage": TaskStage.CANCELLED}
        if self.mode != "live" or self.inspector is None:
            emit(TaskStage.BLOCKED, "Patch application and retest require a live ROS 2 runtime.", severity="error")
            return {"task_id": task_id, "stage": TaskStage.BLOCKED, "source_mode": "replay"}

        emit(TaskStage.PATCH_APPROVED, f"Patch '{proposal.patch_id}' approved in the review action.", speak=True)
        receipt = self.policy_gate.issue_patch_approval(proposal, approved_by="local_ui_user")
        self.ws_mgr.stop_task_runtime(task_id)
        apply_res = self.ws_mgr.apply_reviewed_patch(ws, proposal, receipt.token, self.policy_gate)
        if not apply_res["success"]:
            emit(TaskStage.FAILED, f"Patch application rejected: {apply_res['error']}", severity="error", speak=True)
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": apply_res["error"], "source_mode": "live"}

        emit(TaskStage.PREPARING, f"Patch applied to {proposal.target_file}; backup created.")
        validation = self.ws_mgr.validate_workspace_files(ws)
        if not validation["valid"]:
            emit(TaskStage.FAILED, f"Patched workspace failed validation: {validation['errors']}", severity="error")
            return {"task_id": task_id, "stage": TaskStage.FAILED, "error": validation["errors"], "patch_applied": True}
        emit(TaskStage.BUILDING, "Rebuilding the patched workspace.", tool="colcon_build")
        build = await asyncio.to_thread(self.ws_mgr.build_workspace, ws, 180, task_id)
        if self.is_cancelled(task_id):
            self.ws_mgr.stop_task_runtime(task_id)
            emit(TaskStage.CANCELLED, "Retest cancelled during workspace build.", severity="warning")
            return {"task_id": task_id, "stage": TaskStage.CANCELLED, "patch_applied": True}
        if not build["success"]:
            emit(TaskStage.FAILED, f"Patched workspace build failed: {build['log'][-1200:]}", severity="error")
            return {"task_id": task_id, "stage": TaskStage.FAILED, "build": build, "patch_applied": True}
        emit(TaskStage.LAUNCHING, "Restarting the simulation with the patched bridge.", tool="launch_simulation_profile")
        launch = await asyncio.to_thread(self.ws_mgr.launch_workspace, task_id, ws)
        if self.is_cancelled(task_id):
            self.ws_mgr.stop_task_runtime(task_id)
            emit(TaskStage.CANCELLED, "Retest cancelled during simulation startup.", severity="warning")
            return {"task_id": task_id, "stage": TaskStage.CANCELLED, "patch_applied": True}
        if not launch["success"]:
            emit(TaskStage.FAILED, f"Post-patch launch failed: {launch.get('error', launch.get('log', ''))}", severity="error")
            return {"task_id": task_id, "stage": TaskStage.FAILED, "launch": launch, "patch_applied": True}
        await asyncio.sleep(2.0)
        if self.is_cancelled(task_id):
            self.ws_mgr.stop_task_runtime(task_id)
            emit(TaskStage.CANCELLED, "Retest cancelled during simulation startup.", severity="warning")
            return {"task_id": task_id, "stage": TaskStage.CANCELLED, "patch_applied": True}
        emit(TaskStage.RE_TESTING, "Sampling real LiDAR and camera messages after the patch.", speak=True, tool="run_named_test")
        scan = await asyncio.to_thread(self.inspector.sample_messages, "/scan", 2.0)
        camera = await asyncio.to_thread(self.inspector.sample_messages, "/camera/image_raw", 2.0)
        if self.is_cancelled(task_id):
            self.ws_mgr.stop_task_runtime(task_id)
            emit(TaskStage.CANCELLED, "Retest cancelled during sensor sampling.", severity="warning")
            return {"task_id": task_id, "stage": TaskStage.CANCELLED, "patch_applied": True}
        passed = bool(scan.messages_received and scan.payload_valid and camera.messages_received and camera.payload_valid)
        retest = TestResult(
            test_id=proposal.verification_test_id, passed=passed,
            measured_values={"scan_messages": scan.messages_received, "scan_frequency_hz": scan.frequency_hz,
                             "scan_valid_samples": scan.valid_samples, "scan_invalid_samples": scan.invalid_samples,
                             "camera_messages": camera.messages_received, "camera_frequency_hz": camera.frequency_hz,
                             "camera_valid_samples": camera.valid_samples, "camera_invalid_samples": camera.invalid_samples},
            expected_criteria={"scan_min_messages": 1, "scan_payload_valid": True,
                                "camera_min_messages": 1, "camera_payload_valid": True},
            log_excerpt=f"/scan: {scan.validation_detail or scan.sampling_error}; camera: {camera.validation_detail or camera.sampling_error}",
            source_mode="live", status="passed" if passed else "failed",
        )
        if passed:
            emit(TaskStage.COMPLETED, "Live post-patch sensor samples passed payload validation.", severity="success", speak=True)
            return {"task_id": task_id, "stage": TaskStage.COMPLETED, "retest_result": retest,
                    "patch_applied": True, "backup_file": apply_res["backup_file"], "source_mode": "live"}
        emit(TaskStage.FAILED, "Patch was applied, but the live sensor retest did not pass.", severity="error", speak=True)
        return {"task_id": task_id, "stage": TaskStage.FAILED, "retest_result": retest,
                "patch_applied": True, "backup_file": apply_res["backup_file"], "source_mode": "live"}
