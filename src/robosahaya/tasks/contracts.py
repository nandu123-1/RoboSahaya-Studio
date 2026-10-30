"""Shared Task Execution Contracts for RoboSahaya Studio.

Defines the declarative task specification, skill action steps, action results,
observation records, verification results, and simulator task controller interfaces.
These contracts form the single source of truth between Agent A (Task Planning & Orchestration)
and Agent B (Simulator-Specific Runtime Control).
"""
from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


# =========================================================================
# TASK CLASSIFICATION & STATES
# =========================================================================

class TaskType(str, Enum):
    """Categorized robot task types across multi-robot domains."""
    MANIPULATION_PICK_PLACE = "manipulation_pick_place"
    MANIPULATION_TRAJECTORY = "manipulation_trajectory"
    MOBILE_WHEEL_CONTROL = "mobile_wheel_control"
    MOBILE_NAVIGATION = "mobile_navigation"
    DRONE_FLIGHT = "drone_flight"
    LEGGED_LOCOMOTION = "legged_locomotion"
    PERCEPTION_INSPECTION = "perception_inspection"
    SIMULATION_LIFECYCLE = "simulation_lifecycle"
    CUSTOM = "custom"


class TaskExecutionState(str, Enum):
    """Execution state machine states."""
    TASK_RECEIVED = "task_received"
    TASK_PARSED = "task_parsed"
    CAPABILITIES_CHECKED = "capabilities_checked"
    SCENE_VALIDATED = "scene_validated"
    PLAN_READY = "plan_ready"
    PLAN_APPROVED = "plan_approved"
    ACTION_SUBMITTED = "action_submitted"
    ACTION_EXECUTING = "action_executing"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    OBSERVING = "observing"
    INTERMEDIATE_ACHIEVED = "intermediate_achieved"
    RECOVERING = "recovering"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FINAL_VERIFIED = "final_verified"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class SkillStatus(str, Enum):
    """Execution status of an individual skill action."""
    READY = "ready"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class VerificationVerdict(str, Enum):
    """Final measurable verification verdicts."""
    VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
    VERIFIED_FAILURE = "VERIFIED_FAILURE"
    UNVERIFIED = "UNVERIFIED"
    BLOCKED = "BLOCKED"


# =========================================================================
# ACTION STEPS & SKILL CONTRACTS
# =========================================================================

class SkillAction(BaseModel):
    """An individual atomic action step in a planned skill sequence."""
    model_config = ConfigDict(extra="ignore")

    step_index: int = Field(ge=0, description="0-based sequence index")
    skill_id: str = Field(description="Canonical skill identifier in SkillRegistry")
    description: str = Field(description="Human-readable description of this action")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Typed parameters for the skill")
    preconditions: List[str] = Field(default_factory=list, description="Required preconditions before execution")
    expected_observation: Dict[str, Any] = Field(default_factory=dict, description="Expected sensor/state values")
    timeout_sec: float = Field(default=5.0, gt=0.0, description="Step timeout in seconds")
    is_critical: bool = Field(default=True, description="Whether failure of this step aborts the task")
    can_retry: bool = Field(default=True, description="Whether this step allows bounded recovery retries")
    max_retries: int = Field(default=2, ge=0, description="Maximum recovery retry attempts")
    status: Literal["pending", "executing", "succeeded", "failed", "skipped", "recovering"] = "pending"


# Alias for backwards compatibility
ActionStep = SkillAction


class ActionResult(BaseModel):
    """Result of an individual skill execution with measured observations."""
    model_config = ConfigDict(extra="ignore")

    skill_id: str
    status: SkillStatus = SkillStatus.READY
    success: bool = False
    step_index: Optional[int] = None
    measured_data: Dict[str, Any] = Field(default_factory=dict)
    details: str = ""
    error_reason: Optional[str] = None
    duration_sec: float = 0.0
    retry_count: int = 0


# Alias for backwards compatibility
SkillResult = ActionResult


class ObservationRecord(BaseModel):
    """Observation recorded after an action step execution."""
    model_config = ConfigDict(extra="ignore")

    step_index: int
    skill_id: str
    timestamp: float = Field(default=0.0)
    preconditions_met: bool = True
    commanded_parameters: Dict[str, Any] = Field(default_factory=dict)
    observed_state: Dict[str, Any] = Field(default_factory=dict)
    step_success: bool = False
    recovery_attempted: bool = False
    details: str = ""
    observation_source: str = "ground_truth_sim"
    lifecycle_stage: str = "action_executed"


# Alias for backwards compatibility
ExecutionObservation = ObservationRecord


# =========================================================================
# VERIFICATION PREDICATES & CONDITIONS
# =========================================================================

class PlacementPredicate(BaseModel):
    """Measurable geometric predicate verifying object placement."""
    model_config = ConfigDict(extra="ignore")

    target_entity_id: str
    destination_entity_id: str
    target_center_xyz: Tuple[float, float, float]
    allowed_radius: float = Field(default=0.10, gt=0.0)
    min_height: float = Field(default=0.78)
    max_height: float = Field(default=0.90)
    require_upright: bool = False

    def evaluate(self, actual_xyz: Tuple[float, float, float]) -> Tuple[bool, str]:
        """Evaluate whether actual object coordinates satisfy the placement criteria."""
        dx = actual_xyz[0] - self.target_center_xyz[0]
        dy = actual_xyz[1] - self.target_center_xyz[1]
        dist_2d = math.sqrt(dx * dx + dy * dy)
        actual_z = actual_xyz[2]

        if dist_2d > self.allowed_radius:
            return (
                False,
                f"Horizontal displacement error: {dist_2d:.4f}m exceeds allowed radius {self.allowed_radius:.4f}m."
            )

        if not (self.min_height <= actual_z <= self.max_height):
            return (
                False,
                f"Vertical position error: z={actual_z:.4f}m outside allowed surface range [{self.min_height:.4f}m, {self.max_height:.4f}m]."
            )

        return (
            True,
            f"Object placed successfully within target radius ({dist_2d:.4f}m <= {self.allowed_radius:.4f}m) and height bounds."
        )


class VerificationCondition(BaseModel):
    """An observable post-condition evaluated after task completion."""
    model_config = ConfigDict(extra="ignore")

    name: str
    metric: str
    comparator: Literal["eq", "neq", "gt", "gte", "lt", "lte", "within_tolerance", "is_true"]
    expected_value: Any
    tolerance: float = 0.05
    is_mandatory: bool = True

    def check(self, measured_value: Any) -> Tuple[bool, str]:
        """Check measured value against expectation."""
        if measured_value is None:
            return False, f"Metric '{self.metric}' was not measured (None)."

        try:
            if self.comparator == "is_true":
                ok = bool(measured_value)
            elif self.comparator == "within_tolerance":
                diff = abs(float(measured_value) - float(self.expected_value))
                ok = diff <= self.tolerance
            elif self.comparator == "gte":
                ok = float(measured_value) >= float(self.expected_value)
            elif self.comparator == "gt":
                ok = float(measured_value) > float(self.expected_value)
            elif self.comparator == "lte":
                ok = float(measured_value) <= float(self.expected_value)
            elif self.comparator == "lt":
                ok = float(measured_value) < float(self.expected_value)
            elif self.comparator == "eq":
                ok = measured_value == self.expected_value
            elif self.comparator == "neq":
                ok = measured_value != self.expected_value
            else:
                ok = False
        except (ValueError, TypeError) as exc:
            return False, f"Evaluation error for {self.metric}: {exc}"

        msg = f"{self.name}: measured {measured_value} (expected {self.expected_value}, {self.comparator})"
        return ok, msg


class VerificationResult(BaseModel):
    """Structured report of final task verification."""
    model_config = ConfigDict(extra="ignore")

    task_id: str
    verdict: VerificationVerdict = VerificationVerdict.UNVERIFIED
    success: bool = False
    placement_satisfied: Optional[bool] = None
    placement_details: Optional[str] = None
    condition_results: List[Dict[str, Any]] = Field(default_factory=list)
    measured_values: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


# =========================================================================
# STRUCTURED TASK SPECIFICATION
# =========================================================================

def compute_task_fingerprint(
    instruction: str,
    robot_id: str,
    world_id: str,
    action_sequence: List[SkillAction],
    simulator_adapter: str,
) -> str:
    """Generate deterministic SHA-256 fingerprint for plan invalidation detection."""
    steps_repr = [
        f"{s.step_index}:{s.skill_id}:{json.dumps(s.parameters, sort_keys=True)}"
        for s in action_sequence
    ]
    payload = f"{instruction.strip()}|{robot_id.strip()}|{world_id.strip()}|{simulator_adapter.strip()}|{';'.join(steps_repr)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TaskSpecification(BaseModel):
    """Authoritative structured specification for multi-robot task execution.
    
    This specification bridges natural language planning and simulator execution.
    """
    model_config = ConfigDict(extra="ignore")

    task_id: str
    original_instruction: str
    task_type: TaskType
    canonical_robot_id: str
    robot_model_version: str = "v1.0.0"
    simulator_adapter: str = "gazebo_harmonic"
    world_id: str = "tabletop_world"

    # Entities and goals
    target_objects: List[str] = Field(default_factory=list)
    destination: Optional[str] = None

    # Capabilities & prerequisites
    required_capabilities: List[str] = Field(default_factory=list)
    required_sensors: List[str] = Field(default_factory=list)
    required_controllers: List[str] = Field(default_factory=list)

    # Action sequence
    action_sequence: List[SkillAction] = Field(default_factory=list)
    preconditions: List[str] = Field(default_factory=list)

    # Verification criteria
    intermediate_criteria: Dict[str, Any] = Field(default_factory=dict)
    placement_predicate: Optional[PlacementPredicate] = None
    verification_conditions: List[VerificationCondition] = Field(default_factory=list)
    final_criteria: Dict[str, Any] = Field(default_factory=dict)

    # Execution controls
    timeout_sec: float = Field(default=30.0, gt=0.0)
    safety_constraints: List[str] = Field(default_factory=list)
    is_supported: bool = True
    rejection_reason: Optional[str] = None
    known_limitations: List[str] = Field(default_factory=list)

    # Plan Invalidation & Integrity
    plan_fingerprint: str = ""
    is_stale: bool = False
    plan_status: Literal["draft", "approved", "rejected", "invalidated", "executing", "completed", "failed"] = "draft"

    def compute_fingerprint(self) -> str:
        """Compute and set the plan's deterministic SHA-256 fingerprint."""
        fp = compute_task_fingerprint(
            instruction=self.original_instruction,
            robot_id=self.canonical_robot_id,
            world_id=self.world_id,
            action_sequence=self.action_sequence,
            simulator_adapter=self.simulator_adapter,
        )
        self.plan_fingerprint = fp
        return fp

    def is_invalidated_by(self, new_instruction: str, new_robot_id: str, new_world_id: str) -> bool:
        """Check if instruction or environment settings have drifted from this plan."""
        return (
            self.original_instruction.strip() != new_instruction.strip()
            or self.canonical_robot_id.strip() != new_robot_id.strip()
            or self.world_id.strip() != new_world_id.strip()
        )


# =========================================================================
# SIMULATOR TASK CONTROLLER INTERFACE (FOR AGENT B)
# =========================================================================

class SimulatorTaskController(ABC):
    """Abstract interface defining the simulator-side runtime contract.
    
    Agent B implements these methods in simulator adapters (e.g. MuJoCo, Gazebo Harmonic)
    to execute atomic skills, query simulation telemetry, and verify physical predicates.
    """

    @abstractmethod
    def execute_skill(
        self,
        skill_id: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        """Execute a concrete atomic skill against the simulation runtime.
        
        Args:
            skill_id: Canonical identifier of the skill (e.g. 'move_joints', 'set_wheel_velocity').
            params: Validated parameter dictionary for the skill.
            context: Execution context containing workspace path, runtime handles, etc.
            
        Returns:
            ActionResult with execution status, measured data, and diagnostic details.
        """
        pass

    @abstractmethod
    def observe_state(
        self,
        query: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query simulator state (e.g. object pose, joint positions, sensor readings, contacts).
        
        Args:
            query: Query parameters (e.g. {'entity_id': 'target_cube_red', 'attribute': 'position'}).
            context: Execution context containing active simulation session.
            
        Returns:
            Dictionary of observed state values.
        """
        pass

    @abstractmethod
    def verify_predicate(
        self,
        predicate_type: str,
        spec: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Verify an environment predicate or placement condition in simulation.
        
        Args:
            predicate_type: Type of predicate ('placement', 'velocity_bounds', 'contact').
            spec: Predicate parameters and tolerances.
            context: Execution context.
            
        Returns:
            Tuple of (passed: bool, message: str).
        """
        pass
