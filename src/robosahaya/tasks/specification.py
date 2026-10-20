"""Structured Task Specification contracts for RoboSahaya Studio.

Defines the declarative task model, action steps, preconditions,
placement predicates, and verification criteria.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


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


class ActionStep(BaseModel):
    """An individual atomic action step in a planned skill sequence."""
    model_config = ConfigDict(extra="ignore")

    step_index: int = Field(ge=0)
    skill_id: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[str] = Field(default_factory=list)
    expected_observation: Dict[str, Any] = Field(default_factory=dict)
    timeout_sec: float = Field(default=5.0, gt=0.0)
    is_critical: bool = True
    status: Literal["pending", "executing", "succeeded", "failed", "skipped", "recovering"] = "pending"


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


class TaskSpecification(BaseModel):
    """Authoritative structured specification for multi-robot task execution."""
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
    action_sequence: List[ActionStep] = Field(default_factory=list)
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
