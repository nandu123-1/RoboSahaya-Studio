"""Universal Robot Task Execution Engine for RoboSahaya Studio.

Supports multi-robot skills, closed-loop control, autonomous task planning,
and evidence-based verification across registered robots and simulator backends.
"""
from __future__ import annotations

from .contracts import (
    ActionStep,
    ActionResult,
    ExecutionObservation,
    ObservationRecord,
    PlacementPredicate,
    SkillAction,
    SkillResult,
    SkillStatus,
    SimulatorTaskController,
    TaskExecutionState,
    TaskSpecification,
    TaskType,
    VerificationCondition,
    VerificationResult,
    VerificationVerdict,
    compute_task_fingerprint,
)
from .capabilities import (
    CapabilityType,
    RobotCapability,
    RobotCapabilityResolver,
)
from .scene import (
    SceneEntity,
    WorldScene,
    get_world_scene,
)
from .skills import (
    BaseSkill,
    SkillRegistry,
    get_skill_registry,
)
from .interpreter import (
    TaskInterpreter,
    get_task_interpreter,
)
from .executor import (
    ClosedLoopExecutor,
)

__all__ = [
    # Contracts
    "ActionStep",
    "ActionResult",
    "ExecutionObservation",
    "ObservationRecord",
    "PlacementPredicate",
    "SkillAction",
    "SkillResult",
    "SkillStatus",
    "SimulatorTaskController",
    "TaskExecutionState",
    "TaskSpecification",
    "TaskType",
    "VerificationCondition",
    "VerificationResult",
    "VerificationVerdict",
    "compute_task_fingerprint",
    # Capabilities & Scene
    "CapabilityType",
    "RobotCapability",
    "RobotCapabilityResolver",
    "SceneEntity",
    "WorldScene",
    "get_world_scene",
    # Skills
    "BaseSkill",
    "SkillRegistry",
    "get_skill_registry",
    # Interpreter & Executor
    "TaskInterpreter",
    "get_task_interpreter",
    "ClosedLoopExecutor",
]
