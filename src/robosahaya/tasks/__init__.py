"""Universal Robot Task Execution Engine for RoboSahaya Studio.

Supports multi-robot skills, closed-loop control, autonomous task planning,
and evidence-based verification across registered robots and simulator backends.
"""
from __future__ import annotations

from .specification import (
    ActionStep,
    PlacementPredicate,
    TaskSpecification,
    TaskType,
    VerificationCondition,
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
    SkillResult,
    SkillStatus,
    get_skill_registry,
)
from .interpreter import (
    TaskInterpreter,
    get_task_interpreter,
)
from .executor import (
    ClosedLoopExecutor,
    ExecutionObservation,
    TaskExecutionState,
)

__all__ = [
    "ActionStep",
    "PlacementPredicate",
    "TaskSpecification",
    "TaskType",
    "VerificationCondition",
    "CapabilityType",
    "RobotCapability",
    "RobotCapabilityResolver",
    "SceneEntity",
    "WorldScene",
    "get_world_scene",
    "BaseSkill",
    "SkillRegistry",
    "SkillResult",
    "SkillStatus",
    "get_skill_registry",
    "TaskInterpreter",
    "get_task_interpreter",
    "ClosedLoopExecutor",
    "ExecutionObservation",
    "TaskExecutionState",
]
