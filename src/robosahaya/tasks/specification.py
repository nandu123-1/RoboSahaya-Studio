"""Structured Task Specification contracts for RoboSahaya Studio.

Defines the declarative task model, action steps, preconditions,
placement predicates, and verification criteria. Re-exports canonical contracts
from robosahaya.tasks.contracts.
"""
from __future__ import annotations

# Re-export canonical contract definitions
from .contracts import (
    ActionStep,
    ActionResult,
    ExecutionObservation,
    ObservationRecord,
    PlacementPredicate,
    SkillAction,
    SkillResult,
    SkillStatus,
    TaskExecutionState,
    TaskSpecification,
    TaskType,
    VerificationCondition,
    VerificationResult,
    VerificationVerdict,
    compute_task_fingerprint,
)

__all__ = [
    "TaskType",
    "TaskExecutionState",
    "SkillStatus",
    "VerificationVerdict",
    "ActionStep",
    "SkillAction",
    "ActionResult",
    "SkillResult",
    "ObservationRecord",
    "ExecutionObservation",
    "PlacementPredicate",
    "VerificationCondition",
    "VerificationResult",
    "TaskSpecification",
    "compute_task_fingerprint",
]
