"""Closed-Loop Task Execution Engine and State Machine for RoboSahaya Studio.

Orchestrates multi-robot skill execution, intermediate simulation observation,
bounded recovery, and measurable verification without fabricated results.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from robosahaya.evidence_models import TaskEvent, TaskStage, TestResult
from .skills import SkillRegistry, SkillResult, SkillStatus, get_skill_registry
from .specification import ActionStep, TaskSpecification

logger = logging.getLogger(__name__)


class TaskExecutionState(str, Enum):
    """Execution state machine states."""
    TASK_RECEIVED = "task_received"
    TASK_PARSED = "task_parsed"
    CAPABILITIES_CHECKED = "capabilities_checked"
    SCENE_VALIDATED = "scene_validated"
    PLAN_READY = "plan_ready"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    OBSERVING = "observing"
    RECOVERING = "recovering"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ExecutionObservation(BaseModel):
    """Observation recorded after an action step execution."""
    model_config = ConfigDict(extra="ignore")

    step_index: int
    skill_id: str
    timestamp: float = Field(default_factory=time.time)
    preconditions_met: bool
    commanded_parameters: Dict[str, Any] = Field(default_factory=dict)
    observed_state: Dict[str, Any] = Field(default_factory=dict)
    step_success: bool
    recovery_attempted: bool = False
    details: str = ""


class ClosedLoopExecutor:
    """Manages closed-loop skill execution with intermediate observation and recovery."""

    def __init__(self, skill_registry: Optional[SkillRegistry] = None, max_retries: int = 2):
        self.skill_registry = skill_registry or get_skill_registry()
        self.max_retries = max_retries
        self._is_cancelled = False

    def cancel(self) -> None:
        """Interrupt and cancel running task safely."""
        self._is_cancelled = True

    async def execute_task(
        self,
        specification: TaskSpecification,
        context: Dict[str, Any],
        on_event_cb: Optional[Callable[[TaskEvent], None]] = None
    ) -> Dict[str, Any]:
        """Execute the structured task specification step-by-step with closed-loop feedback."""
        self._is_cancelled = False
        task_id = specification.task_id

        def emit(stage: TaskStage, msg: str, severity: str = "info", tool: Optional[str] = None):
            evt = TaskEvent(
                task_id=task_id,
                event_id=f"evt_{int(time.time()*1000)}",
                stage=stage,
                severity=severity,
                user_facing_message=msg,
                tool_name=tool,
            )
            if on_event_cb:
                on_event_cb(evt)
            return evt

        # 1. Check if specification is supported
        if not specification.is_supported:
            reason = specification.rejection_reason or "Task rejected: unsupported capabilities or scene objects."
            emit(TaskStage.BLOCKED, reason, severity="error")
            return {
                "task_id": task_id,
                "state": TaskExecutionState.BLOCKED.value,
                "stage": TaskStage.BLOCKED,
                "success": False,
                "error": reason,
                "observations": [],
            }

        emit(TaskStage.PLANNING, f"Task parsed: {specification.task_type.value.upper()} ({len(specification.action_sequence)} action steps).")
        emit(TaskStage.PREPARING, f"Checking scene objects: {', '.join(specification.target_objects) or 'nominal'}. Robot: {specification.canonical_robot_id}.")

        observations: List[ExecutionObservation] = []
        overall_success = True
        failure_reason: Optional[str] = None

        # Add specification placement predicate into execution context
        if specification.placement_predicate:
            context["placement_predicate"] = specification.placement_predicate

        # 2. Iterate through action steps
        for step in specification.action_sequence:
            if self._is_cancelled:
                emit(TaskStage.CANCELLED, "Execution interrupted by cancellation request.", severity="warning")
                return {
                    "task_id": task_id,
                    "state": TaskExecutionState.CANCELLED.value,
                    "stage": TaskStage.CANCELLED,
                    "success": False,
                    "observations": observations,
                }

            skill = self.skill_registry.get_skill(step.skill_id)
            if not skill:
                err = f"Skill '{step.skill_id}' is not registered in the execution engine."
                emit(TaskStage.FAILED, err, severity="error")
                return {
                    "task_id": task_id,
                    "state": TaskExecutionState.FAILED.value,
                    "stage": TaskStage.FAILED,
                    "success": False,
                    "error": err,
                    "observations": observations,
                }

            emit(TaskStage.LAUNCHING, f"Step {step.step_index}/{len(specification.action_sequence)}: {step.description}", tool=step.skill_id)

            # Step execution loop with bounded retries
            step_success = False
            last_skill_res: Optional[SkillResult] = None
            retry_count = 0

            while retry_count <= self.max_retries and not step_success:
                try:
                    # Execute skill
                    last_skill_res = skill.execute(step.parameters, context)
                    step_success = last_skill_res.success
                except Exception as exc:
                    logger.error("Error executing skill '%s': %s", step.skill_id, exc)
                    last_skill_res = SkillResult(
                        skill_id=step.skill_id,
                        status=SkillStatus.FAILED,
                        success=False,
                        details=f"Exception during execution: {exc}",
                        error_reason=str(exc),
                    )
                    step_success = False

                if not step_success and retry_count < self.max_retries and step.is_critical:
                    retry_count += 1
                    emit(
                        TaskStage.TESTING,
                        f"Observation check failed: {last_skill_res.details}. Attempting recovery retry {retry_count}/{self.max_retries}...",
                        severity="warning",
                    )
                    time.sleep(0.2)
                else:
                    break

            # Record step observation
            obs = ExecutionObservation(
                step_index=step.step_index,
                skill_id=step.skill_id,
                preconditions_met=True,
                commanded_parameters=step.parameters,
                observed_state=last_skill_res.measured_data if last_skill_res else {},
                step_success=step_success,
                recovery_attempted=retry_count > 0,
                details=last_skill_res.details if last_skill_res else "",
            )
            observations.append(obs)

            if step_success:
                emit(TaskStage.TESTING, f"Step {step.step_index} verified: {last_skill_res.details}")
            else:
                emit(TaskStage.FAILED, f"Critical failure at Step {step.step_index}: {last_skill_res.details}", severity="error")
                overall_success = False
                failure_reason = f"Step {step.step_index} ({step.description}) failed: {last_skill_res.details}"
                break

        # 3. Post-execution Verification Phase
        if overall_success:
            emit(TaskStage.TESTING, "Evaluating final task verification conditions...")
            condition_failures = []
            for cond in specification.verification_conditions:
                # Find measured metric across observations
                measured = None
                for obs in reversed(observations):
                    if cond.metric in obs.observed_state:
                        measured = obs.observed_state[cond.metric]
                        break

                passed, cond_msg = cond.check(measured)
                if not passed and cond.is_mandatory:
                    condition_failures.append(cond_msg)
                emit(TaskStage.TESTING, f"Verification condition: {cond_msg}", severity="info" if passed else "error")

            if condition_failures:
                overall_success = False
                failure_reason = "Final verification failed: " + "; ".join(condition_failures)

        # 4. Final Verdict and TestResult Synthesis
        test_id = context.get("primary_test_id") or f"verify_{specification.task_type.value}"
        test_result = TestResult(
            test_id=test_id,
            passed=overall_success,
            measured_values={
                "steps_completed": len(observations),
                "total_steps": len(specification.action_sequence),
                "observations": [o.model_dump() for o in observations],
            },
            expected_criteria={"task_success": True},
            log_excerpt=(
                f"Task '{specification.original_instruction}' completed successfully."
                if overall_success else
                f"Task failed: {failure_reason}"
            ),
            status="passed" if overall_success else "failed",
        )

        final_stage = TaskStage.COMPLETED if overall_success else TaskStage.FAILED
        status_msg = (
            f"Task '{specification.original_instruction}' verified and COMPLETED successfully."
            if overall_success else
            f"Task FAILED: {failure_reason}"
        )
        emit(final_stage, status_msg, severity="success" if overall_success else "error", tool="task_verifier")

        return {
            "task_id": task_id,
            "state": TaskExecutionState.COMPLETED.value if overall_success else TaskExecutionState.FAILED.value,
            "stage": final_stage,
            "success": overall_success,
            "error": failure_reason,
            "test_result": test_result,
            "observations": observations,
            "specification": specification,
        }
