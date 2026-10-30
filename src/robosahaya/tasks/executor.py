"""Closed-Loop Task Execution Engine and State Machine for RoboSahaya Studio.

Orchestrates multi-robot skill execution, intermediate simulation observation,
bounded recovery, and measurable verification without fabricated results.
Coordinates with Agent B via documented contracts and SimulatorTaskController.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from robosahaya.evidence_models import TaskEvent, TaskStage, TestResult
from .contracts import (
    ActionResult,
    ActionStep,
    ExecutionObservation,
    ObservationRecord,
    SkillAction,
    SkillResult,
    SkillStatus,
    TaskExecutionState,
    TaskSpecification,
    VerificationResult,
    VerificationVerdict,
)
from .skills import SkillRegistry, get_skill_registry

logger = logging.getLogger(__name__)


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
            specification.plan_status = "rejected"
            emit(TaskStage.BLOCKED, reason, severity="error")
            verif_res = VerificationResult(
                task_id=task_id,
                verdict=VerificationVerdict.BLOCKED,
                success=False,
                summary=reason,
            )
            return {
                "task_id": task_id,
                "state": TaskExecutionState.BLOCKED.value,
                "stage": TaskStage.BLOCKED,
                "success": False,
                "error": reason,
                "observations": [],
                "specification": specification,
                "verification_result": verif_res,
            }

        specification.plan_status = "executing"
        emit(TaskStage.PLANNING, f"Task parsed: {specification.task_type.value.upper()} ({len(specification.action_sequence)} action steps).")
        emit(TaskStage.PREPARING, f"Checking scene objects: {', '.join(specification.target_objects) or 'nominal'}. Robot: {specification.canonical_robot_id}.")

        observations: List[ObservationRecord] = []
        overall_success = True
        failure_reason: Optional[str] = None

        # Add specification placement predicate into execution context
        if specification.placement_predicate:
            context["placement_predicate"] = specification.placement_predicate

        # 2. Iterate through action steps
        for step in specification.action_sequence:
            if self._is_cancelled:
                specification.plan_status = "failed"
                emit(TaskStage.CANCELLED, "Execution interrupted by cancellation request.", severity="warning")
                verif_res = VerificationResult(
                    task_id=task_id,
                    verdict=VerificationVerdict.UNVERIFIED,
                    success=False,
                    summary="Execution interrupted by user cancellation.",
                )
                return {
                    "task_id": task_id,
                    "state": TaskExecutionState.CANCELLED.value,
                    "stage": TaskStage.CANCELLED,
                    "success": False,
                    "observations": observations,
                    "specification": specification,
                    "verification_result": verif_res,
                }

            skill = self.skill_registry.get_skill(step.skill_id)
            if not skill:
                err = f"Skill '{step.skill_id}' is not registered in the execution engine."
                specification.plan_status = "failed"
                emit(TaskStage.FAILED, err, severity="error")
                verif_res = VerificationResult(
                    task_id=task_id,
                    verdict=VerificationVerdict.VERIFIED_FAILURE,
                    success=False,
                    summary=err,
                )
                return {
                    "task_id": task_id,
                    "state": TaskExecutionState.FAILED.value,
                    "stage": TaskStage.FAILED,
                    "success": False,
                    "error": err,
                    "observations": observations,
                    "specification": specification,
                    "verification_result": verif_res,
                }

            emit(TaskStage.LAUNCHING, f"Step {step.step_index}/{len(specification.action_sequence)}: {step.description}", tool=step.skill_id)

            # Step execution loop with bounded retries
            step_success = False
            last_skill_res: Optional[ActionResult] = None
            retry_count = 0
            allowed_retries = step.max_retries if getattr(step, "can_retry", True) else 0

            while retry_count <= allowed_retries and not step_success:
                try:
                    last_skill_res = skill.execute(step.parameters, context)
                    step_success = last_skill_res.success
                except Exception as exc:
                    logger.error("Error executing skill '%s': %s", step.skill_id, exc)
                    last_skill_res = ActionResult(
                        skill_id=step.skill_id,
                        status=SkillStatus.FAILED,
                        success=False,
                        details=f"Exception during execution: {exc}",
                        error_reason=str(exc),
                    )
                    step_success = False

                if not step_success and retry_count < allowed_retries and step.is_critical:
                    retry_count += 1
                    emit(
                        TaskStage.TESTING,
                        f"Observation check failed: {last_skill_res.details}. Attempting recovery retry {retry_count}/{allowed_retries}...",
                        severity="warning",
                    )
                    time.sleep(0.2)
                else:
                    break

            # Record step observation
            obs = ObservationRecord(
                step_index=step.step_index,
                skill_id=step.skill_id,
                timestamp=time.time(),
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
        condition_results: List[Dict[str, Any]] = []
        placement_satisfied = None
        placement_details = None

        if overall_success:
            emit(TaskStage.TESTING, "Evaluating final task verification conditions...")
            condition_failures = []
            for cond in specification.verification_conditions:
                measured = None
                for obs in reversed(observations):
                    if cond.metric in obs.observed_state:
                        measured = obs.observed_state[cond.metric]
                        break

                passed, cond_msg = cond.check(measured)
                condition_results.append({
                    "condition_name": cond.name,
                    "metric": cond.metric,
                    "measured": measured,
                    "expected": cond.expected_value,
                    "passed": passed,
                    "message": cond_msg,
                })
                if not passed and cond.is_mandatory:
                    condition_failures.append(cond_msg)
                emit(TaskStage.TESTING, f"Verification condition: {cond_msg}", severity="info" if passed else "error")

            # Check placement predicate specifically if present
            if specification.placement_predicate:
                for obs in reversed(observations):
                    if "object_xyz" in obs.observed_state:
                        pos = obs.observed_state["object_xyz"]
                        placement_satisfied, placement_details = specification.placement_predicate.evaluate(pos)
                        break

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

        specification.plan_status = "completed" if overall_success else "failed"
        final_stage = TaskStage.COMPLETED if overall_success else TaskStage.FAILED
        status_msg = (
            f"Task '{specification.original_instruction}' verified and COMPLETED successfully."
            if overall_success else
            f"Task FAILED: {failure_reason}"
        )
        emit(final_stage, status_msg, severity="success" if overall_success else "error", tool="task_verifier")

        verdict = VerificationVerdict.VERIFIED_SUCCESS if overall_success else VerificationVerdict.VERIFIED_FAILURE
        verif_res = VerificationResult(
            task_id=task_id,
            verdict=verdict,
            success=overall_success,
            placement_satisfied=placement_satisfied,
            placement_details=placement_details,
            condition_results=condition_results,
            measured_values={
                "steps_completed": len(observations),
                "total_steps": len(specification.action_sequence),
            },
            summary=status_msg,
        )

        return {
            "task_id": task_id,
            "state": TaskExecutionState.COMPLETED.value if overall_success else TaskExecutionState.FAILED.value,
            "stage": final_stage,
            "success": overall_success,
            "error": failure_reason,
            "test_result": test_result,
            "verification_result": verif_res,
            "observations": observations,
            "specification": specification,
        }
