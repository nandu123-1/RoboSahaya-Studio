# RoboSahaya Studio: Shared Task Execution Contracts

This document formalizes the canonical task execution contracts between:
- **Agent A (Task Planning & Orchestration)**: Natural-language interpretation, entity grounding, capability-based skill selection, plan review, plan invalidation, and UI orchestration.
- **Agent B (Simulator Runtime Control)**: Simulator-specific execution (MuJoCo, Gazebo Harmonic), direct physics stepping, controller interfacing, intermediate observation sampling, and physical verification.

Source file: [`src/robosahaya/tasks/contracts.py`](file:///media/sunraku/workspace/gnani_ai_task_planning/src/robosahaya/tasks/contracts.py)

---

## 1. Architecture & Separation of Concerns

```
+----------------------------------------------------------------------------------+
| AGENT A: Planning & Orchestration                                                |
|                                                                                  |
|   User Prompt / Override                                                         |
|         │                                                                        |
|         ▼                                                                        |
|   TaskInterpreter ──► CapabilityResolver ──► SceneGrounding                     |
|         │                                                                        |
|         ▼                                                                        |
|   TaskSpecification (Declarative Plan, Steps, Verification Predicates)          |
|         │                                                                        |
|         ▼                                                                        |
|   UI Review Card (User Supervised Gate) ──► Approval                             |
+--------------------------------─────────┼────────────────────────────────────────+
                                          │ Passes TaskSpecification & Context
                                          ▼
+----------------------------------------------------------------------------------+
| AGENT B: Simulator Runtime Control (SimulatorTaskController)                     |
|                                                                                  |
|   execute_skill(skill_id, params, context) ──► ActionResult                      |
|         │                                                                        |
|         ▼                                                                        |
|   observe_state(query, context) ─────────────► ObservationRecord                 |
|         │                                                                        |
|         ▼                                                                        |
|   verify_predicate(type, spec, context) ─────► VerificationResult                |
+----------------------------------------------------------------------------------+
```

---

## 2. Core Contract Specifications

### 2.1 `TaskSpecification`
Declarative description of the planned task. Produced by Agent A, consumed by Agent B and the execution engine.

```python
class TaskSpecification(BaseModel):
    task_id: str
    original_instruction: str
    task_type: TaskType  # MANIPULATION_PICK_PLACE, MOBILE_WHEEL_CONTROL, etc.
    canonical_robot_id: str  # e.g. "franka_panda", "rsk", "differential_drive"
    robot_model_version: str = "v1.0.0"
    simulator_adapter: str  # "mujoco" or "gazebo_harmonic"
    world_id: str  # e.g. "tabletop_world", "soccer_pitch", "obstacle_world"

    # Entities and goals
    target_objects: List[str]  # e.g. ["target_cube_red"]
    destination: Optional[str]  # e.g. "target_bin_green"

    # Capabilities & prerequisites
    required_capabilities: List[str]
    required_sensors: List[str]
    required_controllers: List[str]

    # Planned execution sequence
    action_sequence: List[SkillAction]
    preconditions: List[str]

    # Verification criteria
    intermediate_criteria: Dict[str, Any]
    placement_predicate: Optional[PlacementPredicate]
    verification_conditions: List[VerificationCondition]
    final_criteria: Dict[str, Any]

    # Execution controls
    timeout_sec: float = 30.0
    safety_constraints: List[str]
    is_supported: bool = True
    rejection_reason: Optional[str] = None
    known_limitations: List[str]

    # Invalidation & State
    plan_fingerprint: str = ""
    is_stale: bool = False
    plan_status: Literal["draft", "approved", "rejected", "invalidated", "executing", "completed", "failed"] = "draft"
```

### 2.2 `SkillAction` (alias `ActionStep`)
An atomic executable action step planned by Agent A.

```python
class SkillAction(BaseModel):
    step_index: int  # 0, 1, 2, ...
    skill_id: str    # e.g. "pick_and_place", "set_wheel_velocity", "move_joints"
    description: str
    parameters: Dict[str, Any]  # e.g. {"wheel1_speed": 5.44, "duration_sec": 1.5}
    preconditions: List[str]
    expected_observation: Dict[str, Any]
    timeout_sec: float = 5.0
    is_critical: bool = True
    can_retry: bool = True
    max_retries: int = 2
    status: Literal["pending", "executing", "succeeded", "failed", "skipped", "recovering"] = "pending"
```

### 2.3 `ActionResult` (alias `SkillResult`)
The result returned by Agent B after executing a `SkillAction`.

```python
class ActionResult(BaseModel):
    skill_id: str
    status: SkillStatus  # READY, EXECUTING, SUCCEEDED, FAILED, BLOCKED, CANCELLED
    success: bool
    step_index: Optional[int] = None
    measured_data: Dict[str, Any]  # Actual telemetry, joint angles, object coords, etc.
    details: str
    error_reason: Optional[str] = None
    duration_sec: float = 0.0
    retry_count: int = 0
```

### 2.4 `ObservationRecord` (alias `ExecutionObservation`)
A recorded observation capturing state during and after an action step.

```python
class ObservationRecord(BaseModel):
    step_index: int
    skill_id: str
    timestamp: float
    preconditions_met: bool
    commanded_parameters: Dict[str, Any]
    observed_state: Dict[str, Any]
    step_success: bool
    recovery_attempted: bool = False
    details: str = ""
```

### 2.5 `VerificationResult`
The definitive result of task verification evaluated by Agent B and the verification engine.

```python
class VerificationResult(BaseModel):
    task_id: str
    verdict: VerificationVerdict  # VERIFIED_SUCCESS, VERIFIED_FAILURE, UNVERIFIED, BLOCKED
    success: bool
    placement_satisfied: Optional[bool] = None
    placement_details: Optional[str] = None
    condition_results: List[Dict[str, Any]]
    measured_values: Dict[str, Any]
    summary: str
```

### 2.6 `SimulatorTaskController` Interface
Implemented by `SimulatorAdapter` subclasses (`GazeboAdapter`, `MujocoAdapter`).

```python
class SimulatorTaskController(ABC):
    @abstractmethod
    def execute_skill(
        self,
        skill_id: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ActionResult:
        """Execute a concrete atomic skill against the simulation runtime."""
        pass

    @abstractmethod
    def observe_state(
        self,
        query: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query simulator state (joints, objects, sensor telemetry)."""
        pass

    @abstractmethod
    def verify_predicate(
        self,
        predicate_type: str,
        spec: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Verify an environment predicate or placement condition in simulation."""
        pass
```

---

## 3. Guarantees and Invariants

1. **No Fabricated Success**: A task is never reported as `completed` or `succeeded` unless physical observations verify the target conditions (e.g. geometric placement radius, velocity response).
2. **Honest Invalidation**: When user input, robot selection, or world settings change after plan review, `is_stale` is set to `True`, requiring re-planning before execution can begin.
3. **Capability Gating**: Only skills matching registered robot capabilities and supported simulator adapters are included in `action_sequence`. If requested capabilities cannot be satisfied, `is_supported` is set to `False` with an explicit `rejection_reason`.
4. **Simulator Independence**: Agent A never invokes simulator subprocesses directly; Agent A plans declarative `TaskSpecification` instances that Agent B executes through `SimulatorTaskController`.
