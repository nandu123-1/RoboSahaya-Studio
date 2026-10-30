"""Simulator integration tests.

The RSK tests load the real cached MuJoCo Menagerie model. Tests prefixed
``test_mocked_`` replace only the passive viewer; they retain real MuJoCo
physics. The Gazebo test starts the installed Gazebo Sim server when present.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from robosahaya.simulators.task_execution import (
    GazeboTaskExecutionAdapter,
    MujocoTaskExecutionAdapter,
    SimulatorTaskExecutionAdapter,
)
from robosahaya.simulators.mujoco_adapter import MujocoAdapter
from robosahaya.simulators.mujoco_runner import run as run_mujoco_viewer
from robosahaya.tasks.interpreter import TaskInterpreter
from robosahaya.tasks.contracts import (
    ActionResult,
    ObservationRecord,
    SkillStatus,
    VerificationResult,
    VerificationVerdict,
)
from robosahaya.tasks.executor import TaskExecutionState
from robosahaya.tasks.specification import ActionStep, TaskSpecification, TaskType


def _rsk_scene_path() -> Path:
    candidates = (
        Path.home() / ".cache/robot_descriptions/mujoco_menagerie/robot_soccer_kit/scene.xml",
        Path.home() / ".cache/robosahaya/robot_models/rsk/v1.0.0/model/scene.xml",
    )
    scene = next((path for path in candidates if path.is_file()), None)
    if scene is None:
        pytest.skip("Real RSK MuJoCo scene is not present in the local model cache.")
    pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")
    return scene


def _rsk_spec(
    *,
    task_id: str = "rsk_wheel_task_01",
    wheel_name: str = "wheel1_speed",
    velocity: float = 5.44,
    duration_sec: float = 0.2,
    timeout_sec: float = 3.0,
) -> TaskSpecification:
    return TaskSpecification(
        task_id=task_id,
        original_instruction=f"Set {wheel_name} to {velocity} rad/s.",
        task_type=TaskType.MOBILE_WHEEL_CONTROL,
        canonical_robot_id="rsk",
        simulator_adapter="mujoco",
        world_id="soccer_pitch",
        timeout_sec=timeout_sec,
        action_sequence=[ActionStep(
            step_index=0,
            skill_id="set_wheel_velocity",
            description=f"Set {wheel_name} angular velocity.",
            parameters={
                "wheel_name": wheel_name,
                "velocity": velocity,
                "duration_sec": duration_sec,
            },
            timeout_sec=timeout_sec,
            expected_observation={"motion_verified": True},
        )],
    )


def test_integration_real_mujoco_rsk_wheel1_speed_moves_actual_wheel():
    """The named actuator must move its bound joint in the real RSK MJCF."""
    scene = _rsk_scene_path()
    mujoco = pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")
    model = mujoco.MjModel.from_xml_path(str(scene))
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel1_speed")
    assert actuator_id >= 0
    assert model.actuator_ctrlrange[actuator_id].tolist() == [-16.0, 16.0]

    result = asyncio.run(MujocoTaskExecutionAdapter().execute_task(
        _rsk_spec(), context={"model_path": scene}
    ))

    assert result["success"] is True
    assert result["state"] == TaskExecutionState.COMPLETED.value
    assert result["source_mode"] == "live"
    assert result["test_result"].source_mode == "live"
    assert isinstance(result["verification_result"], VerificationResult)
    assert result["verification_result"].verdict is VerificationVerdict.VERIFIED_SUCCESS
    assert isinstance(result["action_results"][0], ActionResult)
    assert isinstance(result["observations"][0], ObservationRecord)
    assert len(result["action_results"]) == 1
    action = result["action_results"][0]
    assert action.success is True
    assert action.measured_data["actuator_name"] == "wheel1_speed"
    assert action.measured_data["joint_name"] == "wheel1_speed"
    assert action.measured_data["commanded_rad_s"] == pytest.approx(5.44)
    assert action.measured_data["control_range_rad_s"] == pytest.approx([-16.0, 16.0])
    assert action.measured_data["steps_simulated"] >= 1
    assert action.measured_data["motion_verified"] is True
    assert abs(action.measured_data["joint_position_delta_rad"]) > 0.1
    assert abs(action.measured_data["measured_joint_velocity_rad_s"]) > 5.0
    assert result["observations"][0].observed_state["actuator_name"] == "wheel1_speed"


def test_integration_planner_contract_preconditions_and_verification_pass_on_real_rsk():
    """Execute the exact ActionStep and predicates produced by Agent A's interpreter."""
    scene = _rsk_scene_path()
    specification = TaskInterpreter().interpret_instruction(
        "Set RSK wheel1_speed to 5.44 rad/s",
        robot_id_override="rsk",
        world_id_override="soccer_pitch",
        task_id="rsk_shared_contract_01",
    )
    assert specification.is_supported is True
    assert specification.preconditions == [
        "Robot rsk model loaded",
        "World soccer_pitch materialized",
    ]
    assert specification.action_sequence[0].preconditions == [
        "model_physics_initialized",
        "actuator_wheel1_speed_ready",
    ]
    assert specification.action_sequence[0].expected_observation == {"actuator_commanded": True}

    result = asyncio.run(MujocoTaskExecutionAdapter().execute_task(
        specification, context={"model_path": scene}
    ))

    assert result["success"] is True
    assert result["test_result"].passed is True
    assert result["observations"][0].observed_state["actuator_commanded"] is True
    assert result["observations"][0].observed_state["measured_joint_velocity"] == pytest.approx(5.44, abs=1.5)


def test_integration_shared_controller_executes_named_actuator_and_observes_real_state():
    """The registered SimulatorAdapter methods execute and observe real RSK state."""
    scene = _rsk_scene_path()
    mujoco = pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    adapter = MujocoAdapter()
    context = {
        "canonical_robot_id": "rsk",
        "model_path": scene,
        "mujoco_model": model,
        "mujoco_data": data,
        "timeout_sec": 1.0,
    }

    action = adapter.execute_skill(
        "set_wheel_velocity",
        {"wheel1_speed": 5.44, "duration_sec": 0.08},
        context,
    )
    observed = adapter.observe_state({"joint_name": "wheel1_speed"}, context)
    complete_state = adapter.observe_state({}, context)
    passed, details = adapter.verify_predicate(
        "velocity_bounds",
        {"joint_name": "wheel1_speed", "minimum": 0.5, "maximum": 16.0},
        context,
    )

    assert isinstance(action, ActionResult)
    assert action.success is True
    assert action.measured_data["actuator_name"] == "wheel1_speed"
    assert action.measured_data["commanded_rad_s"] == pytest.approx(5.44)
    assert action.measured_data["motion_verified"] is True
    assert all(float(control) == 0.0 for control in data.ctrl)
    assert observed["available"] is True
    assert observed["joint_velocity_rad_s"] > 0.5
    assert len(complete_state["joints"]["base_freejoint"]["position"]) == 7
    assert len(complete_state["joints"]["base_freejoint"]["velocity"]) == 6
    assert complete_state["sensors"] == {}
    assert passed is True, details


def test_shared_mujoco_controller_blocks_wrong_robot_identity():
    result = MujocoAdapter().execute_skill(
        "set_wheel_velocity",
        {"wheel1_speed": 5.44},
        {"canonical_robot_id": "unitree_go1"},
    )
    assert result.status is SkillStatus.BLOCKED
    assert result.success is False
    assert "canonical robot_id='rsk'" in result.error_reason


@pytest.mark.parametrize(
    ("wheel_name", "velocity", "expected_reason"),
    [
        ("not_a_wheel", 5.44, "absent from the loaded MuJoCo model"),
        ("wheel1_speed", 20.0, "exceeds model control range"),
    ],
)
def test_integration_rsk_rejects_unknown_actuator_and_model_limit(
    wheel_name: str,
    velocity: float,
    expected_reason: str,
):
    """Invalid requests are rejected instead of falling back or clipping."""
    scene = _rsk_scene_path()
    result = asyncio.run(MujocoTaskExecutionAdapter().execute_task(
        _rsk_spec(
            task_id="rsk_wheel_rejected_01",
            wheel_name=wheel_name,
            velocity=velocity,
        ),
        context={"model_path": scene},
    ))

    assert result["success"] is False
    assert result["state"] == TaskExecutionState.BLOCKED.value
    assert len(result["action_results"]) == 1
    assert result["action_results"][0].success is False
    assert expected_reason in result["action_results"][0].error_reason
    assert result["observations"][0].step_success is False


def test_gazebo_panda_pick_place_is_blocked_without_real_control_and_planning():
    """The model/world alone must not be reported as a successful grasp task."""
    repo_root = Path(__file__).resolve().parents[2]
    specification = TaskSpecification(
        task_id="panda_pick_blocked_01",
        original_instruction="Pick up the red cube and place it in the green bin.",
        task_type=TaskType.MANIPULATION_PICK_PLACE,
        canonical_robot_id="franka_panda",
        simulator_adapter="gazebo_harmonic",
        world_id="tabletop_world",
        target_objects=["target_cube_red"],
        destination="target_bin_green",
        action_sequence=[ActionStep(
            step_index=0,
            skill_id="pick_and_place",
            description="Pick and place the red cube.",
            parameters={"target_object_id": "target_cube_red", "destination_id": "target_bin_green"},
        )],
    )

    result = asyncio.run(SimulatorTaskExecutionAdapter(
        gazebo_adapter=GazeboTaskExecutionAdapter(repo_root=repo_root)
    ).execute_task(specification))

    assert result["success"] is False
    assert result["state"] == TaskExecutionState.BLOCKED.value
    assert result["simulator"] == "gazebo_harmonic"
    assert result["readiness"]["world_matches_specification"] is True
    assert result["readiness"]["required_world_entities"] == {
        "target_bin_green": True,
        "target_cube_red": True,
    }
    assert result["readiness"]["ros2_control_command_interfaces"] is False
    assert result["readiness"]["joint_trajectory_controller_configured"] is False
    assert result["readiness"]["gripper_controller_configured"] is False
    assert result["readiness"]["moveit_config_files"] == {
        "srdf": False,
        "kinematics": False,
        "controllers": False,
    }
    assert "No grasp" in result["error"]
    assert result["action_results"][0].success is False


class _MockPassiveViewer:
    """Test double for viewer lifecycle only; MuJoCo model/data remain real."""

    def __init__(self, data, *, sync_delay_sec: float = 0.0):
        self.data = data
        self.sync_delay_sec = sync_delay_sec
        self.running = True
        self.closed = False
    def is_running(self):
        return self.running

    def sync(self):
        if self.sync_delay_sec > 0:
            import time
            time.sleep(self.sync_delay_sec)

    def close(self):
        self.running = False
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_mocked_viewer_keeps_real_physics_advancing_and_closes():
    """Mock only the UI window and verify the adapter steps real model data."""
    scene = _rsk_scene_path()
    mujoco = pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")
    import importlib
    mujoco_viewer = importlib.import_module("mujoco.viewer")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    fake_viewer = _MockPassiveViewer(data)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mujoco_viewer, "launch_passive", lambda _model, _data: fake_viewer)
    try:
        result = asyncio.run(MujocoTaskExecutionAdapter().execute_task(
            _rsk_spec(task_id="rsk_viewer_mock_01", duration_sec=0.04),
            context={
                "mujoco_model": model,
                "mujoco_data": data,
                "viewer": True,
                "viewer_hold_sec": 0.04,
            },
        ))
    finally:
        monkeypatch.undo()

    assert result["success"] is True


def test_mocked_viewer_cancellation_stops_action_and_cleans_up():
    """Cancellation interrupts a bounded real-physics loop and closes the mock viewer."""
    import importlib

    mujoco = pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")
    mujoco_viewer = importlib.import_module("mujoco.viewer")

    async def run_case():
        scene = _rsk_scene_path()
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        fake_viewer = _MockPassiveViewer(data, sync_delay_sec=0.002)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mujoco_viewer, "launch_passive", lambda _model, _data: fake_viewer)
        adapter = MujocoTaskExecutionAdapter(max_task_runtime_sec=5.0)
        running = asyncio.create_task(adapter.execute_task(
            _rsk_spec(task_id="rsk_cancel_mock_01", duration_sec=1.0, timeout_sec=5.0),
            context={"mujoco_model": model, "mujoco_data": data, "viewer": True},
        ))
        try:
            await asyncio.sleep(0.04)
            assert adapter.cancel("rsk_cancel_mock_01") is True
            result = await asyncio.wait_for(running, timeout=2.0)
        finally:
            monkeypatch.undo()
        return result, fake_viewer, data

    result, fake_viewer, data = asyncio.run(run_case())
    assert result["state"] == TaskExecutionState.CANCELLED.value
    assert fake_viewer.closed is True


def test_mocked_viewer_runner_applies_named_command_then_stops_at_runtime(monkeypatch):
    """The runner applies a requested wheel velocity for a bounded action duration, then stops."""
    import importlib
    scene = _rsk_scene_path()
    mujoco = pytest.importorskip("mujoco", reason="MuJoCo Python package is not installed.")
    mujoco_viewer = importlib.import_module("mujoco.viewer")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    fake_viewer = _MockPassiveViewer(data)
    monkeypatch.setattr(mujoco_viewer, "launch_passive", lambda _model, _data: fake_viewer)

    measured = run_mujoco_viewer(
        str(scene),
        velocity=5.44,
        wheel_name="wheel1_speed",
        action_duration_sec=0.04,
        max_runtime_sec=0.2,
    )
    assert measured["actuator_name"] == "wheel1_speed"
    assert measured["commanded_rad_s"] == 5.44
    assert measured["motion_verified"] is True


def test_integration_gazebo_headless_world_runs_when_gazebo_is_installed():
    """Run the committed Gazebo world for bounded real physics iterations."""
    gz = shutil.which("gz")
    if gz is None:
        pytest.skip("Gazebo Sim executable is not installed.")

    world = Path(__file__).resolve().parents[2] / "simulation/worlds/obstacle_course/obstacle_world.sdf"
    assert world.is_file()
    env = os.environ.copy()
    env["GZ_PARTITION"] = f"robosahaya_test_{os.getpid()}"
    result = subprocess.run(
        [gz, "sim", "-s", "-r", "--iterations", "10", str(world)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert result.returncode == 0
