"""Bounded MuJoCo viewer loop with exact named-actuator control."""
from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer


def run(
    model_path_str: str,
    velocity: float = 0.0,
    wheel_name: str = "wheel1_speed",
    action_duration_sec: float = 0.0,
    max_runtime_sec: float = 60.0,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Run physics and viewer together; command only the named actuator.

    ``action_duration_sec`` bounds the period for which the requested control
    remains applied. Physics continues with zeroed controls while the viewer is
    open, up to ``max_runtime_sec``. The caller may also set ``cancel_event``.
    """
    path = Path(model_path_str).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo model path '{path}' does not exist.")
    if not math.isfinite(velocity):
        raise ValueError("Actuator velocity must be finite.")
    if not math.isfinite(action_duration_sec) or not 0.0 <= action_duration_sec <= 10.0:
        raise ValueError("action_duration_sec must be between 0 and 10 seconds.")
    if not math.isfinite(max_runtime_sec) or not 0.05 <= max_runtime_sec <= 300.0:
        raise ValueError("max_runtime_sec must be between 0.05 and 300 seconds.")
    if action_duration_sec > max_runtime_sec:
        raise ValueError("Actuator action duration cannot exceed viewer runtime.")

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    control_id: Optional[int] = None
    joint_id: Optional[int] = None
    qpos_address: Optional[int] = None
    qvel_address: Optional[int] = None
    ctrl_range: Optional[list[float]] = None

    if action_duration_sec > 0.0:
        control_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, wheel_name)
        if control_id < 0:
            raise ValueError(f"Named actuator '{wheel_name}' is absent; no fallback actuator is used.")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_name)
        if joint_id < 0 or int(model.actuator_trnid[control_id, 0]) != joint_id:
            raise ValueError(f"Actuator '{wheel_name}' is not bound to its same-named wheel joint.")
        if bool(model.actuator_ctrllimited[control_id]):
            low, high = (float(value) for value in model.actuator_ctrlrange[control_id])
            ctrl_range = [low, high]
            if not low <= velocity <= high:
                raise ValueError(
                    f"Requested {velocity:.4f} rad/s is outside the model control range "
                    f"[{low:.4f}, {high:.4f}] rad/s for '{wheel_name}'."
                )
        else:
            ctrl_range = [float("-inf"), float("inf")]
        qpos_address = int(model.jnt_qposadr[joint_id])
        qvel_address = int(model.jnt_dofadr[joint_id])

    cancel_event = cancel_event or threading.Event()
    initial_position = float(data.qpos[qpos_address]) if qpos_address is not None else None
    action_was_active = action_duration_sec > 0.0
    action_position: Optional[float] = None
    action_velocity: Optional[float] = None
    data.ctrl[:] = 0.0
    if action_was_active and control_id is not None:
        data.ctrl[control_id] = velocity

    print(
        f"Starting MuJoCo viewer for {path.name}; "
        + (f"{wheel_name}={velocity:.4f} rad/s for {action_duration_sec:.3f}s."
           if action_was_active else "view-only mode; all actuators remain zero."),
        flush=True,
    )

    viewer_closed = False
    viewer_thread = None
    threads_before = set(threading.enumerate())
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            new_threads = set(threading.enumerate()) - threads_before
            for t in new_threads:
                if "_launch_internal" in str(getattr(t, "_target", "")):
                    viewer_thread = t
                    break
            # Start bounded wall-clock windows after the passive viewer opens;
            # startup time must not consume the requested physics action window.
            started = time.monotonic()
            action_deadline = started + action_duration_sec
            runtime_deadline = started + max_runtime_sec
            while viewer.is_running() and not cancel_event.is_set() and time.monotonic() < runtime_deadline:
                now = time.monotonic()
                if action_was_active and now >= action_deadline:
                    if qpos_address is not None and qvel_address is not None:
                        action_position = float(data.qpos[qpos_address])
                        action_velocity = float(data.qvel[qvel_address])
                    data.ctrl[:] = 0.0
                    action_was_active = False

                step_started = time.monotonic()
                mujoco.mj_step(model, data)
                viewer.sync()
                sleep_time = float(model.opt.timestep) - (time.monotonic() - step_started)
                if sleep_time > 0:
                    cancel_event.wait(min(sleep_time, max(0.0, runtime_deadline - time.monotonic())))
            viewer_closed = not viewer.is_running()
    finally:
        data.ctrl[:] = 0.0
        # Ensure the GLFW/GLX viewer thread shuts down cleanly before process exit
        # to prevent GLXBadDrawable/GLXBadContext race on display context destruction.
        if viewer_thread and viewer_thread.is_alive():
            viewer_thread.join(timeout=3.0)

    measured: Dict[str, Any] = {
        "sim_time_sec": float(data.time),
        "viewer_closed_by_user": viewer_closed,
        "cancelled": cancel_event.is_set(),
    }
    if control_id is not None and qpos_address is not None and qvel_address is not None:
        final_position = action_position if action_position is not None else float(data.qpos[qpos_address])
        final_velocity = action_velocity if action_velocity is not None else float(data.qvel[qvel_address])
        position_delta = final_position - float(initial_position)
        motion_verified = (
            math.isfinite(position_delta)
            and math.isfinite(final_velocity)
            and abs(position_delta) > 1e-4
            and abs(final_velocity) > 0.05
        )
        measured.update({
            "actuator_name": wheel_name,
            "control_range_rad_s": ctrl_range,
            "commanded_rad_s": velocity,
            "final_joint_position_rad": final_position,
            "joint_position_delta_rad": position_delta,
            "measured_joint_velocity_rad_s": final_velocity,
            "motion_verified": motion_verified,
        })
        print(
            f"MuJoCo measurement: actuator={wheel_name}, "
            f"joint_delta={position_delta:.6f} rad, velocity={final_velocity:.4f} rad/s, "
            f"motion_verified={motion_verified}.",
            flush=True,
        )
    return measured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("velocity", nargs="?", type=float, default=0.0)
    parser.add_argument("wheel_name", nargs="?", default="wheel1_speed")
    parser.add_argument("action_duration_sec", nargs="?", type=float, default=0.0)
    parser.add_argument("max_runtime_sec", nargs="?", type=float, default=60.0)
    args = parser.parse_args()

    cancellation = threading.Event()

    def request_cancel(_signum: int, _frame: Any) -> None:
        cancellation.set()

    signal.signal(signal.SIGINT, request_cancel)
    signal.signal(signal.SIGTERM, request_cancel)
    try:
        run(
            args.model_path,
            args.velocity,
            args.wheel_name,
            args.action_duration_sec,
            args.max_runtime_sec,
            cancellation,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"MuJoCo runner failed: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())