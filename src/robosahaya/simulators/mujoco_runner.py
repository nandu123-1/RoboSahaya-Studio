"""Interactive active simulation runner for MuJoCo models in RoboSahaya Studio.

Launches the MuJoCo desktop viewer, applies commanded actuator velocities,
and continuously steps physics in real time.
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def run(model_path_str: str, velocity: float = 5.44, duration_sec: float = 60.0) -> None:
    path = Path(model_path_str)
    if not path.exists():
        print(f"Error: Model path '{path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)

    # Set initial actuator control signals
    # For RSK (3 omni-wheels), driving wheel1 and wheel2 opposite drives forward along Y axis
    if m.nu >= 2:
        d.ctrl[0] = velocity
        d.ctrl[1] = -velocity
    elif m.nu == 1:
        d.ctrl[0] = velocity

    print(f"Starting MuJoCo interactive viewer for {path.name} with velocity={velocity} rad/s...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time < duration_sec):
            step_start = time.time()

            # Continuously maintain control input during drive phase
            if time.time() - start_time < duration_sec:
                if m.nu >= 2:
                    d.ctrl[0] = velocity
                    d.ctrl[1] = -velocity
                elif m.nu == 1:
                    d.ctrl[0] = velocity

            mujoco.mj_step(m, d)
            viewer.sync()

            # Sync with simulation timestep for smooth real-time visualization
            elapsed = time.time() - step_start
            sleep_time = float(m.opt.timestep) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # After active driving phase, keep viewer open until user closes window
        while viewer.is_running():
            d.ctrl[:] = 0.0
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m robosahaya.simulators.mujoco_runner <model_path> [velocity]")
        sys.exit(1)

    xml_path = sys.argv[1]
    vel = float(sys.argv[2]) if len(sys.argv) > 2 else 5.44
    run(xml_path, vel)
