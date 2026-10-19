# RoboSahaya Robot Model Library

RoboSahaya now includes a searchable catalogue layer for common robot families and an explicit action to fetch/cache one robot description at a time.

## Install the optional wider catalogue

From the repository root, with the RoboSahaya environment active:

```bash
pip install -e '.[robot-catalogue]'
```

The optional `robot_descriptions` package indexes 200+ robot descriptions and imports a selected model as a Python submodule. Its first import can download/cache that model's upstream description. RoboSahaya does not import model modules just to list or search the catalogue; a download is attempted only when the user presses **Prepare / Cache Description**.

## Current execution support vs catalogue support

- `differential_drive`: bundled RoboSahaya runtime template.
- `franka_panda`: maps to the existing `robotic_arm_6dof` source folder, which contains Panda link meshes and Panda joint names.
- Other catalogue entries (for example SO-101, UR5e, Unitree Go2, Crazyflie, Fetch, and humanoids) are **description-only** until a Gazebo launch adapter, correct spawn pose, physics/collision behavior, controllers, sensor/bridge configuration, and task tests have been implemented and verified.

Downloading a URDF or MJCF file proves only that a source description was fetched and parsed. It does not mean the model is ready to launch in Gazebo Harmonic, has active controllers, or can perform a task. MJCF-only models are especially not interchangeable with URDF/Gazebo models. RoboSahaya blocks execution for unverified models rather than silently substituting the Panda or differential-drive robot.

## Catalogue feature behavior

- Search by robot name or family.
- Filter by robot family.
- Check supported file format and upstream licence label.
- Explicitly prepare/cache one model description and record its path, SHA-256 digest, basic XML statistics, and inspection time in `~/.cache/robosahaya/robot_models/<model-id>/metadata.json`.
- Keep `gazebo_runtime_verified` and `controller_verified` set to `false` for downloaded models until runtime tests are implemented separately.

## Sources and licensing

Most non-bundled entries are indexed through [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py). The catalogue's licence field is a starting point; review the exact upstream repository and its current licence before redistributing any robot assets. A catalogue listing is not a licence grant.
