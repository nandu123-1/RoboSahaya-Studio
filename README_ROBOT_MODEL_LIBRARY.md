# RoboSahaya Cloud Robot Model Library & Readiness Architecture

RoboSahaya Studio includes a cloud-backed robot model infrastructure, on-demand asset downloader, verified local cache, and an authoritative 9-dimensional readiness verification engine.

---

## 1. Cloud Architecture & Database Schema

The cloud library separates metadata from large binary assets using **Supabase** (Postgres + Storage), with automatic zero-crash fallback to **Offline Mode** when unconfigured.

### SQL Migration
Schema migrations are located in [`migrations/001_cloud_robot_library.sql`](file:///media/sunraku/workspace/gnani_ai/migrations/001_cloud_robot_library.sql):
- `robot_models`: Canonical robot identifiers, display names, families, aliases, and licences.
- `robot_model_versions`: Version tracking, entrypoints, description formats (`URDF`, `MJCF`), and content fingerprints.
- `robot_model_assets`: Per-file relative paths, SHA-256 hashes, byte sizes, and object storage keys.
- `simulator_adapters`: Registered simulation engines (`gazebo_harmonic`, `mujoco`).
- `robot_runtime_compatibility`: Supported pairings between robot versions and simulator adapters.
- `robot_verification_runs`: Immutable records of test runs, environment fingerprints, and check outcomes.
- **Security**: Row Level Security (RLS) enabled with public read access and backend-only service role writes.

---

## 2. The 9-Dimensional Readiness Model

A catalogue entry or cloud download is never proof of simulation or execution readiness. RoboSahaya tracks 9 independent dimensions:

| Dimension | Evaluated Precondition | Positive Status |
|---|---|---|
| **1. Catalogue Identity** | Known canonical ID and registered metadata | `registered` / `catalogued` |
| **2. Cloud Availability** | Asset package registered in cloud storage or upstream | `cloud_available` / `local_only` |
| **3. Local Cache Status** | Assets downloaded and verified without corruption | `cache_verified` |
| **4. Description Validity** | XML/MJCF syntax parses and external meshes resolve | `model_validated` |
| **5. Simulator Compatibility** | Registered adapter matches model format | `compatible` |
| **6. Simulation Readiness** | Model loads and steps physics without error | `simulation_ready` |
| **7. Controller Readiness** | Task-required actuators respond to control inputs | `control_ready` |
| **8. Sensor Readiness** | Task-required sensors stream valid topic messages | `sensor_ready` |
| **9. Task Verification** | Defined task success criteria verified | `task_verified` |

**Composite Verdict**:
A robot reaches `EXECUTION_READY` only when all preconditions required for its specific execution profile pass. If mandatory controllers or sensors fail, the robot is marked `BLOCKED` with an explicit list of unmet requirements. If evidence exceeds its freshness limit, it is marked `STALE_EVIDENCE`.

---

## 3. On-Demand Downloads & Local Cache Safety

Located in [`src/robosahaya/model_cache.py`](file:///media/sunraku/workspace/gnani_ai/src/robosahaya/model_cache.py):
- **Cache Root**: `~/.cache/robosahaya/robot_models/<robot_id>/<model_version>/`
- **Path Traversal Protection**: Rejects `..`, absolute paths, and symlink escapes from the sandbox.
- **Cryptographic Verification**: Verifies SHA-256 digests against `manifest.json`.
- **Atomic Promotion**: Downloads and extracts into temporary staging directories (`.tmp_<uuid>`). Partial or failed downloads are purged and never overwrite valid caches.
- **Concurrency Locks**: Per-model thread locks prevent race conditions during parallel downloads.

---

## 4. Multi-Simulator Support: Gazebo & MuJoCo

- **Gazebo Harmonic**: Default runtime for URDF/Xacro models (`differential_drive`, `franka_panda`, `so_arm101`).
- **MuJoCo 3.16.0**: Dedicated runtime for native MJCF models (`rsk` / Robot Soccer Kit).
- **Format Incompatibility Guard**: Incompatible model formats are strictly rejected by the adapter registry. Gazebo never receives MJCF files, and MuJoCo never receives raw ROS 2 launch workspaces.

---

## 5. Configuration & Environment Variables

Copy `.env.example` to `.env` to configure cloud storage (optional):
```bash
# Supabase Cloud Robot Library Configuration (Optional)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-backend-service-role-key
SUPABASE_ROBOT_BUCKET=robot-model-packages
```
If unset, RoboSahaya Studio operates seamlessly in **Offline Mode**, utilizing local bundled models and local mock storage.

---

## 6. Running Tests

Run the full automated test suite:
```bash
pytest tests/
```
Output:
```text
============================= 101 passed in 19.74s =============================
```
