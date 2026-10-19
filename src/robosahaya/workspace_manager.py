import difflib
import atexit
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import threading
import xml.etree.ElementTree as ET
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from robosahaya.evidence_models import PatchProposal
from robosahaya.policy_gate import PolicyGate
from robosahaya.robot_catalog import RobotModelCatalog


PACKAGE_XML = """<?xml version=\"1.0\"?>
<package format=\"3\">
  <name>robosahaya_simulation</name><version>0.1.0</version>
  <description>Managed RoboSahaya Gazebo simulation profile.</description>
  <maintainer email=\"local@localhost\">RoboSahaya</maintainer><license>Apache-2.0</license>
  <buildtool_depend>ament_python</buildtool_depend>
  <exec_depend>ament_index_python</exec_depend>
  <exec_depend>launch</exec_depend><exec_depend>launch_ros</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend><exec_depend>ros_gz_sim</exec_depend>
  <exec_depend>ros_gz_bridge</exec_depend><exec_depend>rosgraph_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend><exec_depend>geometry_msgs</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
</package>
"""

PACKAGE_SETUP = """from glob import glob
from setuptools import setup

package_name = 'robosahaya_simulation'
setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/worlds', glob('worlds/*')),
        ('share/' + package_name + '/meshes', glob('meshes/*.*')),
        ('share/' + package_name + '/meshes/visual', glob('meshes/visual/*')),
        ('share/' + package_name + '/meshes/collision', glob('meshes/collision/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RoboSahaya',
    maintainer_email='local@localhost',
    description='Managed RoboSahaya Gazebo simulation profile',
    license='Apache-2.0',
)
"""

class WorkspaceManager:
    """Manages isolated per-task run workspaces, golden templates, validation, and safe patching."""

    def __init__(self, base_workspace_root: Optional[Path] = None, templates_dir: Optional[Path] = None):
        self.base_root = (base_workspace_root or Path("/tmp/robosahaya_workspaces")).resolve()
        self.base_root.mkdir(parents=True, exist_ok=True)
        
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.templates_dir = (templates_dir or (repo_root / "simulation" / "templates")).resolve()
        self.policy_gate = PolicyGate(self.base_root)
        self.robot_catalog = RobotModelCatalog()
        self._processes: dict[str, subprocess.Popen] = {}
        self._builds: dict[str, subprocess.Popen] = {}
        self._process_lock = threading.RLock()
        atexit.register(self.shutdown_all)

    def list_robot_models(self, search: str = "", family: str = "All"):
        """List registered and optionally discovered robot descriptions without downloading them."""
        return self.robot_catalog.list_models(search=search, family=family)

    def prepare_robot_description(self, model_id: str) -> Dict[str, Any]:
        """Fetch/cache and structurally inspect a selected model description on explicit request."""
        return self.robot_catalog.prepare_description(model_id)

    def get_task_workspace(self, task_id: str) -> Path:
        """Returns the isolated workspace directory for a specific task."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{4,80}", task_id):
            raise ValueError("Task IDs must contain only letters, digits, hyphens, or underscores.")
        clean_id = task_id
        ws = (self.base_root / clean_id).resolve()
        if not self.policy_gate.validate_workspace_path(ws, self.base_root):
            raise ValueError(f"Task ID '{task_id}' results in path traversal outside workspace root.")
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def assemble_simulation_template(
        self,
        task_id: str,
        template_id: str = "differential_drive",
        parameters: Optional[Dict[str, Any]] = None,
        world_id: str = "obstacle_world",
    ) -> Path:
        """
        Assembles a simulation workspace from verified golden templates using bounded parameters.
        """
        # Map alias
        if template_id in ("robotic_arm_manipulator", "franka_panda"):
            # franka_panda is a user-facing catalogue ID; the bundled source
            # directory keeps its legacy name for backwards compatibility.
            template_id = "robotic_arm_6dof"

        approved_templates = ("differential_drive", "robotic_arm_6dof", "so_arm101")
        if template_id not in approved_templates:
            raise ValueError(f"Template '{template_id}' is not an approved golden template.")

        approved_worlds = ("obstacle_world", "tabletop_world")
        if world_id not in approved_worlds:
            raise ValueError(f"World '{world_id}' is not an approved simulation world.")

        ws = self.get_task_workspace(task_id)
        source_dir = self.templates_dir / template_id

        if not source_dir.exists():
            raise FileNotFoundError(f"Template directory '{source_dir}' does not exist.")

        # Numeric and identifier values are bounded before they reach Xacro.
        if template_id == "robotic_arm_6dof":
            p = {
                "robot_name": "sahaya_arm",
                "base_radius": "0.12",
                "arm_reach": "0.75",
                "arm_payload": "3.0",
                "camera_rate": "15.0",
            }
            bounds = {
                "base_radius": (0.05, 0.5),
                "arm_reach": (0.2, 2.0),
                "arm_payload": (0.1, 25.0),
                "camera_rate": (1.0, 60.0),
            }
        elif template_id == "so_arm101":
            p = {
                "robot_name": "so101_arm",
                "camera_rate": "15.0",
            }
            bounds = {
                "camera_rate": (1.0, 60.0),
            }
        else:
            p = {
                "robot_name": "sahaya_bot",
                "chassis_length": "0.45",
                "chassis_width": "0.35",
                "chassis_height": "0.15",
                "chassis_mass": "5.0",
                "wheel_radius": "0.08",
                "wheel_separation": "0.38",
                "lidar_rate": "10.0"
            }
            bounds = {
                "chassis_length": (0.2, 1.5), "chassis_width": (0.15, 1.0),
                "chassis_height": (0.05, 0.6), "chassis_mass": (0.5, 80.0),
                "wheel_radius": (0.03, 0.4), "wheel_separation": (0.12, 1.2),
                "lidar_rate": (1.0, 40.0),
            }

        if parameters:
            unknown = set(parameters) - set(p)
            if unknown:
                raise ValueError(f"Unsupported template parameters: {', '.join(sorted(unknown))}")
            p.update({k: str(v) for k, v in parameters.items()})
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", p["robot_name"]):
            raise ValueError("robot_name must be a simple ROS/Gazebo identifier")

        for key, (minimum, maximum) in bounds.items():
            try:
                value = float(p[key])
            except ValueError as exc:
                raise ValueError(f"{key} must be numeric") from exc
            if not minimum <= value <= maximum:
                raise ValueError(f"{key} must be between {minimum} and {maximum}")
            p[key] = str(value)

        package_root = ws / "src" / "robosahaya_simulation"
        if package_root.exists():
            shutil.rmtree(package_root)
        shutil.copytree(source_dir, package_root)

        # Locate simulation world
        world_source = None
        for candidate in self.templates_dir.parent.glob(f"worlds/*/{world_id}.sdf"):
            if candidate.exists():
                world_source = candidate
                break
        if world_source is None:
            world_source = self.templates_dir.parent / "worlds" / "obstacle_course" / f"{world_id}.sdf"
        if not world_source.exists():
            raise FileNotFoundError(f"Simulation world '{world_id}.sdf' not found.")

        (package_root / "worlds").mkdir(exist_ok=True)
        shutil.copy2(world_source, package_root / "worlds" / f"{world_id}.sdf")
        # Ensure fallback compatibility for launch files
        if world_id != "obstacle_world":
            shutil.copy2(world_source, package_root / "worlds" / "obstacle_world.sdf")
        if world_id != "tabletop_world":
            shutil.copy2(world_source, package_root / "worlds" / "tabletop_world.sdf")
        xacro_file = package_root / "urdf" / "model.urdf.xacro"
        xacro_file.parent.mkdir(exist_ok=True)
        shutil.move(str(package_root / "model.urdf.xacro"), str(xacro_file))
        # Convert the small, explicit template placeholders before invoking Xacro.
        xacro_text = xacro_file.read_text(encoding="utf-8")
        for key, value in p.items():
            xacro_text = xacro_text.replace("{{ " + key + " }}", value)
        xacro_file.write_text(xacro_text, encoding="utf-8")
        (package_root / "config").mkdir(exist_ok=True)
        shutil.move(str(package_root / "bridge.yaml"), str(package_root / "config" / "bridge.yaml"))
        (package_root / "launch").mkdir(exist_ok=True)
        shutil.move(str(package_root / "robot.launch.py"), str(package_root / "launch" / "robot.launch.py"))
        (package_root / "resource").mkdir(exist_ok=True)
        (package_root / "resource" / "robosahaya_simulation").write_text("", encoding="utf-8")
        (package_root / "package.xml").write_text(PACKAGE_XML, encoding="utf-8")
        (package_root / "setup.py").write_text(PACKAGE_SETUP, encoding="utf-8")
        (package_root / "setup.cfg").write_text(
            "[develop]\nscript_dir=$base/lib/robosahaya_simulation\n[install]\ninstall_scripts=$base/lib/robosahaya_simulation\n",
            encoding="utf-8",
        )
        generated = subprocess.run(
            ["xacro", str(xacro_file)], cwd=ws, capture_output=True, text=True, timeout=20, check=False
        )
        if generated.returncode != 0:
            raise RuntimeError(f"Xacro expansion failed: {generated.stderr[-3000:]}")
        (package_root / "urdf" / "model.urdf").write_text(generated.stdout, encoding="utf-8")

        # Create manifest.json
        manifest = {
            "task_id": task_id,
            "template_id": template_id,
            "world_id": world_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parameters": p,
            "files": ["src/robosahaya_simulation/urdf/model.urdf", "src/robosahaya_simulation/config/bridge.yaml",
                      "src/robosahaya_simulation/launch/robot.launch.py", "src/robosahaya_simulation/worlds/obstacle_world.sdf"]
        }
        with open(ws / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Create backward-compatible convenience links at workspace root
        for src_rel, link_name in [
            (package_root / "urdf" / "model.urdf", ws / "model.urdf"),
            (package_root / "config" / "bridge.yaml", ws / "bridge.yaml"),
            (package_root / "launch" / "robot.launch.py", ws / "robot.launch.py"),
        ]:
            if link_name.is_symlink() or link_name.exists():
                link_name.unlink()
            try:
                link_name.symlink_to(src_rel.relative_to(ws))
            except Exception:
                shutil.copy2(src_rel, link_name)

        return ws

    def validate_workspace_files(self, workspace_dir: Path) -> Dict[str, Any]:
        """
        Validates XML and YAML syntax of generated workspace files.
        """
        results = {"valid": True, "checks": {}, "errors": []}

        package_root = workspace_dir / "src" / "robosahaya_simulation"
        # Validate the expanded URDF, source Xacro and the actual SDF world.
        urdf_file = package_root / "urdf" / "model.urdf"
        if urdf_file.exists():
            try:
                ET.parse(str(urdf_file))
                results["checks"]["urdf_xml"] = "VALID"
            except Exception as e:
                results["valid"] = False
                results["checks"]["urdf_xml"] = f"INVALID: {str(e)}"
                results["errors"].append(str(e))
        else:
            results["valid"] = False
            results["checks"]["urdf_xml"] = "MISSING"

        # 2. Validate bridge.yaml
        bridge_file = package_root / "config" / "bridge.yaml"
        if bridge_file.exists():
            try:
                with open(bridge_file, "r") as f:
                    bridge_data = yaml.safe_load(f)
                    if not isinstance(bridge_data, list) or not bridge_data:
                        raise ValueError("bridge configuration must be a non-empty list")
                results["checks"]["bridge_yaml"] = "VALID"
            except Exception as e:
                results["valid"] = False
                results["checks"]["bridge_yaml"] = f"INVALID: {str(e)}"
                results["errors"].append(str(e))
        else:
            results["valid"] = False
            results["checks"]["bridge_yaml"] = "MISSING"

        world_file = package_root / "worlds" / "obstacle_world.sdf"
        try:
            ET.parse(str(world_file))
            results["checks"]["world_sdf"] = "VALID"
        except Exception as exc:
            results["valid"] = False
            results["checks"]["world_sdf"] = f"INVALID: {exc}"
            results["errors"].append(str(exc))
        launch_file = package_root / "launch" / "robot.launch.py"
        if not launch_file.is_file():
            results["valid"] = False
            results["errors"].append("Simulation launch file is missing")
        else:
            try:
                import ast
                ast.parse(launch_file.read_text(encoding="utf-8"), filename=str(launch_file))
                results["checks"]["launch_file"] = "VALID_PYTHON_SYNTAX"
            except Exception as exc:
                results["valid"] = False
                results["checks"]["launch_file"] = f"INVALID: {exc}"
                results["errors"].append(str(exc))

        return results

    def build_workspace(self, workspace_dir: Path, timeout_sec: int = 180, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Build the generated ament workspace with a fixed colcon invocation."""
        proc = None
        try:
            proc = subprocess.Popen(
                ["colcon", "build", "--symlink-install"], cwd=workspace_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                start_new_session=True,
            )
            if task_id:
                with self._process_lock:
                    self._builds[task_id] = proc
            output, _ = proc.communicate(timeout=timeout_sec)
            return {
                "success": proc.returncode == 0,
                "return_code": proc.returncode,
                "log": output[-12000:],
                "command": ["colcon", "build", "--symlink-install"],
            }
        except subprocess.TimeoutExpired as exc:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    output, _ = proc.communicate(timeout=3)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    output, _ = proc.communicate()
                return {"success": False, "return_code": proc.returncode,
                        "log": output[-12000:], "timed_out": True}
            return {"success": False, "return_code": None, "log": str(exc)[-3000:], "timed_out": True}
        except FileNotFoundError as exc:
            return {"success": False, "return_code": None, "log": str(exc), "unavailable": True}
        finally:
            if task_id:
                with self._process_lock:
                    self._builds.pop(task_id, None)

    def cleanup_all_simulation_processes(self, exclude_task_id: Optional[str] = None) -> None:
        """Forcefully terminates lingering Gazebo, robot_state_publisher, and bridge processes."""
        with self._process_lock:
            tids = [tid for tid in list(self._processes) if tid != exclude_task_id]
        for tid in tids:
            self.stop_task_runtime(tid, timeout_sec=2.0)

        # Forcefully terminate any orphaned background processes
        for cmd in ("pkill -9 -f robot_state_publisher", "pkill -9 -f parameter_bridge", "pkill -9 -f 'gz sim'"):
            try:
                subprocess.run(cmd, shell=True, capture_output=True, timeout=2)
            except Exception:
                pass

    def launch_workspace(self, task_id: str, workspace_dir: Path, startup_timeout_sec: float = 8.0, headless: bool = False) -> Dict[str, Any]:
        """Launch only the registered simulation profile and retain its process handle."""
        # Stop any previously running simulation profiles before launching a new one
        self.cleanup_all_simulation_processes(exclude_task_id=task_id)

        with self._process_lock:
            old = self._processes.get(task_id)
            if old is not None and old.poll() is None:
                return {"success": True, "pid": old.pid, "already_running": True}
            log_path = workspace_dir / "launch.log"
            log_stream = log_path.open("ab")
            env = os.environ.copy()
            prefix = str(workspace_dir / "install" / "robosahaya_simulation")
            current = env.get("AMENT_PREFIX_PATH", "")
            env["AMENT_PREFIX_PATH"] = prefix + (os.pathsep + current if current else "")
            current_gz = env.get("GZ_SIM_RESOURCE_PATH", "")
            env["GZ_SIM_RESOURCE_PATH"] = f"{prefix}/share:{workspace_dir}/src/robosahaya_simulation" + (os.pathsep + current_gz if current_gz else "")
            try:
                proc = subprocess.Popen(
                    ["ros2", "launch", "robosahaya_simulation", "robot.launch.py", "use_sim_time:=true", f"headless:={'true' if headless else 'false'}"],
                    cwd=workspace_dir, env=env, stdin=subprocess.DEVNULL,
                    stdout=log_stream, stderr=subprocess.STDOUT, start_new_session=True,
                )
            except FileNotFoundError as exc:
                log_stream.close()
                return {"success": False, "error": str(exc), "unavailable": True}
            log_stream.close()
            self._processes[task_id] = proc
        try:
            code = proc.wait(timeout=max(0.1, startup_timeout_sec))
            return {"success": False, "return_code": code, "log": self._tail(log_path),
                    "error": "Simulation launch exited before the startup window completed."}
        except subprocess.TimeoutExpired:
            return {"success": True, "pid": proc.pid, "log_file": str(log_path)}

    def open_gazebo_gui(self, workspace_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Launch the Gazebo GUI client and attach it to the running simulation server."""
        if not shutil.which("gz"):
            return {"success": False, "error": "gz executable not found on PATH."}
        env = os.environ.copy()
        if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
            return {"success": False, "error": "No graphical DISPLAY or WAYLAND_DISPLAY environment variable detected."}
        if workspace_dir:
            prefix = str(Path(workspace_dir) / "install" / "robosahaya_simulation")
            current_gz = env.get("GZ_SIM_RESOURCE_PATH", "")
            env["GZ_SIM_RESOURCE_PATH"] = f"{prefix}/share:{Path(workspace_dir)}/src/robosahaya_simulation" + (os.pathsep + current_gz if current_gz else "")
        try:
            gui_proc = subprocess.Popen(
                ["gz", "sim", "-g"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            return {"success": True, "pid": gui_proc.pid}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @staticmethod
    def _tail(path: Path, limit: int = 8000) -> str:
        try:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - limit))
                return stream.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def stop_task_runtime(self, task_id: str, timeout_sec: float = 5.0) -> None:
        with self._process_lock:
            proc = self._processes.pop(task_id, None)
            build = self._builds.get(task_id)
        for running in (proc, build):
            if running is None or running.poll() is not None:
                continue
            try:
                os.killpg(running.pid, signal.SIGTERM)
                running.wait(timeout=timeout_sec)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(running.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def shutdown_all(self) -> None:
        """Stop every process group managed by this application before exit."""
        with self._process_lock:
            task_ids = set(self._processes) | set(self._builds)
        for task_id in task_ids:
            self.stop_task_runtime(task_id)

    @staticmethod
    def apply_unified_diff(original_text: str, unified_diff: str) -> str:
        """Apply a unified diff only when every context/removal line matches exactly."""
        source = original_text.splitlines(keepends=True)
        patch_lines = unified_diff.splitlines(keepends=True)
        result: list[str] = []
        source_cursor = 0
        i = 0
        saw_hunk = False
        changed = False
        hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        while i < len(patch_lines):
            header = hunk_re.match(patch_lines[i].rstrip("\r\n"))
            if not header:
                i += 1
                continue
            saw_hunk = True
            old_start = int(header.group(1))
            expected_old = int(header.group(2) or 1)
            expected_new = int(header.group(4) or 1)
            consumed_old = 0
            produced_new = 0
            hunk_start = max(0, old_start - 1)
            if hunk_start < source_cursor or hunk_start > len(source):
                raise ValueError("Patch hunk has invalid or overlapping source offsets")
            result.extend(source[source_cursor:hunk_start])
            source_cursor = hunk_start
            i += 1
            while i < len(patch_lines) and not patch_lines[i].startswith("@@ "):
                line = patch_lines[i]
                if line.startswith("\\ No newline"):
                    i += 1
                    continue
                marker, body = line[:1], line[1:]
                if marker == " ":
                    if source_cursor >= len(source) or source[source_cursor] != body:
                        raise ValueError("Patch context does not match the current file")
                    result.append(body)
                    source_cursor += 1
                    consumed_old += 1
                    produced_new += 1
                elif marker == "-":
                    if source_cursor >= len(source) or source[source_cursor] != body:
                        raise ValueError("Patch removal does not match the current file")
                    source_cursor += 1
                    consumed_old += 1
                    changed = True
                elif marker == "+":
                    result.append(body)
                    produced_new += 1
                    changed = True
                else:
                    break
                i += 1
            if consumed_old != expected_old or produced_new != expected_new:
                raise ValueError("Patch hunk line counts do not match its header")
        if not saw_hunk:
            raise ValueError("Unified diff contains no patch hunks")
        if not changed:
            raise ValueError("Unified diff contains no file changes")
        result.extend(source[source_cursor:])
        return "".join(result)

    def inject_demo_fault(self, workspace_dir: Path, fault_type: str = "missing_bridge_scan") -> PatchProposal:
        """
        Deterministically injects a known fault into the task workspace and returns the corresponding repair proposal.
        """
        target_file_rel = "src/robosahaya_simulation/config/bridge.yaml"
        bridge_file = workspace_dir / target_file_rel
        if not bridge_file.exists():
            raise FileNotFoundError("bridge.yaml missing in workspace.")

        with open(bridge_file, "r") as f:
            original_lines = f.readlines()

        if fault_type == "missing_bridge_scan":
            # Mutate /scan to /scan_disabled in bridge
            faulty_lines = [
                line.replace('ros_topic_name: "/scan"', 'ros_topic_name: "/scan_disabled"')
                for line in original_lines
            ]
            with open(bridge_file, "w") as f:
                f.writelines(faulty_lines)

            # Generate unified diff that will restore it
            diff = "".join(difflib.unified_diff(
                faulty_lines,
                original_lines,
                fromfile="a/" + target_file_rel,
                tofile="b/" + target_file_rel
            ))
            diff_hash = PolicyGate.compute_diff_sha256(diff)

            return PatchProposal(
                patch_id="patch_enable_laser_bridge",
                target_file=target_file_rel,
                unified_diff=diff,
                diff_sha256=diff_hash,
                base_file_sha256=hashlib.sha256("".join(faulty_lines).encode("utf-8")).hexdigest(),
                explanation="Restore ROS 2 /scan topic mapping in ros_gz_bridge configuration",
                requires_approval=True,
                verification_test_id="test_laser_scan_payload",
                backup_required=True
            )

        raise ValueError(f"Unknown fault type: {fault_type}")

    def apply_reviewed_patch(
        self,
        workspace_dir: Path,
        proposal: PatchProposal,
        approval_token: str,
        policy_gate: Optional[PolicyGate] = None,
    ) -> Dict[str, Any]:
        """
        Safely applies an approved patch with hash binding and backup creation.
        """
        scoped_policy = PolicyGate(workspace_dir)
        try:
            target_path = scoped_policy.resolve_workspace_path(proposal.target_file, must_exist=True)
        except Exception as exc:
            return {"success": False, "error": f"Unsafe patch target: {exc}"}
        if not target_path.is_file():
            return {"success": False, "error": f"Target file '{proposal.target_file}' not found."}

        try:
            current_text = target_path.read_text(encoding="utf-8")
            current_hash = hashlib.sha256(current_text.encode("utf-8")).hexdigest()
            if proposal.base_file_sha256 is not None and current_hash != proposal.base_file_sha256:
                return {"success": False, "error": "Target file changed after proposal creation; review a fresh patch."}
            actual_diff_hash = hashlib.sha256(proposal.unified_diff.encode("utf-8")).hexdigest()
            if actual_diff_hash != proposal.sha256:
                return {"success": False, "error": "Patch diff digest does not match its contents."}
            gate = policy_gate or self.policy_gate
            gate.consume_patch_approval(proposal, approval_token)
            patched_text = self.apply_unified_diff(current_text, proposal.unified_diff)
            backup_path = target_path.with_suffix(target_path.suffix + ".bak")
            shutil.copy2(target_path, backup_path)
            root_bak = workspace_dir / (Path(proposal.target_file).name + ".bak")
            if not root_bak.exists():
                try:
                    root_bak.symlink_to(backup_path.relative_to(workspace_dir))
                except Exception:
                    shutil.copy2(backup_path, root_bak)
            temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
            temp_path.write_text(patched_text, encoding="utf-8")
            os.replace(temp_path, target_path)

            return {
                "success": True,
                "backup_file": str(backup_path),
                "patched_file": str(target_path),
                "patch_id": proposal.patch_id
            }
        except Exception as ex:
            return {"success": False, "error": f"Patch application failed: {str(ex)}"}
