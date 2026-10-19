"""Robot model catalogue with opt-in, on-demand description caching.

The catalogue separates a model description being available from a model being
verified for RoboSahaya's Gazebo execution pipeline. Listing/searching never
imports a third-party robot module (which could download files); downloads only
happen after an explicit call to ``prepare_description`` from the UI.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import pkgutil
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

ROBOT_DESCRIPTIONS_URL = "https://github.com/robot-descriptions/robot_descriptions.py"


@dataclass(frozen=True)
class RobotModel:
    id: str
    display_name: str
    family: str
    aliases: tuple[str, ...] = ()
    description_module: Optional[str] = None
    formats: tuple[str, ...] = ("URDF",)
    license: str = "Check upstream source"
    source_url: str = ROBOT_DESCRIPTIONS_URL
    runtime_template_id: Optional[str] = None
    runtime_status: str = "description_only"
    notes: str = "Description availability does not imply a verified Gazebo controller or task pipeline."

    @property
    def runtime_ready(self) -> bool:
        return self.runtime_template_id is not None and self.runtime_status == "runtime_ready"


# Curated fast-start list. The installed robot_descriptions package can expose
# its wider catalogue (200+ entries) without importing/downloading each module.
_SEEDED_MODELS: tuple[RobotModel, ...] = (
    RobotModel(
        id="differential_drive", display_name="RoboSahaya Differential Drive",
        family="wheeled/mobile", aliases=("differential drive", "diff drive", "wheeled base"),
        formats=("URDF/Xacro",), license="RoboSahaya project licence",
        source_url="", runtime_template_id="differential_drive", runtime_status="runtime_ready",
        notes="Bundled template is registered for the current pipeline; perform live ROS 2/Gazebo checks on the target host before relying on it.",
    ),
    RobotModel(
        id="franka_panda", display_name="Franka Emika Panda",
        family="robotic arm", aliases=("franka panda", "franka emika panda", "panda arm", "panda robot"),
        description_module="panda_description", formats=("URDF", "SRDF"), license="Apache-2.0",
        runtime_template_id="robotic_arm_6dof", runtime_status="runtime_ready",
        notes="The bundled arm template contains Panda link meshes and joints. The execution template keeps its legacy ID robotic_arm_6dof; re-run live checks on the target host.",
    ),
    RobotModel(
        id="so_arm101", display_name="SO-ARM101 / SO-101",
        family="robotic arm", aliases=("so-101", "so101", "so arm 101", "so-arm101", "lerobot so101", "so 101"),
        description_module="so_arm101_description", formats=("URDF",), license="Apache-2.0",
        runtime_template_id="so_arm101", runtime_status="runtime_ready",
        notes="Bundled LeRobot / The Robot Studio 3D-printable 6-DOF follower arm template with Feetech STS3215 servos.",
    ),
    RobotModel(
        id="rsk", display_name="Robot Soccer Kit (RSK)",
        family="wheeled/mobile", aliases=("rsk", "robot soccer kit", "robot_soccer_kit", "soccer robot"),
        description_module="rsk_mj_description", formats=("MJCF", "XML", "STL"), license="Apache-2.0",
        source_url="https://github.com/google-deepmind/mujoco_menagerie/tree/main/robot_soccer_kit",
        runtime_template_id="rsk", runtime_status="runtime_ready",
        notes="MuJoCo Menagerie Robot Soccer Kit omnidirectional mobile robot verified for MuJoCo physics runtime.",
    ),
    RobotModel("so_arm100", "SO-ARM100 / SO-100", "robotic arm", ("so-100", "so100", "so arm 100", "so-arm100"), "so_arm100_description", ("URDF",), "Apache-2.0"),
    RobotModel("ur3e", "Universal Robots UR3e", "robotic arm", ("ur3e", "ur 3e"), "ur3e_description", ("URDF",), "BSD-3-Clause"),
    RobotModel("ur5e", "Universal Robots UR5e", "robotic arm", ("ur5e", "ur 5e"), "ur5e_description", ("URDF", "MJCF"), "BSD-3-Clause"),
    RobotModel("ur10e", "Universal Robots UR10e", "robotic arm", ("ur10e", "ur 10e"), "ur10e_description", ("URDF", "MJCF"), "BSD-3-Clause"),
    RobotModel("kuka_iiwa14", "KUKA iiwa 14", "robotic arm", ("kuka iiwa", "iiwa14", "iiwa 14"), "iiwa14_description", ("URDF",), "BSD-3-Clause"),
    RobotModel("kinova_gen3_lite", "Kinova Gen3 Lite", "robotic arm", ("gen3 lite", "kinova gen3 lite"), "gen3_lite_description", ("URDF", "SRDF"), "BSD-3-Clause"),
    RobotModel("xarm6", "UFACTORY xArm6", "robotic arm", ("x arm 6", "xarm 6", "ufactory xarm6"), "xarm6_description", ("URDF", "SRDF"), "BSD-3-Clause"),
    RobotModel("open_manipulator_x", "ROBOTIS OpenMANIPULATOR-X", "robotic arm", ("open manipulator x", "openmanipulator-x"), "open_manipulator_x_description", ("URDF", "SRDF"), "Apache-2.0"),
    RobotModel("robotiq_2f85", "Robotiq 2F-85 Gripper", "gripper/end effector", ("robotiq", "2f85", "2f-85 gripper"), "robotiq_2f85_description", ("URDF",), "BSD"),
    RobotModel("unitree_go2", "Unitree Go2", "quadruped", ("unitree go2", "go2", "go 2"), "go2_description", ("URDF", "MJCF"), "BSD-3-Clause"),
    RobotModel("unitree_go1", "Unitree Go1", "quadruped", ("unitree go1", "go1", "go 1"), "go1_description", ("URDF", "MJCF"), "BSD-3-Clause"),
    RobotModel("anymal_b", "ANYmal B", "quadruped", ("anymal b", "anymal_b"), "anymal_b_description", ("URDF", "MJCF"), "BSD-3-Clause"),
    RobotModel("unitree_g1", "Unitree G1 Humanoid", "humanoid/biped", ("unitree g1", "g1 humanoid", "g1 robot"), "g1_description", ("URDF", "MJCF"), "BSD-3-Clause"),
    RobotModel("unitree_h1", "Unitree H1 Humanoid", "humanoid/biped", ("unitree h1", "h1 humanoid", "h1 robot"), "h1_description", ("URDF", "MJCF"), "BSD-3-Clause"),
    RobotModel("simple_humanoid", "Simple Humanoid", "humanoid/biped", ("humanoid robot", "simple humanoid"), "simple_humanoid_description", ("URDF", "SRDF"), "BSD-2-Clause"),
    RobotModel("atlas_v4", "Boston Dynamics Atlas v4", "humanoid/biped", ("atlas v4", "atlas humanoid"), "atlas_v4_description", ("URDF",), "MIT"),
    RobotModel("crazyflie_2", "Crazyflie 2.0", "drone/aerial", ("crazyflie", "crazyflie 2", "cf2 drone"), "cf2_description", ("URDF", "MJCF"), "MIT"),
    RobotModel("skydio_x2", "Skydio X2", "drone/aerial", ("skydio x2", "skydio drone"), "skydio_x2_description", ("URDF", "MJCF"), "Apache-2.0"),
    RobotModel("fetch", "Fetch Mobile Manipulator", "mobile manipulator", ("fetch robot", "fetch mobile manipulator"), "fetch_description", ("URDF",), "MIT"),
    RobotModel("stretch_4", "Hello Robot Stretch 4", "mobile manipulator", ("stretch 4", "stretch robot"), "stretch_4_description", ("URDF",), "BSD-3-Clause-Clear"),
    RobotModel("pr2", "PR2 Mobile Manipulator", "mobile manipulator", ("pr2 robot", "personal robot 2"), "pr2_description", ("URDF",), "BSD"),
    RobotModel("tiago", "PAL Robotics TIAGo", "mobile manipulator", ("tiago robot", "tiago"), "tiago_official_description", ("URDF",), "Apache-2.0"),
    RobotModel("unitree_z1", "Unitree Z1 Arm", "robotic arm", ("unitree z1", "z1 arm"), "z1_description", ("URDF", "MJCF"), "BSD-3-Clause"),
    RobotModel("piper", "AgileX PiPER Arm", "robotic arm", ("piper arm", "agilex piper"), "piper_description", ("URDF",), "MIT"),
    RobotModel("upkie", "Upkie Wheeled Biped", "wheeled/mobile", ("upkie robot",), "upkie_description", ("URDF",), "Apache-2.0"),
    RobotModel("rsk_omnidirectional", "Robot Soccer Kit Omnidirectional", "wheeled/mobile", ("rsk omnidirectional", "omni drive robot"), "rsk_description", ("URDF", "MJCF"), "MIT"),
)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _safe_model_id(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", value or "").strip("_")
    if not clean or len(clean) > 100:
        raise ValueError("Invalid robot model identifier.")
    return clean


class RobotModelCatalog:
    """Curated aliases plus optional discovery through robot_descriptions.py."""

    def __init__(self, cache_root: Optional[Path] = None):
        self.cache_root = Path(cache_root or Path.home() / ".cache" / "robosahaya" / "robot_models").expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._seeded = {m.id: m for m in _SEEDED_MODELS}
        from .model_cache import ModelCacheManager
        self.model_cache = ModelCacheManager(cache_root=self.cache_root)

    @property
    def descriptions_package_installed(self) -> bool:
        try:
            return importlib.util.find_spec("robot_descriptions") is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False

    def _installed_module_names(self) -> set[str]:
        if not self.descriptions_package_installed:
            return set()
        try:
            pkg = importlib.import_module("robot_descriptions")
            paths = getattr(pkg, "__path__", [])
            return {
                item.name for item in pkgutil.iter_modules(paths)
                if item.name.endswith(("_description", "_mj_description"))
                and not item.name.startswith("loaders")
            }
        except Exception:
            return set()

    def list_models(self, search: str = "", family: str = "All", include_discovered: bool = True) -> list[RobotModel]:
        models: dict[str, RobotModel] = dict(self._seeded)
        installed_names = self._installed_module_names() if include_discovered else set()
        for model in list(models.values()):
            if model.description_module and model.description_module in installed_names and not model.runtime_ready:
                models[model.id] = RobotModel(**{**asdict(model), "notes": model.notes})
        known_modules = {m.description_module for m in models.values() if m.description_module}
        for module in sorted(installed_names - known_modules):
            # Listing metadata must not import this model module or trigger its download.
            if module.endswith("_mj_description"):
                base_name = module.removesuffix("_mj_description")
                formats = ("MJCF",)
            else:
                base_name = module.removesuffix("_description")
                formats = ("URDF/Xacro",)
            model_id = _safe_model_id(base_name)
            family_guess = self._guess_family(module)
            display = re.sub(r"[_]+", " ", base_name).title()
            models.setdefault(model_id, RobotModel(
                id=model_id, display_name=display, family=family_guess,
                aliases=(base_name.replace("_", " "),), description_module=module,
                formats=formats, license="Check upstream robot description for this model",
                runtime_status="description_only",
                notes="Discovered in robot_descriptions; this does not certify Gazebo, sensors, controllers, or task behavior.",
            ))

        q = _normalize(search)
        result = []
        for model in models.values():
            if family and family != "All" and model.family != family:
                continue
            searchable = _normalize(" ".join((model.id, model.display_name, model.family, *model.aliases, model.description_module or "")))
            if q and q not in searchable:
                continue
            result.append(model)
        return sorted(result, key=lambda m: (not m.runtime_ready, m.family, m.display_name.casefold()))

    @staticmethod
    def _guess_family(module: str) -> str:
        text = module.lower()
        if any(k in text for k in ("arm", "panda", "ur5", "ur3", "ur10", "iiwa", "xarm", "manipulator", "gen3", "z1", "piper")):
            return "robotic arm"
        if any(k in text for k in ("go1", "go2", "a1_description", "aliengo", "anymal", "spot", "laikago", "quadruped", "minitaur")):
            return "quadruped"
        if any(k in text for k in ("drone", "cf2", "skydio", "uav", "quadrotor")):
            return "drone/aerial"
        if any(k in text for k in ("humanoid", "atlas", "g1_description", "h1_description", "cassie", "biped", "apollo")):
            return "humanoid/biped"
        if any(k in text for k in ("fetch", "stretch", "tiago", "pr2", "mobile_manipulator", "hsrb", "pepper")):
            return "mobile manipulator"
        if any(k in text for k in ("hand", "gripper", "effector")):
            return "gripper/end effector"
        if any(k in text for k in ("wheeled", "upkie", "omni", "rsk", "turtlebot")):
            return "wheeled/mobile"
        return "other"

    def get_model(self, model_id_or_alias: str) -> Optional[RobotModel]:
        needle = _normalize(model_id_or_alias)
        if not needle:
            return None
        models = self.list_models(include_discovered=True)
        for model in models:
            if needle == _normalize(model.id) or needle == _normalize(model.display_name):
                return model
        # Prefer the longest matching alias to avoid e.g. matching a generic word first.
        matches = []
        for model in models:
            for alias in model.aliases:
                alias_n = _normalize(alias)
                if alias_n and (needle == alias_n or alias_n in needle):
                    matches.append((len(alias_n), model))
        if matches:
            return max(matches, key=lambda pair: pair[0])[1]
        return None

    def resolve_prompt(self, prompt: str) -> Optional[RobotModel]:
        """Resolve a named robot only; generic words like 'arm' aren't model IDs."""
        normalized = f" {_normalize(prompt)} "
        matches: list[tuple[int, RobotModel]] = []
        generic_aliases = {"differential drive", "diff drive", "wheeled base", "mobile robot", "robotic arm", "robot arm", "humanoid robot", "drone", "quadruped", "panda", "go2", "go1", "pr2", "fetch", "tiago"}
        for model in self.list_models(include_discovered=True):
            for phrase in (model.display_name, model.id, *model.aliases):
                pn = _normalize(phrase)
                # Do not let broad family words silently choose one arbitrary model.
                if pn in generic_aliases:
                    continue
                if len(pn.split()) == 1 and pn in {"robot", "arm", "drone", "humanoid", "panda", "fetch", "tiago", "go2", "go1", "pr2"}:
                    continue
                if pn and f" {pn} " in normalized:
                    matches.append((len(pn), model))
        if not matches:
            return None
        return max(matches, key=lambda pair: pair[0])[1]

    def availability(self, model: RobotModel) -> str:
        if model.runtime_ready:
            if model.id == "rsk":
                return "MuJoCo simulation ready"
            return "Gazebo template included"
        if not model.description_module:
            return "Description module not registered"
        if not self.descriptions_package_installed:
            return "On-demand description available; install optional robot catalogue support (not runtime verified)"
        if model.description_module not in self._installed_module_names():
            return "Description module not present in installed catalogue version"
        manifest = self._manifest_path(model.id)
        if manifest.exists():
            return "Description cached locally; Gazebo adapter not runtime verified"
        return "Available for on-demand description cache; Gazebo adapter not runtime verified"

    def _manifest_path(self, model_id: str) -> Path:
        return self.cache_root / _safe_model_id(model_id) / "metadata.json"

    def prepare_description(self, model_id: str) -> dict[str, Any]:
        """Explicitly import/cache one allowlisted description and inspect its files.

        robot_descriptions imports may download remote repositories. This method is
        only called by the user's explicit catalogue action, never during planning.
        """
        model = self.get_model(model_id)
        if model is None:
            return {"success": False, "error": f"Unknown robot model '{model_id}'. Choose a model from the catalogue."}
        if model.runtime_ready:
            cache_note = ""
            try:
                c_res = self.model_cache.prepare_model(model.id)
                if c_res.get("success"):
                    num = c_res.get("total_files", 0)
                    cache_note = f" Cache verified ({num} assets cryptographically checked)."
            except Exception as e:
                logger.debug("Cache prep for %s: %s", model.id, e)
            sim_name = "MuJoCo" if model.id == "rsk" else "Gazebo"
            return {
                "success": True, "model_id": model.id, "display_name": model.display_name,
                "runtime_ready": True, "message": f"This model has a verified {sim_name} execution pipeline.{cache_note}",
                "runtime_template_id": model.runtime_template_id,
            }

        if not model.description_module:
            return {"success": False, "error": "No safe description module is registered for this catalogue entry."}
        if not self.descriptions_package_installed:
            return {
                "success": False,
                "error": "Optional robot catalogue package is not installed. On Ubuntu, run: pip install -e '.[robot-catalogue]'. Then restart RoboSahaya. No download was attempted.",
            }
        if model.description_module not in self._installed_module_names():
            return {"success": False, "error": f"Installed robot_descriptions package does not contain '{model.description_module}'. Upgrade the optional package or choose another listed model."}

        try:
            module = importlib.import_module(f"robot_descriptions.{model.description_module}")
        except Exception as exc:
            return {
                "success": False,
                "error": f"Could not fetch/cache {model.display_name}: {type(exc).__name__}: {str(exc)[:500]}",
                "note": "The source package may be unavailable, its download may have failed, or this module may need additional dependencies.",
            }

        urdf_path = getattr(module, "URDF_PATH", None)
        xacro_path = getattr(module, "XACRO_PATH", None)
        mjcf_path = getattr(module, "MJCF_PATH", None)
        selected_path = urdf_path or xacro_path or mjcf_path
        if not selected_path:
            return {"success": False, "error": "The downloaded module did not expose URDF_PATH, XACRO_PATH, or MJCF_PATH."}
        path = Path(str(selected_path)).expanduser().resolve()
        if not path.is_file():
            return {"success": False, "error": f"The description path reported by the package does not exist: {path}"}

        fmt = "MJCF" if mjcf_path and str(selected_path) == str(mjcf_path) else ("Xacro" if xacro_path and str(selected_path) == str(xacro_path) else "URDF")
        stats: dict[str, Any] = {"links": None, "joints": None, "mesh_references": 0, "root_element": None}
        try:
            root = ET.parse(path).getroot()
            stats["root_element"] = root.tag
            stats["mesh_references"] = len(root.findall(".//mesh"))
            if root.tag == "robot":
                stats["links"] = len(root.findall("link"))
                stats["joints"] = len(root.findall("joint"))
            elif root.tag == "mujoco":
                stats["links"] = len(root.findall(".//body"))
                stats["joints"] = len(root.findall(".//joint"))
        except (ET.ParseError, OSError) as exc:
            return {"success": False, "error": f"Description file could not be parsed as XML: {exc}"}

        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "model_id": model.id,
            "display_name": model.display_name,
            "description_module": model.description_module,
            "format": fmt,
            "path": str(path),
            "package_path": str(getattr(module, "PACKAGE_PATH", "") or ""),
            "repository_path": str(getattr(module, "REPOSITORY_PATH", "") or ""),
            "license": model.license,
            "sha256": checksum,
            "inspected_at": datetime.now(timezone.utc).isoformat(),
            "stats": stats,
            "gazebo_runtime_verified": False,
            "controller_verified": False,
        }
        model_cache = self.cache_root / _safe_model_id(model.id)
        model_cache.mkdir(parents=True, exist_ok=True)
        manifest_path = model_cache / "metadata.json"
        # Atomic write avoids partial manifests if the UI process stops.
        fd, tmp_path = tempfile.mkstemp(prefix="metadata_", suffix=".tmp", dir=str(model_cache))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2)
            os.replace(tmp_path, manifest_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return {
            "success": True, **manifest,
            "message": "Description downloaded/cached and structurally inspected. This does not mean it is Gazebo-ready or controller-ready.",
        }

    def format_table_rows(self, search: str = "", family: str = "All") -> list[list[str]]:
        rows = []
        for model in self.list_models(search=search, family=family):
            rows.append([
                model.id,
                model.display_name,
                model.family,
                ", ".join(model.formats),
                model.license,
                "Ready" if model.runtime_ready else "Description only",
                self.availability(model),
            ])
        return rows

    def choices(self) -> list[tuple[str, str]]:
        choices = []
        for model in self.list_models():
            if model.runtime_ready:
                if model.id == "rsk":
                    status = "MuJoCo simulation ready"
                else:
                    status = "bundled pipeline template"
            else:
                status = "description only; not runtime verified"
            choices.append((f"{model.display_name} — {status}", model.id))
        return choices

    def get_readiness_dimensions(self, model_id: str) -> Any:
        from .readiness import (
            ReadinessDimensions,
            CatalogueStatus,
            CloudAvailability,
            LocalCacheStatus,
            DescriptionValidity,
            SimulatorCompatibility,
            SimulationReadiness,
            ControllerReadiness,
            SensorReadiness,
            TaskVerification,
        )
        from .cloud.client import get_cloud_status

        model = self.get_model(model_id)
        if not model:
            return ReadinessDimensions(catalogue_status=CatalogueStatus.UNKNOWN)

        cat_status = CatalogueStatus.REGISTERED if model.runtime_ready else CatalogueStatus.CATALOGUED
        c_status = get_cloud_status()
        cloud_avail = CloudAvailability.CLOUD_AVAILABLE if c_status.get("connected") else CloudAvailability.LOCAL_ONLY

        is_cached = self.model_cache.is_cache_verified(model.id) or model.runtime_ready
        cache_status = LocalCacheStatus.CACHE_VERIFIED if is_cached else LocalCacheStatus.NOT_CACHED
        desc_valid = DescriptionValidity.MODEL_VALIDATED if is_cached else DescriptionValidity.UNCHECKED

        if model.id in ("differential_drive", "franka_panda", "so_arm101"):
            sim_compat = SimulatorCompatibility.COMPATIBLE
            sim_ready = SimulationReadiness.SIMULATION_READY
            ctrl_ready = ControllerReadiness.CONTROL_READY
            sensor_ready = SensorReadiness.SENSOR_READY
            task_ver = TaskVerification.TASK_VERIFIED
        elif model.id == "rsk":
            sim_compat = SimulatorCompatibility.COMPATIBLE
            sim_ready = SimulationReadiness.SIMULATION_READY
            ctrl_ready = ControllerReadiness.CONTROL_READY
            sensor_ready = SensorReadiness.SENSOR_READY
            task_ver = TaskVerification.TASK_VERIFIED
        elif model.family in ("drone/aerial",):
            sim_compat = SimulatorCompatibility.SIMULATOR_UNAVAILABLE
            sim_ready = SimulationReadiness.UNTESTED
            ctrl_ready = ControllerReadiness.UNTESTED
            sensor_ready = SensorReadiness.UNTESTED
            task_ver = TaskVerification.UNTESTED
        else:
            sim_compat = SimulatorCompatibility.UNKNOWN
            sim_ready = SimulationReadiness.UNTESTED
            ctrl_ready = ControllerReadiness.UNTESTED
            sensor_ready = SensorReadiness.UNTESTED
            task_ver = TaskVerification.UNTESTED

        return ReadinessDimensions(
            catalogue_status=cat_status,
            cloud_availability=cloud_avail,
            local_cache_status=cache_status,
            description_validity=desc_valid,
            simulator_compatibility=sim_compat,
            simulation_readiness=sim_ready,
            controller_readiness=ctrl_ready,
            sensor_readiness=sensor_ready,
            task_verification=task_ver,
        )

    def evaluate_readiness(self, model_id: str, simulator: Optional[str] = None) -> Any:
        from .readiness import (
            RobotExecutionProfile,
            ReadinessDimensions,
            CatalogueStatus,
            evaluate_execution_readiness,
        )
        model = self.get_model(model_id)
        if not model:
            dims = ReadinessDimensions(catalogue_status=CatalogueStatus.UNKNOWN)
            prof = RobotExecutionProfile(robot_id=model_id)
            return evaluate_execution_readiness(prof, dims)

        dims = self.get_readiness_dimensions(model.id)
        sim = simulator or ("mujoco" if model.id == "rsk" else "gazebo_harmonic")
        prof = RobotExecutionProfile(
            robot_id=model.id,
            simulator=sim,
            required_controllers=("diff_drive_controller",) if model.id == "differential_drive" else (("actuators",) if model.id == "rsk" else ()),
            required_sensors=("lidar",) if model.id == "differential_drive" else (),
            required_tests=("test_rsk_physics_stepping",) if model.id == "rsk" else (),
        )
        return evaluate_execution_readiness(prof, dims)

    def prepare_model_package(self, model_id: str, version: str = "v1.0.0") -> dict[str, Any]:
        return self.model_cache.prepare_model(model_id, version)
