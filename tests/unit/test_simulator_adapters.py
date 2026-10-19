"""Unit tests for multi-simulator adapters and adapter registry."""
from pathlib import Path
import pytest

from robosahaya.robot_registry import get_robot_registry
from robosahaya.simulators.gazebo_adapter import GazeboAdapter
from robosahaya.simulators.mujoco_adapter import MujocoAdapter
from robosahaya.simulators.registry import SimulatorAdapterRegistry


@pytest.fixture
def registry():
    return get_robot_registry()


@pytest.fixture
def sim_registry():
    return SimulatorAdapterRegistry()


def test_gazebo_adapter_probe():
    adapter = GazeboAdapter()
    env = adapter.probe_environment()
    assert env["simulator"] == "gazebo_harmonic"
    assert "gz_binary" in env


def test_mujoco_adapter_probe():
    adapter = MujocoAdapter()
    env = adapter.probe_environment()
    assert env["simulator"] == "mujoco"
    assert env["available"] is True
    assert env["version"] is not None


def test_adapter_registry_selection(registry, sim_registry):
    # Differential Drive -> Gazebo
    diff_model = registry.get_model("differential_drive")
    adapter, details = sim_registry.select_adapter(diff_model)
    assert adapter is not None
    assert adapter.name == "gazebo_harmonic"

    # Franka Panda -> Gazebo
    panda_model = registry.get_model("franka_panda")
    adapter, details = sim_registry.select_adapter(panda_model)
    assert adapter is not None
    assert adapter.name == "gazebo_harmonic"

    # SO-101 -> Gazebo
    so101_model = registry.get_model("so_arm101")
    adapter, details = sim_registry.select_adapter(so101_model)
    assert adapter is not None
    assert adapter.name == "gazebo_harmonic"

    # RSK -> MuJoCo
    rsk_model = registry.get_model("rsk")
    adapter, details = sim_registry.select_adapter(rsk_model)
    assert adapter is not None
    assert adapter.name == "mujoco"


def test_mujoco_adapter_rsk_validation(registry):
    rsk_model = registry.get_model("rsk")
    adapter = MujocoAdapter()

    # Model path in cache
    cache_path = Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie" / "robot_soccer_kit" / "robot_soccer_kit.xml"
    assert cache_path.exists(), f"RSK XML missing at {cache_path}"

    val = adapter.validate_model(cache_path)
    assert val["valid"] is True
    stats = val["model_stats"]
    assert stats["nbody"] > 0
    assert stats["nu"] == 4  # 4 omni-wheel actuators


def test_mujoco_adapter_rsk_verification(registry):
    adapter = MujocoAdapter()
    cache_path = Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie" / "robot_soccer_kit" / "robot_soccer_kit.xml"

    context = {"model_path": str(cache_path)}
    res = adapter.run_verification(context)
    assert res["success"] is True
    milestones = res["milestones"]
    assert milestones["milestone_a"]["passed"] is True
    assert milestones["milestone_b"]["passed"] is True
    assert milestones["milestone_c"]["passed"] is True
    assert milestones["milestone_d"]["passed"] is True
    assert milestones["milestone_e"]["passed"] is True
