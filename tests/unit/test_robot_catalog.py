import sys
import types
from pathlib import Path

import pytest

from robosahaya.robot_catalog import RobotModelCatalog
from robosahaya.orchestrator import TaskOrchestrator
from robosahaya.task_store import TaskStore
from robosahaya.workspace_manager import WorkspaceManager


def test_catalog_contains_multiple_robot_families(tmp_path):
    catalog = RobotModelCatalog(cache_root=tmp_path / "cache")
    models = catalog.list_models(include_discovered=False)
    families = {model.family for model in models}
    ids = {model.id for model in models}
    assert {"robotic arm", "quadruped", "humanoid/biped", "drone/aerial", "mobile manipulator", "wheeled/mobile"}.issubset(families)
    assert {"franka_panda", "so_arm101", "ur5e", "unitree_go2", "crazyflie_2", "fetch"}.issubset(ids)


def test_runtime_ready_is_distinct_from_description_only(tmp_path):
    catalog = RobotModelCatalog(cache_root=tmp_path / "cache")
    assert catalog.get_model("franka panda").runtime_ready is True
    assert catalog.get_model("SO-101").runtime_ready is True
    assert catalog.get_model("SO-100").runtime_ready is False
    assert "not runtime verified" in catalog.availability(catalog.get_model("SO-100"))


def test_prepare_unknown_model_does_not_import_arbitrary_module(tmp_path):
    catalog = RobotModelCatalog(cache_root=tmp_path / "cache")
    result = catalog.prepare_description("../../do-not-import")
    assert result["success"] is False
    assert "Unknown robot model" in result["error"]


def test_prepare_description_inspects_an_allowlisted_cached_module(tmp_path, monkeypatch):
    urdf = tmp_path / "example.urdf"
    urdf.write_text(
        '<robot name="test"><link name="base"><visual><geometry><mesh filename="package://demo/meshes/base.stl"/></geometry></visual></link>'
        '<link name="arm"/><joint name="joint1" type="fixed"><parent link="base"/><child link="arm"/></joint></robot>',
        encoding="utf-8",
    )
    fake_package = types.ModuleType("robot_descriptions")
    fake_package.__path__ = []
    fake_module = types.ModuleType("robot_descriptions.so_arm100_description")
    fake_module.URDF_PATH = str(urdf)
    fake_module.PACKAGE_PATH = str(tmp_path)
    fake_module.REPOSITORY_PATH = str(tmp_path)

    catalog = RobotModelCatalog(cache_root=tmp_path / "cache")
    monkeypatch.setattr(catalog, "_installed_module_names", lambda: {"so_arm100_description"})
    import importlib
    original_import_module = importlib.import_module

    def controlled_import_module(name, package=None):
        if name == "robot_descriptions.so_arm100_description":
            return fake_module
        if name == "robot_descriptions":
            return fake_package
        return original_import_module(name, package)

    monkeypatch.setattr(RobotModelCatalog, "descriptions_package_installed", property(lambda self: True))
    monkeypatch.setattr("robosahaya.robot_catalog.importlib.import_module", controlled_import_module)
    result = catalog.prepare_description("so_arm100")
    assert result["success"] is True
    assert result["format"] == "URDF"
    assert result["stats"]["links"] == 2
    assert result["stats"]["joints"] == 1
    assert result["stats"]["mesh_references"] == 1
    assert result["gazebo_runtime_verified"] is False
    assert (tmp_path / "cache" / "so_arm100" / "metadata.json").exists()


def test_named_external_model_is_not_silently_substituted(tmp_path):
    root = Path(__file__).resolve().parent.parent.parent
    orchestrator = TaskOrchestrator(
        task_store=TaskStore(tmp_path / "tasks"),
        workspace_manager=WorkspaceManager(
            base_workspace_root=tmp_path / "workspaces",
            templates_dir=root / "simulation" / "templates",
        ),
        mode="mock",
    )
    infer = orchestrator.infer_robot_and_world("Build an SO-100 arm with gripper and camera.")
    assert infer["auto_robot"] == "so_arm100"
    assert infer["unsupported_capabilities"]
    plan = orchestrator.formulate_plan("task_so100_01", "Build an SO-100 arm with gripper and camera.")
    assert plan.is_supported is False
    assert plan.robot_template == "so_arm100"
    assert "will not substitute another robot" in plan.rejection_reason


def test_named_so101_selects_verified_template(tmp_path):
    root = Path(__file__).resolve().parent.parent.parent
    orchestrator = TaskOrchestrator(
        task_store=TaskStore(tmp_path / "tasks"),
        workspace_manager=WorkspaceManager(
            base_workspace_root=tmp_path / "workspaces",
            templates_dir=root / "simulation" / "templates",
        ),
        mode="mock",
    )
    plan = orchestrator.formulate_plan("task_so101_01", "Build an SO-101 arm with gripper and camera.")
    assert plan.is_supported is True
    assert plan.robot_template == "so_arm101"
    assert plan.world_template == "tabletop_world"


def test_named_bundled_panda_selects_existing_template(tmp_path):
    root = Path(__file__).resolve().parent.parent.parent
    orchestrator = TaskOrchestrator(
        task_store=TaskStore(tmp_path / "tasks"),
        workspace_manager=WorkspaceManager(
            base_workspace_root=tmp_path / "workspaces",
            templates_dir=root / "simulation" / "templates",
        ),
        mode="mock",
    )
    plan = orchestrator.formulate_plan("task_panda_01", "Build a Franka Emika Panda for pick and place.")
    assert plan.is_supported is True
    assert plan.robot_template == "franka_panda"
    assert plan.world_template == "tabletop_world"


def test_description_only_model_requires_optional_dependency_when_missing(tmp_path, monkeypatch):
    catalog = RobotModelCatalog(cache_root=tmp_path / "cache")
    monkeypatch.setattr(RobotModelCatalog, "descriptions_package_installed", property(lambda self: False))
    result = catalog.prepare_description("so_arm100")
    assert result["success"] is False
    assert "pip install -e '.[robot-catalogue]'" in result["error"]
    assert not (tmp_path / "cache" / "so_arm100" / "metadata.json").exists()


def test_robot_world_pair_is_validated(tmp_path):
    root = Path(__file__).resolve().parent.parent.parent
    orchestrator = TaskOrchestrator(
        task_store=TaskStore(tmp_path / "tasks"),
        workspace_manager=WorkspaceManager(
            base_workspace_root=tmp_path / "workspaces",
            templates_dir=root / "simulation" / "templates",
        ),
        mode="mock",
    )
    plan = orchestrator.formulate_plan(
        "task_bad_pair_01",
        "Create a differential drive robot.",
        manual_overrides={"robot_template": "differential_drive", "world_template": "tabletop_world"},
    )
    assert plan.is_supported is False
    assert "Invalid robot/world combination" in plan.rejection_reason


def test_named_robot_is_resolved_before_generic_arm_keywords(tmp_path):
    root = Path(__file__).resolve().parent.parent.parent
    orchestrator = TaskOrchestrator(
        task_store=TaskStore(tmp_path / "tasks"),
        workspace_manager=WorkspaceManager(
            base_workspace_root=tmp_path / "workspaces",
            templates_dir=root / "simulation" / "templates",
        ),
        mode="mock",
    )
    infer = orchestrator.infer_robot_and_world("Build a UR5e arm with a camera.")
    assert infer["auto_robot"] == "ur5e"
    assert "unverified_robot_model:ur5e" in infer["unsupported_capabilities"]
    assert "no verified RoboSahaya Gazebo runtime adapter" in infer["robot_reason"]
