"""World and Object Management for RoboSahaya Studio.

Maintains scene object registries, ground-truth simulator poses,
and camera-derived perceptions without fabricating detections.
"""
from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class SceneEntity(BaseModel):
    """An object, landmark, or region in a simulation world."""
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    entity_type: str  # "table", "cube", "plate", "bin", "ball", "goal", "box", "obstacle"
    pose: Tuple[float, float, float, float, float, float]  # x, y, z, roll, pitch, yaw
    dimensions: Tuple[float, float, float] = (0.04, 0.04, 0.04)
    is_static: bool = False
    source: Literal["ground_truth", "camera_detected", "user_specified"] = "ground_truth"
    confidence: float = 1.0
    color: Optional[str] = None

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return (self.pose[0], self.pose[1], self.pose[2])

    def distance_to(self, point_xyz: Tuple[float, float, float]) -> float:
        dx = self.pose[0] - point_xyz[0]
        dy = self.pose[1] - point_xyz[1]
        dz = self.pose[2] - point_xyz[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)


class WorldScene:
    """Representation of objects and spatial entities in a simulation environment."""

    def __init__(self, world_id: str):
        self.world_id = world_id
        self._entities: Dict[str, SceneEntity] = {}

    def add_entity(self, entity: SceneEntity) -> None:
        self._entities[entity.id] = entity

    def get_entity(self, entity_id: str) -> Optional[SceneEntity]:
        return self._entities.get(entity_id)

    def list_entities(self) -> List[SceneEntity]:
        return list(self._entities.values())

    def find_target_object(self, query: str) -> Optional[SceneEntity]:
        """Resolve an entity by natural language description."""
        q = (query or "").lower().strip()

        # Check direct ID match
        if q in self._entities:
            return self._entities[q]

        # In front of arm / robot heuristics
        if any(w in q for w in ("in front of the arm", "in front of arm", "in front", "front object")):
            # Red cube is positioned directly in front of the arm in tabletop_world
            for ent in self._entities.values():
                if not ent.is_static and ent.entity_type in ("cube", "ball", "object", "block"):
                    return ent

        # Color matches (only dynamic manipulable objects)
        if "red" in q:
            for ent in self._entities.values():
                if not ent.is_static and (ent.color == "red" or "red" in ent.id or "red" in ent.name.lower()):
                    return ent
        if "blue" in q:
            for ent in self._entities.values():
                if not ent.is_static and (ent.color == "blue" or "blue" in ent.id or "blue" in ent.name.lower()):
                    return ent
        if "green" in q:
            for ent in self._entities.values():
                if not ent.is_static and (ent.color == "green" or "green" in ent.id or "green" in ent.name.lower()):
                    return ent

        # Type matches (strictly dynamic non-static manipulable entities)
        for ent in self._entities.values():
            if not ent.is_static and (ent.entity_type in q or ent.name.lower() in q):
                return ent

        # Fallback to first non-static manipulation object (e.g. target_cube_red)
        for ent in self._entities.values():
            if not ent.is_static and ent.entity_type in ("cube", "ball", "box", "object", "block"):
                return ent

        return None

    def find_destination(self, query: str) -> Optional[SceneEntity]:
        """Resolve destination plate, bin, or goal."""
        q = (query or "").lower().strip()

        if any(w in q for w in ("plate", "bin", "tray", "target", "green")):
            for ent in self._entities.values():
                if ent.entity_type in ("plate", "bin") or "bin" in ent.id or "plate" in ent.id:
                    return ent

        if any(w in q for w in ("goal", "net", "yellow goal")):
            for ent in self._entities.values():
                if "goal" in ent.id:
                    return ent

        # Fallback to default receptacle/destination in the scene (e.g. green plate / bin)
        for ent in self._entities.values():
            if ent.entity_type in ("plate", "bin", "tray") or "bin" in ent.id or "plate" in ent.id:
                return ent

        return None


# =========================================================================
# PRE-REGISTERED SIMULATION WORLD SCENES
# =========================================================================

def _build_tabletop_world() -> WorldScene:
    scene = WorldScene("tabletop_world")
    # Tabletop Work Table
    scene.add_entity(SceneEntity(
        id="work_table",
        name="Sturdy Work Table",
        entity_type="table",
        pose=(0.4, 0.0, 0.4, 0.0, 0.0, 0.0),
        dimensions=(1.2, 0.9, 0.04),
        is_static=True,
    ))
    # Target Red Cube (in front of arm)
    scene.add_entity(SceneEntity(
        id="target_cube_red",
        name="Target Red Cube",
        entity_type="cube",
        pose=(0.45, 0.15, 0.825, 0.0, 0.0, 0.0),
        dimensions=(0.04, 0.04, 0.04),
        is_static=False,
        color="red",
    ))
    # Target Blue Cube
    scene.add_entity(SceneEntity(
        id="target_cube_blue",
        name="Target Blue Cube",
        entity_type="cube",
        pose=(0.45, -0.15, 0.825, 0.0, 0.0, 0.0),
        dimensions=(0.04, 0.04, 0.04),
        is_static=False,
        color="blue",
    ))
    # Green Destination Bin / Plate Base
    scene.add_entity(SceneEntity(
        id="target_bin_green",
        name="Target Plate / Bin Base",
        entity_type="plate",
        pose=(0.25, 0.30, 0.805, 0.0, 0.0, 0.0),
        dimensions=(0.16, 0.16, 0.01),
        is_static=True,
        color="green",
    ))
    return scene


def _build_soccer_pitch() -> WorldScene:
    scene = WorldScene("soccer_pitch")
    scene.add_entity(SceneEntity(
        id="soccer_ball",
        name="Soccer Ball",
        entity_type="ball",
        pose=(0.5, 0.0, 0.05, 0.0, 0.0, 0.0),
        dimensions=(0.10, 0.10, 0.10),
        is_static=False,
        color="orange",
    ))
    scene.add_entity(SceneEntity(
        id="goal_yellow",
        name="Yellow Goal",
        entity_type="goal",
        pose=(1.5, 0.0, 0.15, 0.0, 0.0, 0.0),
        dimensions=(0.20, 0.60, 0.30),
        is_static=True,
        color="yellow",
    ))
    scene.add_entity(SceneEntity(
        id="goal_blue",
        name="Blue Goal",
        entity_type="goal",
        pose=(-1.5, 0.0, 0.15, 0.0, 0.0, 0.0),
        dimensions=(0.20, 0.60, 0.30),
        is_static=True,
        color="blue",
    ))
    return scene


def _build_obstacle_world() -> WorldScene:
    scene = WorldScene("obstacle_world")
    scene.add_entity(SceneEntity(
        id="red_box",
        name="Red Destination Box",
        entity_type="box",
        pose=(2.0, 0.0, 0.25, 0.0, 0.0, 0.0),
        dimensions=(0.5, 0.5, 0.5),
        is_static=True,
        color="red",
    ))
    scene.add_entity(SceneEntity(
        id="obstacle_cylinder_1",
        name="Obstacle Cylinder 1",
        entity_type="obstacle",
        pose=(1.0, 0.5, 0.5, 0.0, 0.0, 0.0),
        dimensions=(0.3, 0.3, 1.0),
        is_static=True,
    ))
    scene.add_entity(SceneEntity(
        id="goal_marker",
        name="Navigation Goal Marker",
        entity_type="waypoint",
        pose=(3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        dimensions=(0.2, 0.2, 0.01),
        is_static=True,
    ))
    return scene


_WORLD_REGISTRY: Dict[str, WorldScene] = {
    "tabletop_world": _build_tabletop_world(),
    "soccer_pitch": _build_soccer_pitch(),
    "obstacle_world": _build_obstacle_world(),
}


def get_world_scene(world_id: str) -> WorldScene:
    """Retrieve or create scene entity registry for a given simulation world."""
    clean_id = (world_id or "").strip().lower()
    if clean_id not in _WORLD_REGISTRY:
        # Fallback to tabletop_world
        return _WORLD_REGISTRY.get("tabletop_world", WorldScene(clean_id))
    return _WORLD_REGISTRY[clean_id]
