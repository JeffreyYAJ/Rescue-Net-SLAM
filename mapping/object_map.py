"""Lightweight semantic object map data model and serialization."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, values: list[float]) -> "Vec3":
        return cls(x=values[0], y=values[1], z=values[2])


@dataclass
class BBox3D:
    center: Vec3
    size: Vec3  # width, height, depth in meters

    def as_dict(self) -> dict[str, Any]:
        return {"center": self.center.as_list(), "size": self.size.as_list()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BBox3D":
        return cls(
            center=Vec3.from_list(data["center"]),
            size=Vec3.from_list(data["size"]),
        )


@dataclass
class SemanticObject:
    id: str
    class_name: str
    position: Vec3
    confidence: float
    observations: int
    last_seen_ts: float
    agent_id: str = "robot_0"
    bbox3d: Optional[BBox3D] = None

    @classmethod
    def create(
        cls,
        class_name: str,
        position: Vec3,
        confidence: float,
        timestamp: float,
        agent_id: str = "robot_0",
        bbox3d: Optional[BBox3D] = None,
    ) -> "SemanticObject":
        return cls(
            id=str(uuid.uuid4()),
            class_name=class_name,
            position=position,
            confidence=confidence,
            observations=1,
            last_seen_ts=timestamp,
            agent_id=agent_id,
            bbox3d=bbox3d,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "class_name": self.class_name,
            "position": self.position.as_list(),
            "confidence": self.confidence,
            "observations": self.observations,
            "last_seen_ts": self.last_seen_ts,
            "agent_id": self.agent_id,
        }
        if self.bbox3d is not None:
            data["bbox3d"] = self.bbox3d.as_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticObject":
        bbox3d = None
        if "bbox3d" in data and data["bbox3d"] is not None:
            bbox3d = BBox3D.from_dict(data["bbox3d"])
        return cls(
            id=data["id"],
            class_name=data["class_name"],
            position=Vec3.from_list(data["position"]),
            confidence=data["confidence"],
            observations=data["observations"],
            last_seen_ts=data["last_seen_ts"],
            agent_id=data.get("agent_id", "robot_0"),
            bbox3d=bbox3d,
        )


@dataclass
class ObjectMap:
    """Dictionary of geolocated semantic objects (~few KB export)."""

    objects: list[SemanticObject] = field(default_factory=list)
    agent_id: str = "robot_0"
    frame_id: str = "world"

    def add(self, obj: SemanticObject) -> None:
        self.objects.append(obj)

    def size_bytes_estimate(self) -> int:
        return len(json.dumps(self.to_dict(), separators=(",", ":")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "frame_id": self.frame_id,
            "object_count": len(self.objects),
            "objects": [obj.to_dict() for obj in self.objects],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectMap":
        return cls(
            objects=[SemanticObject.from_dict(o) for o in data["objects"]],
            agent_id=data.get("agent_id", "robot_0"),
            frame_id=data.get("frame_id", "world"),
        )

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "ObjectMap":
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
