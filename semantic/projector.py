"""Project 2D detections to 3D world coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from mapping.object_map import BBox3D, SemanticObject, Vec3
from semantic.detector import Detection


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 5000.0

    @classmethod
    def from_yaml(cls, camera: dict) -> "CameraIntrinsics":
        return cls(
            fx=camera["fx"],
            fy=camera["fy"],
            cx=camera["cx"],
            cy=camera["cy"],
            width=camera["width"],
            height=camera["height"],
            depth_scale=camera.get("depth_scale", 5000.0),
        )


@dataclass
class CameraPose:
    """Camera pose in world frame (T_world_cam)."""

    timestamp: float
    translation: np.ndarray  # shape (3,)
    rotation: np.ndarray  # shape (3, 3)

    @classmethod
    def identity(cls, timestamp: float = 0.0) -> "CameraPose":
        return cls(
            timestamp=timestamp,
            translation=np.zeros(3),
            rotation=np.eye(3),
        )


@dataclass
class PoseTrajectory:
    poses: list[CameraPose]

    def nearest(self, timestamp: float) -> CameraPose:
        if not self.poses:
            return CameraPose.identity(timestamp)
        return min(self.poses, key=lambda p: abs(p.timestamp - timestamp))


def load_poses_tum(path: Path) -> PoseTrajectory:
    """Load TUM format: timestamp tx ty tz qx qy qz qw."""
    poses: list[CameraPose] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            t = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            qx, qy, qz, qw = map(float, parts[4:8])
            R = _quat_to_rot(qx, qy, qz, qw)
            poses.append(CameraPose(timestamp=ts, translation=t, rotation=R))
    return PoseTrajectory(poses=poses)


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) to rotation matrix."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )


class ObjectProjector:
    """Back-project 2D bboxes to 3D using RGB-D or monocular fallback."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        assumed_object_height_m: float = 1.7,
    ) -> None:
        self.intrinsics = intrinsics
        self.assumed_object_height_m = assumed_object_height_m

    @classmethod
    def from_slam_config(cls, config_path: Path) -> "ObjectProjector":
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        mono = config.get("mono_depth", {})
        return cls(
            intrinsics=CameraIntrinsics.from_yaml(config["camera"]),
            assumed_object_height_m=mono.get("assumed_object_height_m", 1.7),
        )

    def project_detection_rgbd(
        self,
        detection: Detection,
        depth_image: np.ndarray,
        pose: CameraPose,
        agent_id: str = "robot_0",
    ) -> Optional[SemanticObject]:
        point_cam = self._backproject_rgbd(detection, depth_image)
        if point_cam is None:
            return None
        point_world = self._cam_to_world(point_cam, pose)
        bbox3d = self._estimate_bbox3d(detection, point_cam, pose)
        return SemanticObject.create(
            class_name=detection.class_name,
            position=Vec3(point_world[0], point_world[1], point_world[2]),
            confidence=detection.confidence,
            timestamp=detection.timestamp,
            agent_id=agent_id,
            bbox3d=bbox3d,
        )

    def project_detection_mono(
        self,
        detection: Detection,
        pose: CameraPose,
        agent_id: str = "robot_0",
    ) -> Optional[SemanticObject]:
        point_cam = self._backproject_mono(detection)
        if point_cam is None:
            return None
        point_world = self._cam_to_world(point_cam, pose)
        return SemanticObject.create(
            class_name=detection.class_name,
            position=Vec3(point_world[0], point_world[1], point_world[2]),
            confidence=detection.confidence,
            timestamp=detection.timestamp,
            agent_id=agent_id,
        )

    def _backproject_rgbd(
        self,
        detection: Detection,
        depth_image: np.ndarray,
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = detection.bbox_xyxy
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        h, w = depth_image.shape[:2]
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)

        depth_raw = depth_image[cy, cx]
        if depth_image.dtype == np.uint16:
            depth_m = float(depth_raw) / self.intrinsics.depth_scale
        else:
            depth_m = float(depth_raw)

        if depth_m <= 0.01 or depth_m > 10.0 or np.isnan(depth_m):
            # Sample median in bbox patch
            depth_m = self._median_depth_in_bbox(depth_image, detection)
            if depth_m is None:
                return None

        return self._pixel_to_cam(cx, cy, depth_m)

    def _median_depth_in_bbox(
        self,
        depth_image: np.ndarray,
        detection: Detection,
    ) -> Optional[float]:
        x1, y1, x2, y2 = map(int, detection.bbox_xyxy)
        h, w = depth_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        patch = depth_image[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        if patch.dtype == np.uint16:
            depths = patch.astype(np.float32) / self.intrinsics.depth_scale
        else:
            depths = patch.astype(np.float32)

        valid = depths[(depths > 0.01) & (depths < 10.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _backproject_mono(self, detection: Detection) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = detection.bbox_xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bbox_height_px = max(y2 - y1, 1.0)

        # Pinhole: depth ≈ (f * real_height) / pixel_height
        depth_m = (self.intrinsics.fy * self.assumed_object_height_m) / bbox_height_px
        if depth_m <= 0.01 or depth_m > 50.0:
            return None

        return self._pixel_to_cam(int(cx), int(cy), depth_m)

    def _pixel_to_cam(self, u: int, v: int, depth_m: float) -> np.ndarray:
        x = (u - self.intrinsics.cx) * depth_m / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * depth_m / self.intrinsics.fy
        z = depth_m
        return np.array([x, y, z])

    def _cam_to_world(self, point_cam: np.ndarray, pose: CameraPose) -> np.ndarray:
        return pose.rotation @ point_cam + pose.translation

    def _estimate_bbox3d(
        self,
        detection: Detection,
        center_cam: np.ndarray,
        pose: CameraPose,
    ) -> BBox3D:
        x1, y1, x2, y2 = detection.bbox_xyxy
        depth = center_cam[2]
        width_m = (x2 - x1) * depth / self.intrinsics.fx
        height_m = (y2 - y1) * depth / self.intrinsics.fy
        depth_m = min(width_m, height_m) * 0.5

        center_world = self._cam_to_world(center_cam, pose)
        return BBox3D(
            center=Vec3(center_world[0], center_world[1], center_world[2]),
            size=Vec3(width_m, height_m, depth_m),
        )


def load_depth_image(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth image: {path}")
    return depth


def load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
