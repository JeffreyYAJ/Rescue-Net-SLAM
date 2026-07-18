"""Basic unit tests for Rescue-Net Phase 1 (no GPU/dataset required)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mapping.fusion import fuse_observations
from mapping.object_map import ObjectMap, SemanticObject, Vec3
from semantic.projector import CameraIntrinsics, CameraPose, ObjectProjector
from semantic.types import Detection


class TestObjectMap(unittest.TestCase):
    def test_json_roundtrip(self) -> None:
        obj = SemanticObject.create("victim", Vec3(1.0, 2.0, 3.0), 0.9, 1.5)
        obj_map = ObjectMap(objects=[obj])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            obj_map.save_json(path)
            loaded = ObjectMap.load_json(path)
        self.assertEqual(len(loaded.objects), 1)
        self.assertEqual(loaded.objects[0].class_name, "victim")
        self.assertLess(obj_map.size_bytes_estimate(), 100_000)

    def test_fusion_clusters_nearby(self) -> None:
        o1 = SemanticObject.create("victim", Vec3(0.0, 0.0, 0.0), 0.8, 1.0)
        o2 = SemanticObject.create("victim", Vec3(0.1, 0.0, 0.0), 0.7, 1.1)
        o3 = SemanticObject.create("blocked_exit", Vec3(5.0, 0.0, 0.0), 0.6, 2.0)
        fused = fuse_observations([o1, o2, o3], cluster_distance_m=0.5)
        classes = {o.class_name for o in fused.objects}
        self.assertIn("victim", classes)
        self.assertIn("blocked_exit", classes)
        victim = next(o for o in fused.objects if o.class_name == "victim")
        self.assertEqual(victim.observations, 2)


class TestProjector(unittest.TestCase):
    def test_rgbd_backprojection(self) -> None:
        import numpy as np

        intrinsics = CameraIntrinsics(
            fx=525.0, fy=525.0, cx=320.0, cy=240.0, width=640, height=480
        )
        projector = ObjectProjector(intrinsics)
        det = Detection(
            class_name="victim",
            coco_label="person",
            confidence=0.9,
            bbox_xyxy=(300.0, 200.0, 340.0, 360.0),
            timestamp=0.0,
        )
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[280, 320] = 5000  # 1 meter
        pose = CameraPose.identity(0.0)
        obj = projector.project_detection_rgbd(det, depth, pose)
        self.assertIsNotNone(obj)
        assert obj is not None
        self.assertAlmostEqual(obj.position.z, 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
