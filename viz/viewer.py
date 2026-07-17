"""Open3D viewer for camera trajectory and semantic object map."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import yaml

from mapping.object_map import ObjectMap
from semantic.projector import PoseTrajectory


def load_class_colors(classes_config: Path) -> dict[str, list[float]]:
    with classes_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {name: meta["color"] for name, meta in config["classes"].items()}


def show_map(
    object_map: ObjectMap,
    trajectory: Optional[PoseTrajectory] = None,
    classes_config: Optional[Path] = None,
    sphere_radius: float = 0.08,
) -> None:
    """Display trajectory and semantic objects in an Open3D window."""
    geometries: list[o3d.geometry.Geometry] = []

    colors = {}
    if classes_config is not None:
        colors = load_class_colors(classes_config)

    if trajectory and trajectory.poses:
        traj_points = np.array([p.translation for p in trajectory.poses])
        lines = [[i, i + 1] for i in range(len(traj_points) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(traj_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(
            [[0.3, 0.3, 0.9] for _ in lines]
        )
        geometries.append(line_set)

        # Start / end markers
        start = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 0.7)
        start.paint_uniform_color([0.2, 0.8, 0.2])
        start.translate(traj_points[0])
        geometries.append(start)

        end = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 0.7)
        end.paint_uniform_color([0.8, 0.2, 0.2])
        end.translate(traj_points[-1])
        geometries.append(end)

    for obj in object_map.objects:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        color = colors.get(obj.class_name, [0.7, 0.7, 0.7])
        sphere.paint_uniform_color(color)
        sphere.translate([obj.position.x, obj.position.y, obj.position.z])
        geometries.append(sphere)

    if not geometries:
        print("No geometry to display.")
        return

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Rescue-Net Semantic Object Map",
        width=1280,
        height=720,
    )


def save_map_snapshot(
    object_map: ObjectMap,
    trajectory: Optional[PoseTrajectory],
    output_path: Path,
    classes_config: Optional[Path] = None,
) -> None:
    """Save a minimal PLY point cloud snapshot (trajectory + objects) for headless envs."""
    points: list[list[float]] = []
    colors: list[list[float]] = []

    class_colors = load_class_colors(classes_config) if classes_config else {}

    if trajectory:
        for pose in trajectory.poses:
            points.append(pose.translation.tolist())
            colors.append([0.3, 0.3, 0.9])

    for obj in object_map.objects:
        points.append([obj.position.x, obj.position.y, obj.position.z])
        colors.append(class_colors.get(obj.class_name, [0.9, 0.5, 0.1]))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
