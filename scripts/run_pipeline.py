#!/usr/bin/env python3
"""Rescue-Net Phase 1 pipeline: SLAM poses + semantic detection → 3D object map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mapping.fusion import fuse_from_config
from mapping.object_map import ObjectMap
from semantic.dataset_loaders import (
    iter_euroc_cam0,
    iter_tum_rgbd,
    iter_video,
    load_datasets_config,
)
from semantic.detector import SemanticDetector
from semantic.projector import (
    CameraPose,
    ObjectProjector,
    PoseTrajectory,
    load_poses_tum,
    load_depth_image,
    load_rgb_image,
)
from viz.viewer import save_map_snapshot, show_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescue-Net Phase 1 offline pipeline")
    parser.add_argument(
        "--dataset",
        choices=["tum", "euroc", "video"],
        required=True,
        help="Dataset type",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to dataset root (overrides config/datasets.yaml)",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=REPO_ROOT / "output" / "poses.txt",
        help="TUM-format poses file from ORB-SLAM3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output",
        help="Directory for object_map.json and artifacts",
    )
    parser.add_argument(
        "--classes-config",
        type=Path,
        default=REPO_ROOT / "config" / "semantic" / "classes.yaml",
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=REPO_ROOT / "config" / "datasets.yaml",
    )
    parser.add_argument(
        "--slam-config",
        type=Path,
        default=None,
        help="SLAM/camera config YAML (defaults based on --dataset)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames processed",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Process every Nth frame (default: 5)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Skip interactive Open3D viewer",
    )
    parser.add_argument(
        "--run-slam",
        action="store_true",
        help="Run ORB-SLAM3 via slam/run_slam.sh before semantic pipeline",
    )
    parser.add_argument(
        "--identity-poses",
        action="store_true",
        help="Use identity poses (skip poses file, for testing without SLAM)",
    )
    return parser.parse_args()


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root is not None:
        return args.dataset_root

    cfg = load_datasets_config(args.datasets_config)
    if args.dataset == "tum":
        return Path(cfg["tum"]["freiburg1_xyz"]["root"])
    if args.dataset == "euroc":
        return Path(cfg["euroc"]["mav0"]["root"])
    video_cfg = cfg["video"]["disaster_demo"]
    return Path(video_cfg["root"]) / video_cfg["file"]


def resolve_slam_config(args: argparse.Namespace) -> Path:
    if args.slam_config is not None:
        return args.slam_config
    if args.dataset == "tum":
        return REPO_ROOT / "config" / "slam" / "tum_rgbd.yaml"
    return REPO_ROOT / "config" / "slam" / "euroc_mono.yaml"


def maybe_run_slam(args: argparse.Namespace, dataset_root: Path) -> None:
    if not args.run_slam:
        return

    import subprocess

    script = REPO_ROOT / "slam" / "run_slam.sh"
    cmd = [str(script), args.dataset, str(dataset_root), str(args.poses)]
    print(f"Running SLAM: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_trajectory(args: argparse.Namespace) -> PoseTrajectory:
    if args.identity_poses:
        print("Using identity poses (no SLAM trajectory).")
        return PoseTrajectory(poses=[])

    if not args.poses.exists():
        print(
            f"Poses file not found: {args.poses}\n"
            "Run with --run-slam, provide --poses, or use --identity-poses."
        )
        sys.exit(1)

    trajectory = load_poses_tum(args.poses)
    print(f"Loaded {len(trajectory.poses)} poses from {args.poses}")
    return trajectory


def process_tum(
    args: argparse.Namespace,
    dataset_root: Path,
    detector: SemanticDetector,
    projector: ObjectProjector,
    trajectory: PoseTrajectory,
) -> list:
    observations = []
    frame_iter = iter_tum_rgbd(
        dataset_root,
        max_frames=args.max_frames,
        stride=args.stride,
    )

    for sample in tqdm(frame_iter, desc="TUM RGB-D"):
        if not sample.rgb_path.exists():
            continue

        rgb = load_rgb_image(sample.rgb_path)
        detections = detector.detect_frame(rgb, sample.timestamp)

        if args.identity_poses:
            pose = CameraPose.identity(sample.timestamp)
        else:
            pose = trajectory.nearest(sample.timestamp)

        depth = None
        if sample.depth_path and sample.depth_path.exists():
            depth = load_depth_image(sample.depth_path)

        for det in detections:
            if depth is not None:
                obj = projector.project_detection_rgbd(det, depth, pose)
            else:
                obj = projector.project_detection_mono(det, pose)
            if obj is not None:
                observations.append(obj)

    return observations


def process_euroc(
    args: argparse.Namespace,
    dataset_root: Path,
    detector: SemanticDetector,
    projector: ObjectProjector,
    trajectory: PoseTrajectory,
) -> list:
    observations = []
    frame_iter = iter_euroc_cam0(
        dataset_root,
        max_frames=args.max_frames,
        stride=args.stride,
    )

    for sample in tqdm(frame_iter, desc="EuRoC"):
        if not sample.rgb_path.exists():
            continue

        rgb = load_rgb_image(sample.rgb_path)
        detections = detector.detect_frame(rgb, sample.timestamp)

        if args.identity_poses:
            pose = CameraPose.identity(sample.timestamp)
        else:
            pose = trajectory.nearest(sample.timestamp)

        for det in detections:
            obj = projector.project_detection_mono(det, pose)
            if obj is not None:
                observations.append(obj)

    return observations


def process_video(
    args: argparse.Namespace,
    video_path: Path,
    detector: SemanticDetector,
    projector: ObjectProjector,
    trajectory: PoseTrajectory,
) -> list:
    observations = []
    frame_iter = iter_video(
        video_path,
        max_frames=args.max_frames,
        stride=args.stride,
    )

    for timestamp, rgb in tqdm(frame_iter, desc="Video"):
        detections = detector.detect_frame(rgb, timestamp)

        if args.identity_poses:
            pose = CameraPose.identity(timestamp)
        else:
            pose = trajectory.nearest(timestamp)

        for det in detections:
            obj = projector.project_detection_mono(det, pose)
            if obj is not None:
                observations.append(obj)

    return observations


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = resolve_dataset_root(args)
    slam_config = resolve_slam_config(args)

    if args.dataset in ("tum", "euroc"):
        maybe_run_slam(args, dataset_root)

    trajectory = load_trajectory(args)

    print("Loading YOLO detector...")
    detector = SemanticDetector(args.classes_config, model_name=args.model)
    projector = ObjectProjector.from_slam_config(slam_config)

    if args.dataset == "tum":
        if not dataset_root.exists():
            print(f"TUM dataset not found: {dataset_root}")
            sys.exit(1)
        observations = process_tum(
            args, dataset_root, detector, projector, trajectory
        )
    elif args.dataset == "euroc":
        if not dataset_root.exists():
            print(f"EuRoC dataset not found: {dataset_root}")
            sys.exit(1)
        observations = process_euroc(
            args, dataset_root, detector, projector, trajectory
        )
    else:
        if not dataset_root.exists():
            print(f"Video not found: {dataset_root}")
            sys.exit(1)
        observations = process_video(
            args, dataset_root, detector, projector, trajectory
        )

    print(f"Raw observations: {len(observations)}")

    object_map = fuse_from_config(
        observations,
        args.classes_config,
        agent_id="robot_0",
    )

    out_json = args.output_dir / "object_map.json"
    object_map.save_json(out_json)

    size_kb = object_map.size_bytes_estimate() / 1024
    print(f"Fused objects: {len(object_map.objects)}")
    print(f"Exported {out_json} (~{size_kb:.1f} KB)")

    by_class: dict[str, int] = {}
    for obj in object_map.objects:
        by_class[obj.class_name] = by_class.get(obj.class_name, 0) + 1
    if by_class:
        print("Objects by class:", by_class)

    snapshot = args.output_dir / "map_snapshot.ply"
    save_map_snapshot(object_map, trajectory, snapshot, args.classes_config)
    print(f"Saved snapshot: {snapshot}")

    if not args.no_view:
        show_map(object_map, trajectory, args.classes_config)


if __name__ == "__main__":
    main()
