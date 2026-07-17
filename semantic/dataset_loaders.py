"""Dataset frame loaders for TUM RGB-D, EuRoC, and video files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import yaml


@dataclass
class FrameSample:
    timestamp: float
    rgb_path: Path
    depth_path: Optional[Path] = None


def load_datasets_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_tum_associations(association_file: Path) -> list[tuple[float, str, float, str]]:
    """Parse TUM association file: ts_rgb rgb_path ts_depth depth_path."""
    pairs: list[tuple[float, str, float, str]] = []
    with association_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            ts_rgb, rgb_rel = float(parts[0]), parts[1]
            ts_depth, depth_rel = float(parts[2]), parts[3]
            pairs.append((ts_rgb, rgb_rel, ts_depth, depth_rel))
    return pairs


def iter_tum_rgbd(
    dataset_root: Path,
    association_file: Optional[Path] = None,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> Iterator[FrameSample]:
    """Iterate synchronized RGB-D frames from a TUM dataset."""
    if association_file is None:
        association_file = dataset_root / "associations.txt"

    if not association_file.exists():
        # Build associations from rgb.txt and depth.txt if missing
        association_file = _build_tum_associations(dataset_root)

    pairs = parse_tum_associations(association_file)
    count = 0
    for i, (ts_rgb, rgb_rel, _ts_depth, depth_rel) in enumerate(pairs):
        if i % stride != 0:
            continue
        yield FrameSample(
            timestamp=ts_rgb,
            rgb_path=dataset_root / rgb_rel,
            depth_path=dataset_root / depth_rel,
        )
        count += 1
        if max_frames is not None and count >= max_frames:
            break


def _build_tum_associations(dataset_root: Path) -> Path:
    """Create associations.txt from rgb.txt and depth.txt (nearest timestamp match)."""
    rgb_entries = _parse_tum_timestamp_file(dataset_root / "rgb.txt")
    depth_entries = _parse_tum_timestamp_file(dataset_root / "depth.txt")

    out_path = dataset_root / "associations.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for ts_rgb, rgb_rel in rgb_entries:
            ts_depth, depth_rel = min(depth_entries, key=lambda e: abs(e[0] - ts_rgb))
            f.write(f"{ts_rgb} {rgb_rel} {ts_depth} {depth_rel}\n")
    return out_path


def _parse_tum_timestamp_file(path: Path) -> list[tuple[float, str]]:
    entries: list[tuple[float, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entries.append((float(parts[0]), parts[1]))
    return entries


def iter_euroc_cam0(
    dataset_root: Path,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> Iterator[FrameSample]:
    """Iterate EuRoC cam0 images using cam0/data.csv timestamps."""
    cam_dir = dataset_root / "cam0" / "data"
    times_file = dataset_root / "cam0" / "data.csv"
    timestamps: list[float] = []

    with times_file.open("r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 1:
                continue
            # EuRoC timestamps are in nanoseconds
            timestamps.append(int(parts[0]) / 1e9)

    count = 0
    for i, ts in enumerate(timestamps):
        if i % stride != 0:
            continue
        image_name = f"{int(ts * 1e9):019d}.png"
        rgb_path = cam_dir / image_name
        if not rgb_path.exists():
            continue
        yield FrameSample(timestamp=ts, rgb_path=rgb_path)
        count += 1
        if max_frames is not None and count >= max_frames:
            break


def iter_video(
    video_path: Path,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> Iterator[tuple[float, object]]:
    """Yield (timestamp_sec, rgb_numpy) from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            timestamp = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield timestamp, rgb
            count += 1
            if max_frames is not None and count >= max_frames:
                break
        frame_idx += 1

    cap.release()
