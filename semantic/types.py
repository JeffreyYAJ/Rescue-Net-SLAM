"""Shared semantic detection types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    class_name: str  # rescue-net class (victim, blocked_exit, ...)
    coco_label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2
    timestamp: float
