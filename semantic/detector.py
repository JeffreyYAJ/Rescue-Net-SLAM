"""YOLOv8-nano semantic detector with COCO → rescue class mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from ultralytics import YOLO


@dataclass
class Detection:
    class_name: str  # rescue-net class (victim, blocked_exit, ...)
    coco_label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2
    timestamp: float


class SemanticDetector:
    """Edge-friendly YOLO detector mapping COCO labels to rescue classes."""

    def __init__(
        self,
        classes_config: Path,
        model_name: str = "yolov8n.pt",
        device: Optional[str] = None,
    ) -> None:
        with classes_config.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._class_config = config["classes"]
        self._coco_to_rescue: dict[str, tuple[str, float]] = {}
        for rescue_class, meta in self._class_config.items():
            threshold = meta["confidence_threshold"]
            for coco_label in meta.get("coco_labels", []):
                self._coco_to_rescue[coco_label.lower()] = (rescue_class, threshold)

        self._model = YOLO(model_name)
        self._device = device

    def detect_frame(
        self,
        image: np.ndarray,
        timestamp: float,
    ) -> list[Detection]:
        """Run detection on a BGR/RGB numpy image."""
        results = self._model.predict(
            source=image,
            verbose=False,
            device=self._device,
        )
        detections: list[Detection] = []

        if not results:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            coco_label = names[cls_id].lower()
            confidence = float(box.conf.item())

            mapping = self._coco_to_rescue.get(coco_label)
            if mapping is None:
                continue

            rescue_class, threshold = mapping
            if confidence < threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=rescue_class,
                    coco_label=coco_label,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                    timestamp=timestamp,
                )
            )

        return detections

    def get_class_colors(self, classes_config: Path) -> dict[str, list[float]]:
        with classes_config.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return {
            name: meta["color"]
            for name, meta in config["classes"].items()
        }
