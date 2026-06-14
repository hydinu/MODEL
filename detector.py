# =============================================================================
# detector.py — YOLOv8 person detection logic
# =============================================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List

from ultralytics import YOLO

from config import (
    MODEL_PATH,
    PERSON_CLASS_ID,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
)


@dataclass
class Detection:
    """Holds data for a single detected person."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) as a convenience tuple."""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def label(self) -> str:
        return f"Person  {self.confidence:.0%}"


class PersonDetector:
    """
    Wraps a YOLOv8 model and exposes a clean `detect()` interface that
    returns only *person* detections from a single BGR frame.

    The model is downloaded automatically on first use (~6 MB for yolov8n).
    """

    def __init__(self, model_path: str = MODEL_PATH, verbose: bool = False):
        print(f"[Detector] Loading model: {model_path}")
        self._model   = YOLO(model_path, verbose=verbose)
        self._verbose = verbose
        print("[Detector] Model ready.")

    # ── Public API ───────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on *frame* (BGR, HWC uint8) and return a list of
        :class:`Detection` objects, one per detected person.
        """
        results = self._model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[PERSON_CLASS_ID],  # Filter to persons only inside YOLO
            verbose=self._verbose,
            stream=False,
        )

        detections: List[Detection] = []

        # results is a list with one element per image; we always pass one frame
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id != PERSON_CLASS_ID:
                    continue   # Extra guard (should be redundant)

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(x1, y1, x2, y2, conf))

        return detections
