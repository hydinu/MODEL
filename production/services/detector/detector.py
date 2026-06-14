# =============================================================================
# detector.py — YOLOv8 person detection (headless)
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from ultralytics import YOLO
from config import DetectorConfig as Cfg


@dataclass
class Detection:
    x1: int; y1: int; x2: int; y2: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            'x1': self.x1, 'y1': self.y1,
            'x2': self.x2, 'y2': self.y2,
            'confidence': round(self.confidence, 4),
        }


class PersonDetector:
    def __init__(self):
        print(f'[Detector] Loading model: {Cfg.MODEL_PATH}')
        self._model = YOLO(Cfg.MODEL_PATH, verbose=False)
        print('[Detector] Ready.')

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=frame,
            conf=Cfg.CONFIDENCE_THRESHOLD,
            iou=Cfg.IOU_THRESHOLD,
            classes=[Cfg.PERSON_CLASS_ID],
            verbose=False,
        )
        dets = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets.append(Detection(x1, y1, x2, y2, float(box.conf[0])))
        return dets
