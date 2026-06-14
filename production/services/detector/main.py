# =============================================================================
# main.py — Detector service entry point (headless, no display window)
# =============================================================================
"""
Runs continuously:
  1. Read frame from webcam
  2. Detect persons with YOLOv8
  3. Every PUBLISH_INTERVAL seconds, write result to MongoDB

No GUI.  Designed to run inside a Docker container.
"""
import signal
import sys
import time

import cv2
import numpy as np

from camera    import Camera
from detector  import PersonDetector
from publisher import DetectionPublisher


class FPSCounter:
    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._fps   = 0.0
        self._last  = time.perf_counter()

    def tick(self) -> float:
        now        = time.perf_counter()
        instant    = 1.0 / max(now - self._last, 1e-9)
        self._fps  = self._alpha * instant + (1 - self._alpha) * self._fps
        self._last = now
        return self._fps


_running = True

def _handle_signal(signum, frame):
    global _running
    print(f'\n[Main] Signal {signum} received — shutting down …')
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def main() -> None:
    print('[Main] Crowd Detection — Detector Service (headless)')

    detector  = PersonDetector()
    publisher = DetectionPublisher()
    fps_ctr   = FPSCounter()

    with Camera() as cam:
        print('[Main] Camera open. Detecting …  (SIGTERM to stop)')
        while _running:
            ret, frame = cam.read()
            if not ret:
                print('[Main] Frame grab failed — retrying …')
                time.sleep(0.1)
                continue

            detections = detector.detect(frame)
            fps        = fps_ctr.tick()
            publisher.tick(detections, fps)

    print('[Main] Detector service stopped.')


if __name__ == '__main__':
    main()
