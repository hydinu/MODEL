# =============================================================================
# camera.py — Webcam capture handler (headless, no display)
# =============================================================================
import cv2
import sys
from config import DetectorConfig as Cfg


class Camera:
    """Thin wrapper around cv2.VideoCapture for headless operation."""

    def __init__(self, index: int = Cfg.CAMERA_INDEX):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            sys.exit(f'[Camera] Cannot open camera index {index}.')
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Cfg.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Cfg.FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'[Camera] Opened {w}x{h}')

    def __enter__(self):  return self
    def __exit__(self, *_): self.release()

    def read(self):       return self._cap.read()
    def release(self):
        if self._cap.isOpened():
            self._cap.release()
