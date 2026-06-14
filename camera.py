# =============================================================================
# camera.py — Webcam capture handler
# =============================================================================

import cv2
import sys
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class Camera:
    """
    Thin wrapper around cv2.VideoCapture.

    Usage:
        with Camera() as cam:
            ret, frame = cam.read()
    """

    def __init__(self, index = CAMERA_INDEX):
        if isinstance(index, str):
            # Network stream or video file (e.g., http://... or rtsp://...)
            self._cap = cv2.VideoCapture(index)
        else:
            # Local USB camera - try CAP_DSHOW first (faster on Windows)
            self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                # Fallback to default backend if CAP_DSHOW fails
                self._cap.release()
                self._cap = cv2.VideoCapture(index)

        if not self._cap.isOpened():
            sys.exit(f"[ERROR] Cannot open camera ({index}). "
                     "Check that the webcam/stream is connected and not in use.")

        # Request resolution (driver may clamp to nearest supported size)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimise latency

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened source {index}  |  resolution {actual_w}×{actual_h}")

    # ── Context manager support ──────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    # ── Public API ───────────────────────────────────────────────────────────
    def read(self):
        """Return (success: bool, frame: np.ndarray)."""
        return self._cap.read()

    def release(self):
        """Release the underlying VideoCapture resource."""
        if self._cap.isOpened():
            self._cap.release()
            print("[Camera] Released.")

    @property
    def fps(self) -> float:
        """Reported FPS of the capture device (may be 0 for webcams)."""
        return self._cap.get(cv2.CAP_PROP_FPS)
