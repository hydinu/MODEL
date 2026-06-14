# =============================================================================
# main.py — Entry point for the Crowd Detection System
# =============================================================================
"""
Usage
-----
    python main.py                        # Use defaults from config.py
    python main.py --camera 1             # Use camera index 1
    python main.py --model yolov8s.pt     # Use a larger / more accurate model
    python main.py --no-heatmap           # Disable density heatmap overlay
    python main.py --threshold 5          # Override crowd alert threshold

Hotkeys
-------
    Q / Esc  Quit
    H        Toggle heatmap overlay on / off at runtime
    A        Toggle alert banner on / off at runtime
"""

import argparse
import time

import cv2

from camera import Camera
from detector import PersonDetector
from display import FrameRenderer
from alerts import AlertSystem, draw_alert_overlay
from heatmap import HeatmapAccumulator, draw_colorbar_legend
from logger import CrowdLogger
from config import WINDOW_TITLE, HEATMAP_ENABLED, ALERT_THRESHOLD


# ── Argument parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time crowd detection using YOLOv8 + OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera", type=str, default=None,
        help="Camera index (e.g. 0) or IP stream URL (overrides config.py CAMERA_INDEX).",
    )
    parser.add_argument(
        "--camera-name", type=str, default="Default Camera",
        help="Friendly name for the camera source, logged to MongoDB.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to YOLOv8 .pt weight file (overrides config.py MODEL_PATH).",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Detection confidence threshold 0..1 (overrides config.py).",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Disable the crowd density heatmap overlay.",
    )
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="Crowd count that triggers an alert (overrides config.py ALERT_THRESHOLD).",
    )
    return parser.parse_args()


# ── FPS tracker ──────────────────────────────────────────────────────────────

class FPSCounter:
    """Exponential moving-average FPS counter."""

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._fps   = 0.0
        self._last  = time.perf_counter()

    def tick(self) -> float:
        now       = time.perf_counter()
        instant   = 1.0 / max(now - self._last, 1e-9)
        self._fps = self._alpha * instant + (1 - self._alpha) * self._fps
        self._last = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Override config from CLI flags when supplied
    cam_index  = args.camera if args.camera  is not None else None
    model_path = args.model  if args.model   is not None else None
    conf       = args.conf   if args.conf    is not None else None

    # Try to convert numeric string camera argument to integer index
    if cam_index is not None:
        try:
            cam_index = int(cam_index)
        except ValueError:
            pass

    # Patch config values if overridden (must happen before importing sub-modules
    # that read config at import time; we do it here early, before construction)
    if conf is not None:
        import config
        config.CONFIDENCE_THRESHOLD = conf

    # ── Initialise sub-systems ────────────────────────────────────────────────
    detector = PersonDetector(
        **({"model_path": model_path} if model_path else {})
    )
    renderer  = FrameRenderer()
    fps_ctr   = FPSCounter()
    logger    = CrowdLogger()         # opens / creates logs/crowd_log.csv

    # Alert system
    alert_threshold = args.threshold if args.threshold is not None else ALERT_THRESHOLD
    alert_sys       = AlertSystem(threshold=alert_threshold, camera_name=args.camera_name)
    alerts_on       = True            # runtime toggle via A key

    # Heatmap — lazy init after first frame (dimensions known then)
    heatmap_on  = HEATMAP_ENABLED and not args.no_heatmap
    hm_accum    = None                # HeatmapAccumulator, created on first frame

    cam_kwargs = {}
    if cam_index is not None:
        cam_kwargs["index"] = cam_index

    with Camera(**cam_kwargs) as cam, logger, alert_sys:
        print(f"\n[Main] Starting capture. Window: '{WINDOW_TITLE}'")
        hm_status = "ON" if heatmap_on else "OFF"
        print(f"[Main] Heatmap overlay : {hm_status}  (press H to toggle)")
        print(f"[Main] Alert threshold : {alert_threshold} persons  (press A to toggle banner)")
        print("[Main] Press  Q  or  Esc  to quit.\n")

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

        while True:
            ret, frame = cam.read()
            if not ret:
                print("[Main] Failed to grab frame — retrying …")
                continue

            # ── Detect persons ───────────────────────────────────────────────
            detections  = detector.detect(frame)
            crowd_count = len(detections)

            # ── Calculate FPS ────────────────────────────────────────────────
            fps = fps_ctr.tick()

            # ── Heatmap ───────────────────────────────────────────────────────
            h_px, w_px = frame.shape[:2]

            # Lazy-create accumulator once we know the actual frame size
            if hm_accum is None:
                hm_accum = HeatmapAccumulator(frame_h=h_px, frame_w=w_px)
                print(f"[Main] Heatmap initialised ({w_px}×{h_px}).")

            hm_accum.update(detections)   # decay + stamp blobs (always run)

            if heatmap_on:
                frame = hm_accum.render(frame)           # blended copy
                draw_colorbar_legend(frame)              # legend in-place

            # ── Render HUD + bounding boxes (on top of heatmap) ──────────────
            renderer.render(frame, detections, fps)

            # ── Alert system ──────────────────────────────────────────────────
            alert_state = alert_sys.check(crowd_count)
            if alerts_on:
                draw_alert_overlay(frame, alert_state)   # in-place, top banner

            # ── Show frame ───────────────────────────────────────────────────
            cv2.imshow(WINDOW_TITLE, frame)

            # ── Log crowd count to CSV (every 30 s) ──────────────────────────
            logger.tick(crowd_count)

            # ── Console log (throttled every ~30 frames) ─────────────────────
            if int(fps_ctr.fps * time.perf_counter()) % 30 == 0:
                print(f"\r  FPS: {fps:5.1f}  |  Persons: {crowd_count:3d}   ", end="", flush=True)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # 27 = Esc
                print("\n[Main] Quit signal received.")
                break
            if key in (ord("h"), ord("H")):
                heatmap_on = not heatmap_on
                if not heatmap_on and hm_accum is not None:
                    hm_accum.reset()              # clear buffer when hiding
                print(f"\n[Main] Heatmap {'ON' if heatmap_on else 'OFF'}")
            if key in (ord("a"), ord("A")):
                alerts_on = not alerts_on
                print(f"\n[Main] Alert banner {'ON' if alerts_on else 'OFF'}")

    cv2.destroyAllWindows()
    print("[Main] Done.")


if __name__ == "__main__":
    main()
