# =============================================================================
# config.py — Central configuration for the Crowd Detection System
# =============================================================================

# ── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0          # 0 = default webcam; change for external cam
FRAME_WIDTH         = 1280       # Capture resolution width  (px)
FRAME_HEIGHT        = 720        # Capture resolution height (px)

# ── YOLOv8 Model ────────────────────────────────────────────────────────────
MODEL_PATH          = "yolov8n.pt"   # nano model for speed; swap yolov8s/m/l/x for accuracy
PERSON_CLASS_ID     = 0             # COCO class 0 = "person"
CONFIDENCE_THRESHOLD = 0.40         # Minimum detection confidence (0.0 – 1.0)
IOU_THRESHOLD       = 0.45          # NMS IoU threshold

# ── Bounding Box ────────────────────────────────────────────────────────────
BOX_COLOR           = (0, 255, 128)  # Bright green  (B, G, R)
BOX_THICKNESS       = 2

LABEL_FONT_SCALE    = 0.55
LABEL_THICKNESS     = 1
LABEL_COLOR         = (255, 255, 255)  # White text
LABEL_BG_COLOR      = (0, 200, 100)   # Label background

# ── HUD (Heads-Up Display) ──────────────────────────────────────────────────
HUD_FONT_SCALE      = 0.7
HUD_THICKNESS       = 2
FPS_COLOR           = (0, 220, 255)   # Cyan
COUNT_COLOR         = (255, 180, 0)   # Amber
TITLE_COLOR         = (255, 255, 255) # White
HUD_BG_ALPHA        = 0.45           # Overlay transparency (0=invisible, 1=opaque)

# ── Window ──────────────────────────────────────────────────────────────────
WINDOW_TITLE        = "Crowd Detection — YOLOv8 + OpenCV"

# ── CSV Logging ──────────────────────────────────────────────────────────────
CSV_LOG_PATH        = "logs/crowd_log.csv"  # Relative to project root
CSV_LOG_INTERVAL    = 30                    # Seconds between log entries

# ── Heatmap ───────────────────────────────────────────────────────────────────
import cv2 as _cv2                          # noqa: E402 (import inside config is intentional)

HEATMAP_ENABLED     = True                  # Toggle heatmap overlay on/off
HEATMAP_ALPHA       = 0.55                  # Blend opacity  (0 = invisible, 1 = opaque)
HEATMAP_DECAY       = 0.92                  # Per-frame heat decay factor  (0–1)
#   0.92 @ 30 FPS → heat half-life ≈ 8 frames (~0.27 s) — stays very responsive
#   Raise toward 0.99 for a longer "memory"; lower toward 0.80 for faster fade

HEATMAP_SIGMA_SCALE = 1.2                   # Gaussian spread as fraction of half-bbox size
HEATMAP_AMPLITUDE   = 1.0                   # Peak heat value added per person per frame
HEATMAP_BLUR_KERNEL = 51                    # Final smoothing kernel size (odd int)

HEATMAP_COLORMAP    = _cv2.COLORMAP_JET     # Colourmap: COLORMAP_JET / INFERNO / HOT / TURBO
HEATMAP_SHOW_LEGEND = True                  # Draw colourbar legend in bottom-right corner

# ── Alert System ──────────────────────────────────────────────────────────────
ALERT_THRESHOLD     = 10                    # Persons count that triggers an alert
# ALERT_DB_PATH       = "logs/alerts.db"     # (Retired SQLite path)
MONGO_URI           = "mongodb://localhost:27017/MODEL"  # MongoDB server URI
MONGO_DB            = "crowd"                            # MongoDB database name
MONGO_COLLECTION    = "people_count"                     # MongoDB collection name (people_count)
ALERT_COOLDOWN      = 60                    # Min seconds between successive DB writes
#   Lower to 10 for rapid testing; raise to 300 for production deployments

ALERT_BLINK_HZ      = 2.0                  # Banner pulse frequency (cycles per second)
#   2.0 → banner pulsates twice per second — noticeable but not jarring
