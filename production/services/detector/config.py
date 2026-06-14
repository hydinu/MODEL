# =============================================================================
# config.py — Detector service configuration (reads from environment)
# =============================================================================
import os


class DetectorConfig:
    # MongoDB
    MONGO_URI   = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
    MONGO_DB    = os.environ.get('MONGO_DB', 'crowd_detection')

    # Camera
    CAMERA_INDEX   = int(os.environ.get('CAMERA_INDEX', 0))
    FRAME_WIDTH    = int(os.environ.get('FRAME_WIDTH', 1280))
    FRAME_HEIGHT   = int(os.environ.get('FRAME_HEIGHT', 720))

    # YOLOv8
    MODEL_PATH           = os.environ.get('MODEL_PATH', 'weights/yolov8n.pt')
    PERSON_CLASS_ID      = 0
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.40))
    IOU_THRESHOLD        = float(os.environ.get('IOU_THRESHOLD', 0.45))

    # Publishing
    PUBLISH_INTERVAL = int(os.environ.get('PUBLISH_INTERVAL', 30))  # seconds
    ALERT_THRESHOLD  = int(os.environ.get('ALERT_THRESHOLD', 10))
    CAMERA_ID        = os.environ.get('CAMERA_ID', 'cam-0')
