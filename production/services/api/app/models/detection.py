# =============================================================================
# models/detection.py — Detection document serialiser + validator
# =============================================================================
from bson import ObjectId


def serialize_detection(doc: dict) -> dict:
    """Convert a raw MongoDB document to a JSON-safe dict."""
    return {
        'id'         : str(doc['_id']),
        'timestamp'  : doc['timestamp'].isoformat() if doc.get('timestamp') else None,
        'crowd_count': doc.get('crowd_count', 0),
        'fps'        : doc.get('fps', 0.0),
        'camera_id'  : doc.get('camera_id', 'cam-0'),
        'boxes'      : doc.get('boxes', []),
    }


def validate_detection_payload(payload: dict) -> list[str]:
    """
    Validate an ingest payload.  Returns a list of error strings;
    empty list means the payload is valid.
    """
    errors = []
    if 'crowd_count' not in payload:
        errors.append('crowd_count is required.')
    elif not isinstance(payload['crowd_count'], int) or payload['crowd_count'] < 0:
        errors.append('crowd_count must be a non-negative integer.')
    return errors
