# =============================================================================
# routes/detections.py — Crowd detection endpoints
# =============================================================================
"""
Endpoints
─────────
GET  /api/v1/detections/live       Latest crowd count snapshot
GET  /api/v1/detections            Paginated history with optional filters
POST /api/v1/detections            Ingest a new detection record (from detector service)
GET  /api/v1/detections/<id>       Single detection record
"""
from datetime import datetime, timezone

from bson import ObjectId
from flask import Blueprint, current_app, jsonify, request

from ..extensions import mongo
from ..models.detection import serialize_detection, validate_detection_payload

detections_bp = Blueprint('detections', __name__)


# ── GET /api/v1/detections/live ───────────────────────────────────────────────
@detections_bp.get('/detections/live')
def live():
    """Return the single most-recent detection document."""
    doc = mongo.db.detections.find_one(
        {}, sort=[('timestamp', -1)]
    )
    if not doc:
        return jsonify({'message': 'No detections yet.'}), 204
    return jsonify(serialize_detection(doc)), 200


# ── GET /api/v1/detections ────────────────────────────────────────────────────
@detections_bp.get('/detections')
def list_detections():
    """
    Paginated detection history.

    Query params
    ────────────
    page      : int  (default 1)
    per_page  : int  (default 50, max 500)
    from      : ISO datetime  (inclusive lower bound)
    to        : ISO datetime  (inclusive upper bound)
    min_count : int  (filter by minimum crowd count)
    """
    cfg       = current_app.config
    page      = max(int(request.args.get('page', 1)), 1)
    per_page  = min(
        int(request.args.get('per_page', cfg['DEFAULT_PAGE_SIZE'])),
        cfg['MAX_PAGE_SIZE']
    )
    query = {}

    # Date range filter
    dt_filter = {}
    if from_dt := request.args.get('from'):
        dt_filter['$gte'] = datetime.fromisoformat(from_dt)
    if to_dt := request.args.get('to'):
        dt_filter['$lte'] = datetime.fromisoformat(to_dt)
    if dt_filter:
        query['timestamp'] = dt_filter

    if min_count := request.args.get('min_count'):
        query['crowd_count'] = {'$gte': int(min_count)}

    total = mongo.db.detections.count_documents(query)
    docs  = list(
        mongo.db.detections
        .find(query)
        .sort('timestamp', -1)
        .skip((page - 1) * per_page)
        .limit(per_page)
    )

    return jsonify({
        'data'      : [serialize_detection(d) for d in docs],
        'pagination': {
            'page'      : page,
            'per_page'  : per_page,
            'total'     : total,
            'pages'     : -(-total // per_page),
        },
    }), 200


# ── POST /api/v1/detections ───────────────────────────────────────────────────
@detections_bp.post('/detections')
def ingest_detection():
    """
    Accept a detection payload from the detector worker and store it.

    Body (JSON)
    ───────────
    {
        "crowd_count": 7,
        "fps": 28.4,
        "camera_id": "cam-0",
        "boxes": [
            {"x1":10,"y1":20,"x2":100,"y2":200,"confidence":0.87},
            ...
        ]
    }
    """
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({'error': 'Invalid or missing JSON body.'}), 400

    errors = validate_detection_payload(payload)
    if errors:
        return jsonify({'error': 'Validation failed.', 'details': errors}), 422

    doc = {
        'timestamp'  : datetime.now(timezone.utc),
        'crowd_count': int(payload['crowd_count']),
        'fps'        : float(payload.get('fps', 0.0)),
        'camera_id'  : str(payload.get('camera_id', 'cam-0')),
        'boxes'      : payload.get('boxes', []),
    }

    result = mongo.db.detections.insert_one(doc)

    # Auto-generate alert if threshold exceeded
    threshold = current_app.config['ALERT_THRESHOLD']
    if doc['crowd_count'] > threshold:
        severity = 'CRITICAL' if doc['crowd_count'] >= threshold * 2 else 'WARNING'
        mongo.db.alerts.insert_one({
            'timestamp'   : doc['timestamp'],
            'crowd_count' : doc['crowd_count'],
            'threshold'   : threshold,
            'severity'    : severity,
            'acknowledged': False,
            'message'     : f"{severity}: {doc['crowd_count']} persons (threshold {threshold})",
        })

    return jsonify({'id': str(result.inserted_id), 'status': 'stored'}), 201


# ── GET /api/v1/detections/<id> ───────────────────────────────────────────────
@detections_bp.get('/detections/<doc_id>')
def get_detection(doc_id: str):
    try:
        oid = ObjectId(doc_id)
    except Exception:
        return jsonify({'error': 'Invalid document ID.'}), 400

    doc = mongo.db.detections.find_one({'_id': oid})
    if not doc:
        return jsonify({'error': 'Not found.'}), 404
    return jsonify(serialize_detection(doc)), 200
