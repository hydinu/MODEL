# =============================================================================
# routes/alerts.py — Alert management endpoints
# =============================================================================
"""
Endpoints
─────────
GET    /api/v1/alerts                    Paginated alert list
GET    /api/v1/alerts/<id>               Single alert
PATCH  /api/v1/alerts/<id>/acknowledge   Mark alert as acknowledged
DELETE /api/v1/alerts/<id>              Delete alert
GET    /api/v1/alerts/summary           Counts by severity + unacknowledged total
"""
from datetime import datetime, timezone

from bson import ObjectId
from flask import Blueprint, current_app, jsonify, request

from ..extensions import mongo
from ..models.alert import serialize_alert

alerts_bp = Blueprint('alerts', __name__)


@alerts_bp.get('/alerts/summary')
def alerts_summary():
    """Aggregated overview — total, by severity, unacknowledged."""
    pipeline = [
        {'$group': {
            '_id'      : {'severity': '$severity', 'acked': '$acknowledged'},
            'count'    : {'$sum': 1},
        }}
    ]
    rows = list(mongo.db.alerts.aggregate(pipeline))
    total  = mongo.db.alerts.count_documents({})
    unacked= mongo.db.alerts.count_documents({'acknowledged': False})
    by_sev = {}
    for row in rows:
        sev = row['_id']['severity']
        by_sev[sev] = by_sev.get(sev, 0) + row['count']

    return jsonify({
        'total'          : total,
        'unacknowledged' : unacked,
        'by_severity'    : by_sev,
    }), 200


@alerts_bp.get('/alerts')
def list_alerts():
    cfg      = current_app.config
    page     = max(int(request.args.get('page', 1)), 1)
    per_page = min(int(request.args.get('per_page', cfg['DEFAULT_PAGE_SIZE'])), cfg['MAX_PAGE_SIZE'])

    query = {}
    if severity := request.args.get('severity'):
        query['severity'] = severity.upper()
    if acked := request.args.get('acknowledged'):
        query['acknowledged'] = acked.lower() == 'true'

    total = mongo.db.alerts.count_documents(query)
    docs  = list(
        mongo.db.alerts
        .find(query)
        .sort('timestamp', -1)
        .skip((page - 1) * per_page)
        .limit(per_page)
    )
    return jsonify({
        'data'      : [serialize_alert(d) for d in docs],
        'pagination': {'page': page, 'per_page': per_page, 'total': total},
    }), 200


@alerts_bp.get('/alerts/<doc_id>')
def get_alert(doc_id: str):
    try:
        oid = ObjectId(doc_id)
    except Exception:
        return jsonify({'error': 'Invalid ID.'}), 400
    doc = mongo.db.alerts.find_one({'_id': oid})
    if not doc:
        return jsonify({'error': 'Not found.'}), 404
    return jsonify(serialize_alert(doc)), 200


@alerts_bp.patch('/alerts/<doc_id>/acknowledge')
def acknowledge_alert(doc_id: str):
    try:
        oid = ObjectId(doc_id)
    except Exception:
        return jsonify({'error': 'Invalid ID.'}), 400
    result = mongo.db.alerts.update_one(
        {'_id': oid},
        {'$set': {'acknowledged': True, 'acknowledged_at': datetime.now(timezone.utc)}}
    )
    if result.matched_count == 0:
        return jsonify({'error': 'Not found.'}), 404
    return jsonify({'status': 'acknowledged'}), 200


@alerts_bp.delete('/alerts/<doc_id>')
def delete_alert(doc_id: str):
    try:
        oid = ObjectId(doc_id)
    except Exception:
        return jsonify({'error': 'Invalid ID.'}), 400
    result = mongo.db.alerts.delete_one({'_id': oid})
    if result.deleted_count == 0:
        return jsonify({'error': 'Not found.'}), 404
    return jsonify({'status': 'deleted'}), 200
