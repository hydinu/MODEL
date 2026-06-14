# =============================================================================
# routes/analytics.py — Aggregated crowd analytics
# =============================================================================
"""
Endpoints
─────────
GET /api/v1/analytics/summary   Overall stats (avg, max, min, std)
GET /api/v1/analytics/hourly    Average crowd count per hour of the day
GET /api/v1/analytics/trend     Time-bucketed crowd count trend
"""
from datetime import datetime, timezone, timedelta

from flask import Blueprint, jsonify, request

from ..extensions import mongo

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.get('/analytics/summary')
def summary():
    """Aggregate stats across all stored detections."""
    pipeline = [
        {'$group': {
            '_id'  : None,
            'avg'  : {'$avg': '$crowd_count'},
            'max'  : {'$max': '$crowd_count'},
            'min'  : {'$min': '$crowd_count'},
            'std'  : {'$stdDevPop': '$crowd_count'},
            'total': {'$sum': 1},
            'first': {'$min': '$timestamp'},
            'last' : {'$max': '$timestamp'},
        }}
    ]
    rows = list(mongo.db.detections.aggregate(pipeline))
    if not rows:
        return jsonify({'message': 'No data yet.'}), 204
    r = rows[0]
    return jsonify({
        'total_records'   : r['total'],
        'average_crowd'   : round(r['avg'], 2),
        'max_crowd'       : r['max'],
        'min_crowd'       : r['min'],
        'std_dev'         : round(r['std'], 2),
        'first_detection' : r['first'].isoformat() if r['first'] else None,
        'last_detection'  : r['last'].isoformat()  if r['last']  else None,
    }), 200


@analytics_bp.get('/analytics/hourly')
def hourly():
    """Average crowd count grouped by hour-of-day (0–23) across all history."""
    pipeline = [
        {'$group': {
            '_id'  : {'$hour': '$timestamp'},
            'avg'  : {'$avg': '$crowd_count'},
            'count': {'$sum': 1},
        }},
        {'$sort': {'_id': 1}},
        {'$project': {
            '_id'        : 0,
            'hour'       : '$_id',
            'avg_crowd'  : {'$round': ['$avg', 2]},
            'sample_count': '$count',
        }}
    ]
    rows = list(mongo.db.detections.aggregate(pipeline))
    return jsonify({'data': rows}), 200


@analytics_bp.get('/analytics/trend')
def trend():
    """
    Time-series crowd counts bucketed by interval.

    Query params
    ────────────
    hours    : int  look-back window in hours (default 24)
    bucket   : str  'minute' | 'hour' | 'day'  (default 'hour')
    """
    hours  = int(request.args.get('hours', 24))
    bucket = request.args.get('bucket', 'hour')
    if bucket not in ('minute', 'hour', 'day'):
        return jsonify({'error': "bucket must be 'minute', 'hour', or 'day'"}), 400

    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    date_trunc = {
        'minute': {'$dateTrunc': {'date': '$timestamp', 'unit': 'minute'}},
        'hour'  : {'$dateTrunc': {'date': '$timestamp', 'unit': 'hour'}},
        'day'   : {'$dateTrunc': {'date': '$timestamp', 'unit': 'day'}},
    }[bucket]

    pipeline = [
        {'$match': {'timestamp': {'$gte': since}}},
        {'$group': {
            '_id'      : date_trunc,
            'avg_crowd': {'$avg': '$crowd_count'},
            'max_crowd': {'$max': '$crowd_count'},
            'samples'  : {'$sum': 1},
        }},
        {'$sort': {'_id': 1}},
        {'$project': {
            '_id'      : 0,
            'timestamp': {'$dateToString': {'format': '%Y-%m-%dT%H:%M:%SZ', 'date': '$_id'}},
            'avg_crowd': {'$round': ['$avg_crowd', 1]},
            'max_crowd': '$max_crowd',
            'samples'  : '$samples',
        }}
    ]
    rows = list(mongo.db.detections.aggregate(pipeline))
    return jsonify({'data': rows, 'bucket': bucket, 'hours': hours}), 200
