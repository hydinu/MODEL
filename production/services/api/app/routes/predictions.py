# =============================================================================
# routes/predictions.py — ML forecast endpoints
# =============================================================================
"""
Endpoints
─────────
POST /api/v1/predict          Trigger a new forecast (proxies to ML service)
GET  /api/v1/predict/latest   Latest stored prediction
GET  /api/v1/predict/history  Paginated prediction history
"""
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request
from bson import ObjectId
import requests as http

from ..extensions import mongo

predictions_bp = Blueprint('predictions', __name__)


@predictions_bp.post('/predict')
def trigger_prediction():
    """
    Fetch the last N crowd-count records from MongoDB, ship them to the
    ML microservice, store the returned forecast, and return it.

    Body (JSON, optional)
    ─────────────────────
    { "model": "lstm" | "random_forest",  "seq_len": 20 }
    """
    body     = request.get_json(silent=True) or {}
    model    = body.get('model', 'lstm')
    seq_len  = int(body.get('seq_len', 20))

    if model not in ('lstm', 'random_forest'):
        return jsonify({'error': "model must be 'lstm' or 'random_forest'"}), 400

    # Pull recent history from MongoDB
    docs = list(
        mongo.db.detections
        .find({}, {'timestamp': 1, 'crowd_count': 1, '_id': 0})
        .sort('timestamp', -1)
        .limit(seq_len + 120)      # 120 = max horizon steps
    )
    if len(docs) < seq_len:
        return jsonify({
            'error': f'Need at least {seq_len} detection records. Have {len(docs)}.'
        }), 422

    docs.reverse()   # chronological order
    series = [
        {'timestamp': d['timestamp'].isoformat(), 'crowd_count': d['crowd_count']}
        for d in docs
    ]

    # Forward to ML microservice
    ml_url = current_app.config['ML_SERVICE_URL']
    try:
        resp = http.post(
            f'{ml_url}/predict',
            json={'series': series, 'model': model},
            timeout=60,
        )
        resp.raise_for_status()
        forecast = resp.json()
    except http.exceptions.RequestException as exc:
        current_app.logger.error('ML service error: %s', exc)
        return jsonify({'error': 'ML service unavailable.', 'detail': str(exc)}), 502

    # Persist the forecast
    record = {
        'created_at': datetime.now(timezone.utc),
        'model'     : model,
        'horizons'  : forecast.get('horizons', []),
        'metrics'   : forecast.get('metrics', {}),
    }
    result = mongo.db.predictions.insert_one(record)

    return jsonify({
        'id'      : str(result.inserted_id),
        'model'   : model,
        'horizons': forecast.get('horizons', []),
        'metrics' : forecast.get('metrics', {}),
    }), 201


@predictions_bp.get('/predict/latest')
def latest_prediction():
    doc = mongo.db.predictions.find_one({}, sort=[('created_at', -1)])
    if not doc:
        return jsonify({'message': 'No predictions yet.'}), 204
    doc['_id'] = str(doc['_id'])
    if 'created_at' in doc:
        doc['created_at'] = doc['created_at'].isoformat()
    return jsonify(doc), 200


@predictions_bp.get('/predict/history')
def prediction_history():
    cfg      = current_app.config
    page     = max(int(request.args.get('page', 1)), 1)
    per_page = min(int(request.args.get('per_page', 20)), 200)
    model    = request.args.get('model')

    query = {}
    if model:
        query['model'] = model

    total = mongo.db.predictions.count_documents(query)
    docs  = list(
        mongo.db.predictions
        .find(query)
        .sort('created_at', -1)
        .skip((page - 1) * per_page)
        .limit(per_page)
    )
    for d in docs:
        d['_id'] = str(d['_id'])
        if 'created_at' in d:
            d['created_at'] = d['created_at'].isoformat()

    return jsonify({
        'data'      : docs,
        'pagination': {'page': page, 'per_page': per_page, 'total': total},
    }), 200
