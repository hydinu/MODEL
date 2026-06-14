# =============================================================================
# routes.py — ML microservice endpoints
# =============================================================================
"""
Endpoints
─────────
GET  /health         Liveness probe
POST /predict        Run forecast on supplied time-series
GET  /models         List available models and their status
"""
from datetime import datetime, timezone, timedelta

from flask import Blueprint, jsonify, request

from .forecaster import LSTMForecaster, RFForecaster

ml_bp = Blueprint('ml', __name__)

# Lazy-loaded model instances (initialised on first request)
_lstm = None
_rf   = None


def _get_lstm() -> LSTMForecaster:
    global _lstm
    if _lstm is None:
        _lstm = LSTMForecaster()
    return _lstm


def _get_rf() -> RFForecaster:
    global _rf
    if _rf is None:
        _rf = RFForecaster()
    return _rf


@ml_bp.get('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'crowd-ml'}), 200


@ml_bp.get('/models')
def list_models():
    return jsonify({
        'available': [
            {'name': 'lstm',          'description': 'Stacked LSTM (TensorFlow)', 'horizons': ['15min','30min','1hour']},
            {'name': 'random_forest', 'description': 'Random Forest (scikit-learn)', 'horizons': ['15min']},
        ]
    }), 200


@ml_bp.post('/predict')
def predict():
    """
    Accept a time-series payload and return multi-horizon forecasts.

    Request body
    ────────────
    {
        "model"  : "lstm" | "random_forest",
        "series" : [{"timestamp": "ISO", "crowd_count": 5}, ...]
    }

    Response
    ────────
    {
        "model"   : "lstm",
        "horizons": [
            {"label": "15 min", "predicted_at": "ISO", "crowd_count": 6.2},
            {"label": "30 min", "predicted_at": "ISO", "crowd_count": 7.8},
            {"label": "1 hour", "predicted_at": "ISO", "crowd_count": 5.1}
        ],
        "metrics": {}
    }
    """
    body   = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({'error': 'Missing JSON body.'}), 400

    model_name = body.get('model', 'lstm')
    series     = body.get('series', [])

    if not series:
        return jsonify({'error': '"series" must be a non-empty list.'}), 422
    if len(series) < 20:
        return jsonify({'error': f'Need at least 20 data points; got {len(series)}.'}), 422

    try:
        if model_name == 'lstm':
            result = _get_lstm().predict(series)
        elif model_name == 'random_forest':
            result = _get_rf().predict(series)
        else:
            return jsonify({'error': f'Unknown model: {model_name}'}), 400
    except Exception as exc:
        return jsonify({'error': 'Prediction failed.', 'detail': str(exc)}), 500

    return jsonify({
        'model'   : model_name,
        'horizons': result['horizons'],
        'metrics' : result.get('metrics', {}),
    }), 200
