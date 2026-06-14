# =============================================================================
# routes/health.py — GET /health
# =============================================================================
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify

from ..extensions import mongo

health_bp = Blueprint('health', __name__)


@health_bp.get('/health')
def health():
    """
    Liveness + readiness probe.

    Returns 200 when the API and MongoDB are reachable, 503 otherwise.
    Used by Docker healthcheck and load-balancer probes.
    """
    db_ok = False
    try:
        # Cheap ping — does not require auth if connection is established
        mongo.db.command('ping')
        db_ok = True
    except Exception as exc:
        current_app.logger.error('MongoDB ping failed: %s', exc)

    status_code = 200 if db_ok else 503
    return jsonify({
        'status'    : 'ok' if db_ok else 'degraded',
        'service'   : 'crowd-detection-api',
        'version'   : '1.0.0',
        'timestamp' : datetime.now(timezone.utc).isoformat(),
        'mongodb'   : 'connected' if db_ok else 'unreachable',
    }), status_code
