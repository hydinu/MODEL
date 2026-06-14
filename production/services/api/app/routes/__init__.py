# =============================================================================
# app/routes/__init__.py — Blueprint registration
# =============================================================================
from flask import Flask

from .health      import health_bp
from .detections  import detections_bp
from .alerts      import alerts_bp
from .analytics   import analytics_bp
from .predictions import predictions_bp


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(health_bp)
    app.register_blueprint(detections_bp,  url_prefix='/api/v1')
    app.register_blueprint(alerts_bp,      url_prefix='/api/v1')
    app.register_blueprint(analytics_bp,   url_prefix='/api/v1')
    app.register_blueprint(predictions_bp, url_prefix='/api/v1')
