# =============================================================================
# ML Service — Application Factory
# =============================================================================
from flask import Flask
from .routes import ml_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(ml_bp)
    return app
