# =============================================================================
# app/__init__.py — Flask Application Factory
# =============================================================================
from flask import Flask
from flask_cors import CORS

from .config import Config
from .extensions import mongo
from .middleware.error_handler import register_error_handlers
from .routes import register_blueprints


def create_app(config: type = Config) -> Flask:
    """
    Flask application factory.

    Using the factory pattern allows:
    - Multiple app instances (e.g. for testing)
    - Deferred initialisation of extensions
    - Clean separation of configuration from application logic
    """
    app = Flask(__name__)
    app.config.from_object(config)

    # ── Extensions ────────────────────────────────────────────────────────────
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    mongo.init_app(app)

    # ── Middleware ────────────────────────────────────────────────────────────
    register_error_handlers(app)

    # ── Blueprints ────────────────────────────────────────────────────────────
    register_blueprints(app)

    return app
