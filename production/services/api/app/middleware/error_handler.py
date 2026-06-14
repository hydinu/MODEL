# =============================================================================
# middleware/error_handler.py — Centralised JSON error responses
# =============================================================================
from flask import Flask, jsonify


def register_error_handlers(app: Flask) -> None:
    """Register JSON-formatted error responses for standard HTTP codes."""

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({'error': 'Bad request.', 'detail': str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Resource not found.'}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({'error': 'Method not allowed.'}), 405

    @app.errorhandler(422)
    def unprocessable(e):
        return jsonify({'error': 'Unprocessable entity.', 'detail': str(e)}), 422

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'error': 'Internal server error.'}), 500

    @app.errorhandler(502)
    def bad_gateway(e):
        return jsonify({'error': 'Bad gateway — upstream service failure.'}), 502
