# =============================================================================
# app/config.py — Flask configuration
# =============================================================================
import os


class Config:
    """Base configuration — reads everything from environment variables."""

    # Flask
    SECRET_KEY    = os.environ.get('SECRET_KEY', 'dev-secret-not-for-prod')
    FLASK_ENV     = os.environ.get('FLASK_ENV', 'production')
    DEBUG         = FLASK_ENV == 'development'
    TESTING       = False

    # MongoDB
    MONGO_URI     = os.environ.get(
        'MONGO_URI',
        'mongodb://admin:changeme@localhost:27017/crowd_detection?authSource=admin'
    )
    MONGO_DB      = os.environ.get('MONGO_DB', 'crowd_detection')

    # Alert
    ALERT_THRESHOLD = int(os.environ.get('ALERT_THRESHOLD', 10))

    # ML Service
    ML_SERVICE_URL  = os.environ.get('ML_SERVICE_URL', 'http://ml:5001')

    # Pagination
    DEFAULT_PAGE_SIZE = 50
    MAX_PAGE_SIZE     = 500


class DevelopmentConfig(Config):
    DEBUG  = True
    FLASK_ENV = 'development'


class TestingConfig(Config):
    TESTING  = True
    MONGO_DB = 'crowd_detection_test'
