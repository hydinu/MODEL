// =============================================================================
// mongo/init.js — Database initialisation script
// Runs automatically on first container start via docker-entrypoint-initdb.d
// =============================================================================

db = db.getSiblingDB(process.env.MONGO_INITDB_DATABASE || 'crowd_detection');

// ── Collections with validation schemas ──────────────────────────────────────

db.createCollection('detections', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['timestamp', 'crowd_count'],
      properties: {
        timestamp:   { bsonType: 'date',   description: 'UTC detection time' },
        crowd_count: { bsonType: 'int',    minimum: 0 },
        fps:         { bsonType: 'double', minimum: 0 },
        boxes: {
          bsonType: 'array',
          items: {
            bsonType: 'object',
            required: ['x1','y1','x2','y2','confidence'],
            properties: {
              x1: { bsonType: 'int' }, y1: { bsonType: 'int' },
              x2: { bsonType: 'int' }, y2: { bsonType: 'int' },
              confidence: { bsonType: 'double' }
            }
          }
        },
        camera_id: { bsonType: 'string' }
      }
    }
  }
});

db.createCollection('alerts', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['timestamp', 'crowd_count', 'threshold', 'severity'],
      properties: {
        timestamp:    { bsonType: 'date' },
        crowd_count:  { bsonType: 'int',    minimum: 0 },
        threshold:    { bsonType: 'int',    minimum: 0 },
        severity:     { enum: ['WARNING', 'CRITICAL'] },
        acknowledged: { bsonType: 'bool' },
        message:      { bsonType: 'string' }
      }
    }
  }
});

db.createCollection('predictions', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['created_at', 'model', 'horizons'],
      properties: {
        created_at: { bsonType: 'date' },
        model:      { enum: ['lstm', 'random_forest'] },
        horizons: {
          bsonType: 'array',
          items: {
            bsonType: 'object',
            required: ['label', 'predicted_at', 'crowd_count'],
            properties: {
              label:       { bsonType: 'string' },
              predicted_at:{ bsonType: 'date'   },
              crowd_count: { bsonType: 'double'  }
            }
          }
        },
        metrics: { bsonType: 'object' }
      }
    }
  }
});

// ── Indexes ──────────────────────────────────────────────────────────────────

db.detections.createIndex({ timestamp: -1 });
db.detections.createIndex({ crowd_count: 1 });
db.detections.createIndex({ camera_id: 1, timestamp: -1 });

db.alerts.createIndex({ timestamp: -1 });
db.alerts.createIndex({ severity: 1, acknowledged: 1 });

db.predictions.createIndex({ created_at: -1 });
db.predictions.createIndex({ model: 1, created_at: -1 });

print('✅  crowd_detection database initialised with collections + indexes.');
