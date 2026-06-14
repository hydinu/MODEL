# =============================================================================
# models/alert.py — Alert document serialiser
# =============================================================================


def serialize_alert(doc: dict) -> dict:
    return {
        'id'             : str(doc['_id']),
        'timestamp'      : doc['timestamp'].isoformat() if doc.get('timestamp') else None,
        'crowd_count'    : doc.get('crowd_count', 0),
        'threshold'      : doc.get('threshold', 0),
        'severity'       : doc.get('severity', 'WARNING'),
        'acknowledged'   : doc.get('acknowledged', False),
        'acknowledged_at': doc['acknowledged_at'].isoformat()
                           if doc.get('acknowledged_at') else None,
        'message'        : doc.get('message', ''),
    }
