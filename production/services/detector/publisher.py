# =============================================================================
# publisher.py — Publishes detection records to MongoDB
# =============================================================================
import time
from datetime import datetime, timezone
from typing import List

import pymongo

from config import DetectorConfig as Cfg
from detector import Detection


class DetectionPublisher:
    """
    Batches detection results and writes to MongoDB at PUBLISH_INTERVAL.

    The publisher decouples the frame-rate (30 fps) from the database
    write rate (every 30 s by default), preventing MongoDB overload.
    """

    def __init__(self):
        client      = pymongo.MongoClient(Cfg.MONGO_URI)
        self._db    = client[Cfg.MONGO_DB]
        self._last  = time.monotonic() - Cfg.PUBLISH_INTERVAL  # write immediately on start
        print(f'[Publisher] Connected → {Cfg.MONGO_DB}  interval={Cfg.PUBLISH_INTERVAL}s')

    def tick(
        self,
        detections : List[Detection],
        fps        : float,
    ) -> bool:
        """
        Call every frame.  Returns True if a document was written this call.

        Writes a detection document when the publish interval elapses.
        Auto-inserts an alert document if crowd_count > threshold.
        """
        now = time.monotonic()
        if now - self._last < Cfg.PUBLISH_INTERVAL:
            return False

        self._last   = now
        crowd_count  = len(detections)
        ts           = datetime.now(timezone.utc)

        # ── Detection record ──────────────────────────────────────────────────
        doc = {
            'timestamp'  : ts,
            'crowd_count': crowd_count,
            'fps'        : round(fps, 2),
            'camera_id'  : Cfg.CAMERA_ID,
            'boxes'      : [d.to_dict() for d in detections],
        }
        self._db.detections.insert_one(doc)

        # ── Alert record (if threshold exceeded) ──────────────────────────────
        if crowd_count > Cfg.ALERT_THRESHOLD:
            severity = (
                'CRITICAL' if crowd_count >= Cfg.ALERT_THRESHOLD * 2 else 'WARNING'
            )
            self._db.alerts.insert_one({
                'timestamp'   : ts,
                'crowd_count' : crowd_count,
                'threshold'   : Cfg.ALERT_THRESHOLD,
                'severity'    : severity,
                'acknowledged': False,
                'message'     : (
                    f'{severity}: {crowd_count} persons '
                    f'(threshold {Cfg.ALERT_THRESHOLD})'
                ),
            })
            print(f'[Publisher] ⚠  Alert → {severity}  count={crowd_count}')

        print(
            f'[Publisher] Stored → count={crowd_count}  '
            f'fps={fps:.1f}  boxes={len(detections)}'
        )
        return True
