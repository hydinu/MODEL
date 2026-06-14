# =============================================================================
# alerts.py — Crowd-count alert system: screen overlay + SQLite persistence
# =============================================================================
"""
Two public classes
──────────────────
AlertDatabase
    Thin wrapper around a SQLite database.  Creates the table on first use.
    Provides insert() and query methods.  Safe to use as a context manager.

AlertSystem
    Stateful monitor called once per frame.  Compares crowd_count against
    ALERT_THRESHOLD.  Enforces a cooldown so the database is not spammed.
    Returns an AlertState on every call so main.py can drive the overlay.

One public function
───────────────────
draw_alert_overlay(frame, state)
    Draws an animated warning banner directly onto *frame* (in-place).
    The banner pulses using a sine-wave opacity and shows a blinking
    "⚠ CROWD ALERT" headline with count / threshold / timestamp.
    Remains invisible when state.active is False.
"""

from __future__ import annotations

import math
import pymongo
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from config import (
    ALERT_THRESHOLD,
    MONGO_URI,
    MONGO_DB,
    MONGO_COLLECTION,
    ALERT_COOLDOWN,
    ALERT_BLINK_HZ,
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data transfer object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlertState:
    """Snapshot returned by AlertSystem.check() each frame."""
    active       : bool  = False   # Is crowd currently over threshold?
    crowd_count  : int   = 0       # Current crowd count
    threshold    : int   = 0       # Configured threshold
    just_saved   : bool  = False   # True on the exact frame a DB row was written
    saved_at     : Optional[str] = None   # ISO timestamp of last DB save
    total_alerts : int   = 0       # All-time alert count from this session


# ─────────────────────────────────────────────────────────────────────────────
# 2. MongoDB persistence layer
# ─────────────────────────────────────────────────────────────────────────────

class AlertDatabase:
    """
    Stores alert events in a MongoDB database collection.

    Schema
    ──────
    Document:
        _id         ObjectId
        timestamp   string   (ISO-8601 local time format "%Y-%m-%d %H:%M:%S")
        crowd_count int
        threshold   int
        severity    string   ('WARNING' | 'CRITICAL')
        message     string
    """

    def __init__(
        self,
        mongo_uri: str = MONGO_URI,
        db_name: str = MONGO_DB,
        collection_name: str = MONGO_COLLECTION,
    ) -> None:
        self._uri = mongo_uri
        self._db_name = db_name
        self._collection_name = collection_name

        self._client = pymongo.MongoClient(self._uri)
        self._db = self._client[self._db_name]
        self._col = self._db[self._collection_name]

        # Ensure timestamp field index exists for performant queries
        self._col.create_index("timestamp")
        print(f"[AlertDB] Connected to MongoDB -> {self.db_info}")

    @property
    def db_info(self) -> str:
        """Friendly summary of database connection target."""
        return f"{self._uri.rstrip('/')}/{self._db_name}.{self._collection_name}"

    # ── Write ─────────────────────────────────────────────────────────────────

    def insert(self, crowd_count: int, threshold: int, camera_name: str = "Default Camera") -> str:
        """
        Insert one alert document. Returns the string representation of the inserted _id.

        Severity is 'CRITICAL' when count ≥ 2× threshold, else 'WARNING'.
        """
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        severity = "CRITICAL" if crowd_count >= threshold * 2 else "WARNING"
        excess   = crowd_count - threshold
        message  = (
            f"{severity}: {crowd_count} persons detected "
            f"({excess:+d} over threshold of {threshold}) on camera '{camera_name}'"
        )

        doc = {
            "timestamp": ts,
            "camera_name": camera_name,
            "crowd_count": crowd_count,
            "threshold": threshold,
            "severity": severity,
            "message": message,
        }

        result = self._col.insert_one(doc)
        print(f"[AlertDB] [Alert] Saved to MongoDB -> id={result.inserted_id}  {message}")
        return str(result.inserted_id)

    # ── Read ──────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of alert documents stored."""
        return self._col.count_documents({})

    def recent(self, n: int = 10) -> list[dict]:
        """Return the *n* most recent alerts as plain dicts."""
        docs = self._col.find().sort("_id", -1).limit(n)
        result = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            result.append(doc)
        return result

    def all_rows(self) -> list[dict]:
        """Return every alert document ordered oldest-first."""
        docs = self._col.find().sort("_id", 1)
        result = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            result.append(doc)
        return result

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._client:
            self._client.close()
            print(f"[AlertDB] MongoDB connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Alert logic
# ─────────────────────────────────────────────────────────────────────────────

class AlertSystem:
    """
    Called once per frame with the current crowd_count.
    Manages cooldown so alerts are not saved more than once per
    ALERT_COOLDOWN seconds.

    Parameters
    ──────────
    threshold : int   — Number of persons that triggers an alert.
    cooldown  : float — Minimum seconds between successive DB writes.
    db        : AlertDatabase — persistence layer (injected or auto-created).
    """

    def __init__(
        self,
        threshold: int            = ALERT_THRESHOLD,
        cooldown : float          = ALERT_COOLDOWN,
        db       : Optional[AlertDatabase] = None,
        camera_name: str          = "Default Camera",
    ) -> None:
        self._threshold    = threshold
        self._cooldown     = cooldown
        self._db           = db or AlertDatabase()
        self._camera_name  = camera_name
        self._last_save    = -cooldown       # allow immediate first save
        self._saved_at     : Optional[str] = None
        self._session_count: int = 0

        print(
            f"[AlertSystem] Threshold={threshold} persons  "
            f"| Cooldown={cooldown}s  "
            f"| DB={self._db.db_info}  "
            f"| Camera={camera_name}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, crowd_count: int) -> AlertState:
        """
        Evaluate the current crowd_count and (if warranted) write to the DB.

        Returns an AlertState describing the situation this frame.
        """
        now    = time.monotonic()
        active = crowd_count > self._threshold

        just_saved = False
        if active and (now - self._last_save) >= self._cooldown:
            self._db.insert(crowd_count, self._threshold, self._camera_name)
            self._last_save = now
            self._saved_at  = datetime.now().strftime("%H:%M:%S")
            self._session_count += 1
            just_saved = True

        return AlertState(
            active       = active,
            crowd_count  = crowd_count,
            threshold    = self._threshold,
            just_saved   = just_saved,
            saved_at     = self._saved_at,
            total_alerts = self._session_count,
        )

    @property
    def db(self) -> AlertDatabase:
        return self._db

    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Screen overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_alert_overlay(frame: np.ndarray, state: AlertState) -> None:
    """
    Draw a pulsing warning banner onto *frame* **in-place**.

    Layout (top of frame)
    ─────────────────────
    ┌─────────────────────────────────────────────────────────────────┐
    │  ⚠  CROWD ALERT                           SAVED TO DATABASE  ✔ │
    │      12 persons detected  │  Threshold: 10  │  Last: 14:03:52  │
    └─────────────────────────────────────────────────────────────────┘

    Visual behaviour
    ────────────────
    • Background pulsates between semi-transparent red and deep red using:
          alpha(t) = BASE + AMP · |sin(2π · BLINK_HZ · t)|
    • When state.just_saved is True a brief "✔ SAVED" flash appears.
    • The banner is invisible (returns immediately) when state.active is False.
    """
    if not state.active:
        return

    h, w = frame.shape[:2]
    t    = time.perf_counter()

    # ── Pulsing opacity ───────────────────────────────────────────────────────
    BASE_ALPHA  = 0.55
    PULSE_AMP   = 0.25
    alpha = BASE_ALPHA + PULSE_AMP * abs(math.sin(2 * math.pi * ALERT_BLINK_HZ * t))

    BANNER_H  = 72
    BANNER_Y0 = 0
    BANNER_Y1 = BANNER_H

    # ── Banner background (dark red, pulsing) ────────────────────────────────
    severity_color = (0, 0, 200) if state.crowd_count < state.threshold * 2 \
                     else (0, 0, 255)              # deeper red for CRITICAL

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, BANNER_Y0), (w, BANNER_Y1), severity_color, cv2.FILLED)

    # Accent stripe at the very top
    cv2.rectangle(frame, (0, 0), (w, 3), (50, 50, 255), cv2.FILLED)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Left edge coloured accent bar ────────────────────────────────────────
    cv2.rectangle(frame, (0, BANNER_Y0), (5, BANNER_Y1), (80, 80, 255), cv2.FILLED)

    # ── Headline: "⚠  CROWD ALERT" ───────────────────────────────────────────
    label = "!  CROWD ALERT"          # '!' stands in for ⚠ (ASCII safe)
    is_critical = state.crowd_count >= state.threshold * 2
    headline_color = (50, 50, 255) if is_critical else (50, 180, 255)

    cv2.putText(
        frame, label,
        (16, BANNER_Y0 + 30),
        _FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
    )
    # Coloured version overlaid for tinted effect
    cv2.putText(
        frame, label,
        (16, BANNER_Y0 + 30),
        _FONT, 0.85, headline_color, 1, cv2.LINE_AA,
    )

    # ── Severity badge ────────────────────────────────────────────────────────
    severity_label = "CRITICAL" if is_critical else "WARNING"
    badge_color    = (0, 0, 200)     if is_critical else (0, 100, 220)
    (bw, bh), _   = cv2.getTextSize(severity_label, _FONT, 0.45, 1)
    bx, by = 200, BANNER_Y0 + 14
    cv2.rectangle(frame, (bx - 4, by - bh - 2), (bx + bw + 4, by + 4),
                  badge_color, cv2.FILLED)
    cv2.putText(frame, severity_label, (bx, by),
                _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Detail line: count | threshold | last saved ───────────────────────────
    excess      = state.crowd_count - state.threshold
    detail_text = (
        f"{state.crowd_count} persons detected  "
        f"|  Threshold: {state.threshold}  "
        f"|  Excess: +{excess}"
    )
    cv2.putText(
        frame, detail_text,
        (16, BANNER_Y0 + 58),
        _FONT, 0.52, (220, 220, 220), 1, cv2.LINE_AA,
    )

    # ── Right side: DB-saved indicator ───────────────────────────────────────
    if state.saved_at:
        saved_text  = f"DB SAVED  {state.saved_at}  #{state.total_alerts}"
        (sw, _), _  = cv2.getTextSize(saved_text, _FONT, 0.45, 1)
        sx          = w - sw - 16

        # Green flash on the exact frame of the save, else muted
        saved_color = (100, 255, 100) if state.just_saved else (140, 200, 140)
        cv2.putText(
            frame, saved_text,
            (sx, BANNER_Y0 + 28),
            _FONT, 0.45, saved_color, 1, cv2.LINE_AA,
        )

        # Blinking dot
        dot_r = 5
        dot_c = (w - 10, BANNER_Y0 + 22)
        blink = abs(math.sin(2 * math.pi * 1.5 * t)) > 0.5
        if blink:
            cv2.circle(frame, dot_c, dot_r, (80, 255, 80), cv2.FILLED)

    # ── Alert count badge (bottom-right of banner) ───────────────────────────
    count_label = f"Alerts this session: {state.total_alerts}"
    (cw, _), _  = cv2.getTextSize(count_label, _FONT, 0.38, 1)
    cv2.putText(
        frame, count_label,
        (w - cw - 16, BANNER_Y1 - 8),
        _FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA,
    )
