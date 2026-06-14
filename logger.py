# =============================================================================
# logger.py — Periodic CSV logger for crowd-count snapshots
# =============================================================================
"""
Every LOG_INTERVAL_SECONDS seconds this module appends one row:

    timestamp (ISO-8601),crowd_count

to the CSV file defined in config.py (CSV_LOG_PATH).

Usage
-----
    from logger import CrowdLogger

    logger = CrowdLogger()          # opens / creates the CSV
    logger.tick(crowd_count=5)      # call once per frame; writes only when due
    logger.close()                  # flush & close (or use as context manager)
"""

import csv
import os
import time
from datetime import datetime

from config import CSV_LOG_PATH, CSV_LOG_INTERVAL


class CrowdLogger:
    """
    Writes (timestamp, crowd_count) rows to a CSV file at a fixed interval.

    Parameters
    ----------
    path     : str   – Destination CSV file path (default from config).
    interval : float – Seconds between successive log entries (default from config).
    """

    _HEADER = ["timestamp", "crowd_count"]

    def __init__(
        self,
        path: str = CSV_LOG_PATH,
        interval: float = CSV_LOG_INTERVAL,
    ) -> None:
        self._path     = path
        self._interval = interval
        self._last_log = time.monotonic()   # tracks when we last wrote a row

        # Create parent directories if they don't exist
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

        # Open the file in append mode so previous runs are preserved
        file_exists = os.path.isfile(path)
        self._file  = open(path, "a", newline="", encoding="utf-8")  # noqa: WPS515
        self._writer = csv.writer(self._file)

        # Write header only when creating a fresh file
        if not file_exists or os.path.getsize(path) == 0:
            self._writer.writerow(self._HEADER)
            self._file.flush()
            print(f"[Logger] Created CSV log → {os.path.abspath(path)}")
        else:
            print(f"[Logger] Appending to existing CSV log → {os.path.abspath(path)}")

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self, crowd_count: int) -> bool:
        """
        Called once per frame.  Writes a row if the log interval has elapsed.

        Parameters
        ----------
        crowd_count : Current number of detected persons.

        Returns
        -------
        True if a row was written this call, False otherwise.
        """
        now = time.monotonic()
        if now - self._last_log >= self._interval:
            self._write(crowd_count)
            self._last_log = now
            return True
        return False

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
            print(f"[Logger] CSV log closed → {os.path.abspath(self._path)}")

    # ── Context-manager support ───────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write(self, crowd_count: int) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._writer.writerow([ts, crowd_count])
        self._file.flush()                  # ensure data survives a crash
        print(f"[Logger] Logged → {ts}  |  Persons: {crowd_count}")
