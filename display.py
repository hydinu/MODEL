# =============================================================================
# display.py — Overlay drawing: bounding boxes, labels, HUD (FPS + count)
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
from typing import List

from detector import Detection
from config import (
    BOX_COLOR, BOX_THICKNESS,
    LABEL_FONT_SCALE, LABEL_THICKNESS, LABEL_COLOR, LABEL_BG_COLOR,
    HUD_FONT_SCALE, HUD_THICKNESS,
    FPS_COLOR, COUNT_COLOR, TITLE_COLOR,
    HUD_BG_ALPHA,
    WINDOW_TITLE,
)

# OpenCV font used throughout
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Internal helpers ─────────────────────────────────────────────────────────

def _put_text_with_bg(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    thickness: int,
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    padding: int = 4,
) -> None:
    """Draw *text* at *origin* with a filled background rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x, y = origin
    # Background rect
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), _FONT, font_scale, text_color, thickness, cv2.LINE_AA)


def _blend_hud_background(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    alpha: float,
    color: tuple[int, int, int] = (20, 20, 20),
) -> None:
    """Alpha-blend a dark rectangle onto *frame* for the HUD panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ── Public API ───────────────────────────────────────────────────────────────

class FrameRenderer:
    """
    Stateless renderer: call :meth:`render` every frame to draw all
    visual elements onto the provided numpy array **in-place**.
    """

    def render(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float,
    ) -> np.ndarray:
        """
        Draw bounding boxes, per-box labels, and the HUD on *frame*.

        Parameters
        ----------
        frame      : BGR frame (modified in-place).
        detections : List of :class:`Detection` objects for this frame.
        fps        : Calculated frames-per-second to display.

        Returns
        -------
        The annotated *frame* (same array, for convenience).
        """
        self._draw_boxes(frame, detections)
        self._draw_hud(frame, detections, fps)
        return frame

    # ── Private drawing methods ──────────────────────────────────────────────

    @staticmethod
    def _draw_boxes(frame: np.ndarray, detections: List[Detection]) -> None:
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

            # Confidence label just above the box
            _put_text_with_bg(
                frame,
                text=det.label,
                origin=(x1, max(y1 - 6, 16)),   # clamp so label stays on-screen
                font_scale=LABEL_FONT_SCALE,
                thickness=LABEL_THICKNESS,
                text_color=LABEL_COLOR,
                bg_color=LABEL_BG_COLOR,
            )

    @staticmethod
    def _draw_hud(frame: np.ndarray, detections: List[Detection], fps: float) -> None:
        """Draw a semi-transparent HUD panel in the top-left corner."""
        h, w = frame.shape[:2]

        # ── Panel dimensions ────────────────────────────────────────────────
        panel_w = 310
        panel_h = 110
        margin  = 12
        _blend_hud_background(frame, margin, margin, panel_w, panel_h, HUD_BG_ALPHA)

        # Thin accent border
        cv2.rectangle(
            frame,
            (margin, margin),
            (margin + panel_w, margin + panel_h),
            (80, 80, 80),
            1,
        )

        # ── Text lines ──────────────────────────────────────────────────────
        line_x = margin + 14
        line1_y = margin + 32

        # Title
        cv2.putText(
            frame,
            "Crowd Detection  [YOLOv8]",
            (line_x, line1_y),
            _FONT, HUD_FONT_SCALE * 0.8,
            TITLE_COLOR, 1, cv2.LINE_AA,
        )

        # Person count
        count = len(detections)
        count_text = f"Persons : {count}"
        cv2.putText(
            frame,
            count_text,
            (line_x, line1_y + 34),
            _FONT, HUD_FONT_SCALE,
            COUNT_COLOR, HUD_THICKNESS, cv2.LINE_AA,
        )

        # FPS
        fps_text = f"FPS     : {fps:05.1f}"
        cv2.putText(
            frame,
            fps_text,
            (line_x, line1_y + 68),
            _FONT, HUD_FONT_SCALE,
            FPS_COLOR, HUD_THICKNESS, cv2.LINE_AA,
        )

        # ── Density indicator bar ────────────────────────────────────────────
        # Shows crowd density relative to a reference of 20 persons = full bar
        max_ref   = 20
        bar_x     = margin + 14
        bar_y     = margin + panel_h + 6
        bar_w_max = panel_w - 28
        bar_h     = 6
        fill_ratio = min(count / max_ref, 1.0)
        fill_w    = int(bar_w_max * fill_ratio)

        # Choose color: green → orange → red
        if fill_ratio < 0.5:
            bar_color = (0, 200, 100)
        elif fill_ratio < 0.8:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 60, 220)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + bar_h), (60, 60, 60), cv2.FILLED)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, cv2.FILLED)

        # ── Quit hint in the bottom-right ────────────────────────────────────
        hint = "Press  Q  to quit"
        (tw, _), _ = cv2.getTextSize(hint, _FONT, 0.45, 1)
        cv2.putText(
            frame,
            hint,
            (w - tw - 14, h - 10),
            _FONT, 0.45,
            (160, 160, 160), 1, cv2.LINE_AA,
        )
