# =============================================================================
# heatmap.py — Real-time crowd density heatmap accumulator
# =============================================================================
"""
How it works
────────────
Every frame, for each detected person, a 2-D Gaussian "heat blob" is stamped
at their ground position (bottom-centre of the bounding box — where their feet
are).  These blobs accumulate in a float32 heat buffer the same size as the
camera frame.

Between frames the buffer is multiplied by a decay factor (0 < decay < 1) so
old heat fades away and the map stays responsive to crowd movement.

The heat buffer is then:
  1. Normalised to [0, 255]
  2. Colourised with COLORMAP_JET  (blue → cyan → green → yellow → red)
  3. Alpha-blended over the live BGR frame

Mathematical note — 2-D Gaussian kernel
────────────────────────────────────────
For a person standing at pixel (cx, cy):

    H(x, y) += A · exp( -( (x-cx)²/2σx² + (y-cy)²/2σy² ) )

where
    A   = amplitude (weight per person, default 1.0)
    σx  = horizontal spread  (proportional to bounding-box width / 2)
    σy  = vertical spread    (proportional to bounding-box height / 2)

Using the box size as the spread means that a person who fills more of the
frame (i.e. is physically closer to the camera) contributes a wider blob,
giving a natural sense of proximity weight.

Usage
─────
    hm = HeatmapAccumulator(frame_h=720, frame_w=1280)

    # Called once per frame:
    hm.update(detections)            # stamp blobs + apply decay
    coloured = hm.render(frame)      # returns a new blended BGR image
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from config import (
    HEATMAP_ALPHA,
    HEATMAP_DECAY,
    HEATMAP_SIGMA_SCALE,
    HEATMAP_AMPLITUDE,
    HEATMAP_BLUR_KERNEL,
    HEATMAP_COLORMAP,
    HEATMAP_SHOW_LEGEND,
)
from detector import Detection


class HeatmapAccumulator:
    """
    Maintains a floating-point heat buffer and exposes two public methods:

    update(detections) — decay old heat and stamp new Gaussian blobs.
    render(frame)      — colourise buffer and alpha-blend onto frame copy.

    Parameters
    ──────────
    frame_h, frame_w  : Frame dimensions (must match the live camera output).
    alpha             : Blend opacity of the coloured heatmap layer.
    decay             : Per-frame multiplicative decay [0, 1).
                        0.95 → heat halves every ~14 frames at 30 FPS.
    sigma_scale       : Gaussian σ expressed as a fraction of half-bbox size.
    amplitude         : Peak heat value stamped per person.
    blur_kernel       : Size of final Gaussian blur for smooth edges (odd int).
    colormap          : OpenCV colormap constant applied to the normalised map.
    """

    def __init__(
        self,
        frame_h: int,
        frame_w: int,
        alpha: float         = HEATMAP_ALPHA,
        decay: float         = HEATMAP_DECAY,
        sigma_scale: float   = HEATMAP_SIGMA_SCALE,
        amplitude: float     = HEATMAP_AMPLITUDE,
        blur_kernel: int     = HEATMAP_BLUR_KERNEL,
        colormap: int        = HEATMAP_COLORMAP,
    ) -> None:
        self._h          = frame_h
        self._w          = frame_w
        self._alpha      = alpha
        self._decay      = decay
        self._sigma_scale= sigma_scale
        self._amplitude  = amplitude
        self._blur_kernel= blur_kernel
        self._colormap   = colormap

        # Accumulated heat buffer — float32, same resolution as the camera
        self._heat: np.ndarray = np.zeros((frame_h, frame_w), dtype=np.float32)

        # Pre-build meshgrid coordinate arrays (only once, reused every frame)
        xs = np.arange(frame_w, dtype=np.float32)
        ys = np.arange(frame_h, dtype=np.float32)
        self._XX, self._YY = np.meshgrid(xs, ys)   # shape: (H, W)

    # ─── Public API ──────────────────────────────────────────────────────────

    def update(self, detections: List[Detection]) -> None:
        """
        1. Decay existing heat:  H ← H × decay
        2. Stamp a Gaussian blob for every detected person.

        Ground anchor = bottom-centre of the bounding box, so the hottest
        pixel aligns with where the person stands on the floor plane.
        """
        # Temporal decay — old heat fades
        self._heat *= self._decay

        for det in detections:
            # Ground position: bottom-centre of the bounding box
            cx = (det.x1 + det.x2) / 2.0          # horizontal centre
            cy = float(det.y2)                      # bottom edge (feet)

            # Adaptive σ proportional to box size → closer persons = wider blob
            sigma_x = max((det.x2 - det.x1) / 2.0 * self._sigma_scale, 10.0)
            sigma_y = max((det.y2 - det.y1) / 2.0 * self._sigma_scale, 10.0)

            # 2-D Gaussian: H(x,y) += A · exp(-(dx²/2σx² + dy²/2σy²))
            dx2 = (self._XX - cx) ** 2
            dy2 = (self._YY - cy) ** 2
            blob = self._amplitude * np.exp(
                -(dx2 / (2.0 * sigma_x ** 2) + dy2 / (2.0 * sigma_y ** 2))
            )
            self._heat += blob

    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Colourise the heat buffer and blend it over *frame*.

        Steps:
          1. Optional Gaussian blur  → smooth jagged contours
          2. Normalise to [0, 255]   → uint8
          3. Apply OpenCV colormap   → BGR colour image
          4. Mask zero-heat pixels   → transparent background
          5. Alpha-blend over frame  → output image

        Returns a **new** BGR array (frame is not modified in-place).
        """
        # ── 1. Blur for smooth contours ──────────────────────────────────────
        k = self._blur_kernel | 1          # ensure odd
        blurred = cv2.GaussianBlur(self._heat, (k, k), sigmaX=0, sigmaY=0)

        # ── 2. Normalise ──────────────────────────────────────────────────────
        heat_max = blurred.max()
        if heat_max < 1e-6:
            return frame.copy()            # nothing to show yet

        normalised = np.clip(blurred / heat_max * 255, 0, 255).astype(np.uint8)

        # ── 3. Colourise ──────────────────────────────────────────────────────
        coloured = cv2.applyColorMap(normalised, self._colormap)

        # ── 4. Mask: pixels below a small threshold are treated as background
        #    so we don't tint the entire frame blue (the cold end of JET)
        threshold = 8                          # uint8: ignore near-zero heat
        mask = (normalised > threshold).astype(np.float32)   # 0 or 1

        # Expand mask to 3 channels
        mask3 = np.stack([mask, mask, mask], axis=2)

        # ── 5. Alpha blend: out = α·coloured + (1-α)·frame, but only on mask
        #    Outside the mask we keep the original frame pixel unchanged.
        frame_f   = frame.astype(np.float32)
        coloured_f= coloured.astype(np.float32)

        blended_f = (
            self._alpha * coloured_f * mask3
            + frame_f * (1.0 - self._alpha * mask3)
        )
        return np.clip(blended_f, 0, 255).astype(np.uint8)

    def resize(self, new_h: int, new_w: int) -> None:
        """Resize the heat buffer if the camera resolution changes at runtime."""
        if new_h != self._h or new_w != self._w:
            self._heat = cv2.resize(self._heat, (new_w, new_h))
            self._h, self._w = new_h, new_w
            xs = np.arange(new_w, dtype=np.float32)
            ys = np.arange(new_h, dtype=np.float32)
            self._XX, self._YY = np.meshgrid(xs, ys)

    def reset(self) -> None:
        """Clear the heat buffer (e.g., on scene cut or manual reset)."""
        self._heat[:] = 0.0

    @property
    def raw(self) -> np.ndarray:
        """Read-only view of the float32 heat buffer."""
        return self._heat


# ─── Optional: colourbar legend ──────────────────────────────────────────────

def draw_colorbar_legend(
    frame: np.ndarray,
    colormap: int = HEATMAP_COLORMAP,
    width: int = 20,
    height: int = 140,
    margin: int = 14,
) -> None:
    """
    Draw a vertical colourbar legend in the bottom-right corner of *frame*
    (in-place) labelled Low → High.

    The gradient is built from a 256-entry ramp passed through the same
    colormap used by the heatmap, giving exact visual correspondence.
    """
    if not HEATMAP_SHOW_LEGEND:
        return

    h, w = frame.shape[:2]

    # Build gradient strip: shape (256, 1) → colourise → resize
    ramp   = np.arange(255, -1, -1, dtype=np.uint8).reshape(256, 1)
    strip  = cv2.applyColorMap(ramp, colormap)              # (256,1,3)
    strip  = cv2.resize(strip, (width, height))             # (height, width, 3)

    # Position: bottom-right corner
    x0 = w - margin - width
    y0 = h - margin - height

    # Blend legend background
    overlay = frame.copy()
    pad = 4
    cv2.rectangle(
        overlay,
        (x0 - pad - 36, y0 - pad - 12),
        (x0 + width + pad, y0 + height + pad),
        (20, 20, 20), cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Paste colourbar
    frame[y0:y0 + height, x0:x0 + width] = strip

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "High", (x0 - 34, y0 + 8),
                font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "Low",  (x0 - 30, y0 + height),
                font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "Density", (x0 - 36, y0 - 4),
                font, 0.32, (160, 160, 160), 1, cv2.LINE_AA)
