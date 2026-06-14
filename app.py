# =============================================================================
# app.py — Streamlit Web Interface for Crowd Detection System (Simple UI)
# =============================================================================

import streamlit as st
import cv2
import time
import numpy as np
from camera import Camera
from detector import PersonDetector
from display import FrameRenderer
from logger import CrowdLogger
from config import (
    MODEL_PATH,
    ALERT_THRESHOLD,
    CONFIDENCE_THRESHOLD,
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crowd Detection Dashboard",
    page_icon="👥",
    layout="wide"
)

# ── FPS Counter ───────────────────────────────────────────────────────────────
class FPSCounter:
    """Exponential moving-average FPS counter."""
    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._fps   = 0.0
        self._last  = time.perf_counter()

    def tick(self) -> float:
        now       = time.perf_counter()
        instant   = 1.0 / max(now - self._last, 1e-9)
        self._fps = self._alpha * instant + (1 - self._alpha) * self._fps
        self._last = now
        return self._fps


# ── Model Loader (Cached) ─────────────────────────────────────────────────────
@st.cache_resource
def get_detector() -> PersonDetector:
    """Load and cache the YOLOv8 model detector."""
    return PersonDetector(model_path=MODEL_PATH, verbose=False)


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("👥 Crowd Detection & Monitoring")
st.caption("Real-time crowd analysis and count prediction powered by YOLOv8 + Streamlit")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=CONFIDENCE_THRESHOLD,
    step=0.05,
    help="Minimum confidence value for a person detection to be accepted."
)

st.sidebar.markdown("---")
st.sidebar.write(f"**YOLO Model:** `{MODEL_PATH}`")
st.sidebar.write(f"**Alert Threshold:** `{ALERT_THRESHOLD}` persons")


# ── Controls ──────────────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False

col_start, col_stop = st.columns(2)
with col_start:
    if st.button("Start Detection", use_container_width=True, type="primary"):
        st.session_state.running = True
with col_stop:
    if st.button("Stop Detection", use_container_width=True):
        st.session_state.running = False


# ── Main UI Layout ────────────────────────────────────────────────────────────
st.markdown("---")
metrics_container = st.container()
video_container = st.container()

if st.session_state.running:
    with metrics_container:
        col_count, col_fps, col_density = st.columns(3)
        with col_count:
            count_placeholder = st.empty()
        with col_fps:
            fps_placeholder = st.empty()
        with col_density:
            density_placeholder = st.empty()
            
    with video_container:
        video_placeholder = st.empty()

    # Load resources
    detector = get_detector()
    renderer = FrameRenderer()
    fps_counter = FPSCounter()
    logger = CrowdLogger()

    # Capture loop
    try:
        with Camera() as cam, logger:
            while st.session_state.running:
                ret, frame = cam.read()
                if not ret:
                    st.error("Cannot access camera source. Please make sure the camera is not in use.")
                    st.session_state.running = False
                    break

                # Inference
                detections = detector.detect(frame, conf=conf_threshold)
                count = len(detections)
                fps = fps_counter.tick()

                # Overlay rendering (modifies frame copy in-place)
                annotated_frame = renderer.render(frame.copy(), detections, fps)

                # Update metrics
                count_placeholder.metric("Detected Persons", count)
                fps_placeholder.metric("Processing Speed", f"{fps:.1f} FPS")

                # Show density status
                if count < 5:
                    density_placeholder.success("Density: LOW")
                elif count < ALERT_THRESHOLD:
                    density_placeholder.warning("Density: MEDIUM")
                else:
                    density_placeholder.error("Density: HIGH / CRITICAL")

                # Display frame
                video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                # Log crowd count
                logger.tick(count)

                # Small delay to reduce CPU load when frame rate is high
                time.sleep(0.01)
                
    except SystemExit:
        st.error("Error: The camera application exited abnormally.")
        st.session_state.running = False
    except Exception as e:
        st.error(f"Error occurred during loop: {e}")
        st.session_state.running = False
else:
    st.info("System is offline. Click 'Start Detection' to launch the webcam feed.")
