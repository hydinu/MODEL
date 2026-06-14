# =============================================================================
# app.py — Streamlit Web Interface for Crowd Detection System
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
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling via markdown
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Apply fonts */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Main title styling */
        .title-container {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 25px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        /* Metrics custom cards */
        .metric-card {
            background-color: #1e293b;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #334155;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Status indicator colors */
        .status-low {
            background-color: #059669;
            color: white;
            font-weight: 600;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.25rem;
            letter-spacing: 0.5px;
        }
        .status-medium {
            background-color: #d97706;
            color: white;
            font-weight: 600;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.25rem;
            letter-spacing: 0.5px;
        }
        .status-high {
            background-color: #dc2626;
            color: white;
            font-weight: 600;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.25rem;
            letter-spacing: 0.5px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)


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


# ── App Layout & Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/crowd.png", width=80)
    st.title("Settings")
    
    st.subheader("Model Configuration")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence value for a person detection to be accepted."
    )
    
    st.markdown("---")
    st.subheader("System Info")
    st.info(
        f"**YOLO Model**: `{MODEL_PATH}`\n\n"
        f"**Alert Threshold**: `{ALERT_THRESHOLD}` persons"
    )


# ── Main Content Header ───────────────────────────────────────────────────────
st.markdown("""
    <div class="title-container">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800;">👥 Crowd Detection & Monitoring</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.8; font-size: 1.1rem;">
            Real-time crowd analysis and count prediction powered by YOLOv8 + Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)


# ── Control Buttons ───────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False

control_col1, control_col2 = st.columns(2)
with control_col1:
    if st.button("▶️ Start Detection", use_container_width=True, type="primary"):
        st.session_state.running = True

with control_col2:
    if st.button("⏹️ Stop Detection", use_container_width=True):
        st.session_state.running = False


# ── Metrics and Video Feed Area ───────────────────────────────────────────────
metrics_area = st.container()
video_area = st.container()

if st.session_state.running:
    with metrics_area:
        col1, col2, col3 = st.columns(3)
        with col1:
            count_metric = st.empty()
        with col2:
            fps_metric = st.empty()
        with col3:
            density_metric = st.empty()
            
    with video_area:
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

                # Determine crowd density level
                if count < 5:
                    density_class = "status-low"
                    density_label = "LOW"
                elif count < ALERT_THRESHOLD:
                    density_class = "status-medium"
                    density_label = "MEDIUM"
                else:
                    density_class = "status-high"
                    density_label = "CRITICAL / HIGH"

                # Update live metrics
                count_metric.markdown(
                    f"""
                    <div class="metric-card">
                        <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">
                            👥 Detected Persons
                        </span>
                        <h2 style="margin: 5px 0 0 0; font-size: 2.25rem; font-weight: 800; color: #38bdf8;">
                            {count}
                        </h2>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                fps_metric.markdown(
                    f"""
                    <div class="metric-card">
                        <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">
                            ⚡ Processing speed
                        </span>
                        <h2 style="margin: 5px 0 0 0; font-size: 2.25rem; font-weight: 800; color: #f59e0b;">
                            {fps:.1f} FPS
                        </h2>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                density_metric.markdown(
                    f"""
                    <div class="metric-card" style="padding: 16px;">
                        <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 8px;">
                            📊 Crowd Density Status
                        </span>
                        <div class="{density_class}">
                            {density_label}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

                # Display frame
                # Streamlit defaults to RGB, opencv returns BGR, so convert to RGB or use BGR channel mapping
                video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                # Log crowd count
                logger.tick(count)

                # Small delay to reduce CPU load when frame rate is high
                time.sleep(0.01)
                
    except SystemExit:
        # Catch sys.exit in case camera.py exits
        st.error("Error: The camera application exited abnormally.")
        st.session_state.running = False
    except Exception as e:
        st.error(f"Error occurred during loop: {e}")
        st.session_state.running = False
else:
    st.info("System is offline. Click 'Start Detection' to launch the webcam feed.")
    # Show empty dashboard mockup when off
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">👥 Detected Persons</span>
                <h2 style="margin: 5px 0 0 0; font-size: 2.25rem; font-weight: 800; color: #64748b;">--</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">⚡ Processing speed</span>
                <h2 style="margin: 5px 0 0 0; font-size: 2.25rem; font-weight: 800; color: #64748b;">--</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card" style="padding: 16px;">
                <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 8px;">📊 Crowd Density Status</span>
                <div style="background-color: #334155; color: #94a3b8; padding: 12px; border-radius: 8px; font-weight: 600; text-align: center; font-size: 1.25rem;">
                    OFFLINE
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
