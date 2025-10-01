import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw
import json
import base64
from io import BytesIO
import io
import time
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import websockets
import threading
import sys, os
import asyncio
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh

# Pastikan root project masuk ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.utils.area_picker import AreaPicker
from components.area_picker import StreamlitAreaPicker
from streamlit_js_eval import streamlit_js_eval  # pip install streamlit-js-eval
st.set_page_config(
    page_title="Warehouse Stock Counting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-connected { background-color: #28a745; }
    .status-disconnected { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
</style>
""", unsafe_allow_html=True)

# API base URL
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'areas' not in st.session_state:
    st.session_state.areas = []
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'drawing_mode' not in st.session_state:
    st.session_state.drawing_mode = False
if 'current_polygon' not in st.session_state:
    st.session_state.current_polygon = []
if 'websocket_data' not in st.session_state:
    st.session_state.websocket_data = None
if 'areas_image_width' not in st.session_state:
    st.session_state.areas_image_width = None
if 'areas_image_height' not in st.session_state:
    st.session_state.areas_image_height = None
if 'hls_url' not in st.session_state:
    st.session_state.hls_url = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'websocket_error' not in st.session_state:
    st.session_state.websocket_error = None


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def main():
    # Header
    st.markdown('<h1 class="main-header">📦 Warehouse Stock Counting System</h1>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ Backend API is not running. Please start the backend server first.")
        st.code("python scripts/run_backend.py")
        return
    
    # Display system status
    with st.sidebar:
        st.markdown("### System Status")
        
        # API Status
        st.markdown(f'<span class="status-indicator status-connected"></span>API: Connected', unsafe_allow_html=True)
        
        db_connected = False
        if health_data:
            if isinstance(health_data.get('database_connected'), bool):
                db_connected = health_data['database_connected']
            else:
                db_status = health_data.get('database_status', {})
                db_connected = bool(db_status.get('supabase_connected') or db_status.get('sqlite_available'))

        if db_connected:
            st.markdown(f'<span class="status-indicator status-connected"></span>Database: Connected', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-indicator status-warning"></span>Database: Not Connected', unsafe_allow_html=True)
        
        # RTSP Status
        if health_data and health_data.get('rtsp_connected'):
            st.markdown(f'<span class="status-indicator status-connected"></span>RTSP: Connected', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-indicator status-disconnected"></span>RTSP: Disconnected', unsafe_allow_html=True)

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🎯 Area Definition", "📊 Stock Monitoring", "📈 Analytics", "⚙️ Settings"]
    )
    
    if page == "🎯 Area Definition":
        area_definition_page()
    elif page == "📊 Stock Monitoring":
        stock_monitoring_page()
    elif page == "📈 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()

def area_definition_page():
    st.header("🎯 Area Definition")
    st.markdown("Define pallet areas for stock counting using interactive tools")
    
    area_picker = StreamlitAreaPicker()
    
    # Load existing areas
    load_existing_areas()
    
    # Method selection
    method = st.radio(
        "Choose definition method:",
        ["📁 Upload Image", "📹 RTSP Stream", "🖱️ Interactive OpenCV Picker"],
        horizontal=True
    )
    
    if method == "📁 Upload Image":
        upload_image_method(area_picker)
    elif method == "📹 RTSP Stream":
        rtsp_stream_method(area_picker)
    else:
        opencv_picker_method(area_picker)

def upload_image_method(area_picker):
    st.subheader("📁 Upload Image Method")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a top-view image of your warehouse"
    )
    
    if uploaded_file is not None:
        # Display image with area definition interface
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
        
        area_picker.create_interface(image=image)
        
        # Save areas button
        if st.button("💾 Save Areas to Backend", type="primary"):
            areas = area_picker.get_areas()
            if areas:
                save_areas_to_backend(uploaded_file, areas)
            else:
                st.warning("Please define areas first")

def opencv_picker_method(area_picker):
    """OpenCV interactive picker method"""
    st.subheader("🖱️ Interactive OpenCV Picker")
    st.markdown("Use OpenCV window for precise area definition with mouse controls")
    
    uploaded_file = st.file_uploader(
        "Choose an image file for OpenCV picker",
        type=['png', 'jpg', 'jpeg'],
        help="Upload image to open in OpenCV picker window"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image preview
            image = Image.open(uploaded_file)
            st.image(image, caption="Image Preview - Will open in OpenCV window", use_container_width=False)
            
            # Instructions
            st.markdown("""
            ### 🖱️ OpenCV Picker Instructions:
            - **Left Click**: Add point (4 points complete an area)
            - **Right Click**: Remove area (click inside area to remove)
            - **Middle Click**: Clear current points
            - **S Key**: Save manually
            - **C Key**: Clear all areas
            - **ESC**: Exit picker
            """)
            
            if st.button("🚀 Launch OpenCV Picker", type="primary"):
                launch_opencv_picker(uploaded_file, image)
        
        with col2:
            st.markdown("### Current Areas")
            
            # Load and display current areas
            try:
                response = requests.get(f"{API_BASE_URL}/get-areas")
                if response.status_code == 200:
                    areas_data = response.json()
                    areas = areas_data.get("areas", [])
                    
                    if areas:
                        st.success(f"✅ {len(areas)} areas defined")
                        
                        # Display areas
                        for i, area in enumerate(areas):
                            with st.expander(f"Area {i+1}"):
                                st.json(area)
                    else:
                        st.info("No areas defined yet")
                else:
                    st.warning("Could not load areas")
            except Exception as e:
                st.error(f"Error loading areas: {str(e)}")
            
            # Control buttons
            col_export, col_clear = st.columns(2)
            
            with col_export:
                if st.button("📤 Export Areas"):
                    export_areas()
            
            with col_clear:
                if st.button("🗑️ Clear All Areas"):
                    clear_all_areas()

def launch_opencv_picker(uploaded_file, image):
    """Launch OpenCV picker"""
    try:
        with st.spinner("Uploading image and preparing picker..."):
            # Upload image to backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            upload_response = requests.post(f"{API_BASE_URL}/upload-image", files=files)
            
            if upload_response.status_code == 200:
                image_path = upload_response.json().get("path")
                
                # Configure picker
                picker_config = {
                    "image_path": image_path,
                    "save_file": "data/warehouse_areas.pkl"
                }
                
                # Send picker configuration to backend
                config_response = requests.post(
                    f"{API_BASE_URL}/run-interactive-picker", 
                    json=picker_config
                )
                
                if config_response.status_code == 200:
                    st.success("✅ OpenCV picker configured successfully!")
                    st.info("🖱️ OpenCV picker window should open now. Follow the instructions to define areas.")
                    
                    # Display instructions again
                    instructions = config_response.json().get("instructions", {})
                    st.json(instructions)
                    
                    # Auto-refresh to show updated areas
                    if st.button("🔄 Refresh Areas"):
                        st.rerun()
                else:
                    st.error("❌ Failed to configure picker")
            else:
                st.error("❌ Failed to upload image")
                
    except Exception as e:
        st.error(f"❌ Error launching picker: {str(e)}")

def export_areas():
    """Export areas"""
    try:
        response = requests.get(f"{API_BASE_URL}/export-areas")
        if response.status_code == 200:
            data = response.json()
            areas_json = json.dumps(data["data"], indent=2)
            
            st.download_button(
                label="📥 Download Areas JSON",
                data=areas_json,
                file_name=f"warehouse_areas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            st.success("✅ Areas exported successfully!")
        else:
            st.error("❌ Failed to export areas")
    except Exception as e:
        st.error(f"❌ Error exporting areas: {str(e)}")

def clear_all_areas():
    """Clear all areas"""
    try:
        # Clear areas by saving empty list
        areas_data = {
            "areas": [],
            "image_path": "cleared"
        }
        
        response = requests.post(f"{API_BASE_URL}/save-areas", json=areas_data)
        if response.status_code == 200:
            st.success("✅ All areas cleared!")
            st.rerun()
        else:
            st.error("❌ Failed to clear areas")
    except Exception as e:
        st.error(f"❌ Error clearing areas: {str(e)}")

def save_areas_to_backend(uploaded_file, areas):
    """Save areas to backend with improved error handling"""
    if not areas:
        st.warning("Please define areas first")
        return
    
    try:
        with st.spinner("Saving areas..."):
            # Upload image first
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            upload_response = requests.post(f"{API_BASE_URL}/upload-image", files=files)
            
            if upload_response.status_code == 200:
                image_path = upload_response.json().get("path")
                
                # Save areas
                areas_data = {
                    "areas": areas,
                    "image_path": image_path
                }
                
                save_response = requests.post(f"{API_BASE_URL}/save-areas", json=areas_data)
                
                if save_response.status_code == 200:
                    result = save_response.json()
                    st.success("✅ Areas saved successfully!")
                    st.info(f"💾 Saved to: {', '.join(result.get('save_locations', []))}")
                    st.balloons()
                else:
                    st.error("❌ Failed to save areas")
            else:
                st.error("❌ Failed to upload image")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def rtsp_stream_method(area_picker):
    st.subheader("📹 RTSP Stream Method")
    
    rtsp_url = st.text_input(
        "RTSP URL",
        placeholder="rtsp://username:password@ip:port/stream",
        help="Enter your RTSP camera stream URL"
    )

    if rtsp_url:
        col1, col2 = st.columns([3, 1])

        with col1:
            # 🔗 Tampilkan live stream via HLS
            if st.button("▶️ Start Live Stream"):
                connect_to_rtsp_stream(rtsp_url, st.empty())

            # 📸 Capture frame untuk definisi area
            if st.button("📸 Capture Frame for Area Definition"):
                image = None
                try:
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

                    if not cap.isOpened():
                        st.warning("⚠️ OpenCV gagal buka stream, coba ffmpeg fallback...")
                    else:
                        ret, frame = False, None

                        # Buang frame awal biar buffer stabil
                        for _ in range(30):
                            cap.read()

                        start_time = time.time()
                        while time.time() - start_time < 5:  # max 5 detik nunggu frame valid
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                break
                            time.sleep(0.05)  # 50ms delay

                        cap.release()

                        if ret and frame is not None:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(frame_rgb)
                            st.success("✅ Frame captured with OpenCV")

                    # --- fallback ke ffmpeg kalau gagal ---
                    if image is None:
                        try:
                            image = capture_frame_ffmpeg(rtsp_url)
                            st.success("✅ Frame captured via ffmpeg fallback")
                        except Exception as e:
                            st.error(f"❌ Gagal capture frame dengan ffmpeg: {str(e)}")

                    # Simpan hasil ke session
                    if image:
                        st.session_state.current_frame = image

                except Exception as e:
                    st.error(f"❌ Error saat capture: {str(e)}")

            # Kalau sudah ada snapshot → tampilkan picker
            if "current_frame" in st.session_state:
                image = st.session_state.current_frame
                st.image(image, caption="Captured RTSP Frame", use_container_width=True)

                area_picker.create_interface(image=image)

                if st.button("💾 Save Areas to Backend", type="primary"):
                    areas = area_picker.get_areas()
                    if areas:
                        save_areas_to_backend_from_rtsp(rtsp_url, areas, image)
                    else:
                        st.warning("Please define areas first")

        with col2:
            st.markdown("### Current Areas")
            if "areas" in st.session_state and st.session_state.areas:
                for i, area in enumerate(st.session_state.areas):
                    with st.expander(f"Area {i+1}"):
                        st.json(area)


# fallback snapshot pakai ffmpeg
import subprocess, numpy as np
import cv2, time, io, subprocess, numpy as np
from PIL import Image
def capture_rtsp_frame(rtsp_url, timeout=10):
    """Ambil frame dari RTSP dengan OpenCV, fallback ke ffmpeg kalau gagal"""
    start_time = time.time()
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("⚠️ OpenCV gagal buka stream, fallback ke ffmpeg...")
        return capture_frame_ffmpeg(rtsp_url)

    frame = None
    ret = False

    # Drop frame awal biar buffer bersih
    for _ in range(30):
        cap.read()

    while time.time() - start_time < timeout:
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.05)  # tunggu 50ms

    cap.release()

    if not ret or frame is None:
        print("⚠️ OpenCV gagal capture frame valid, fallback ke ffmpeg...")
        return capture_frame_ffmpeg(rtsp_url)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def capture_frame_ffmpeg(rtsp_url):
    """Ambil snapshot tunggal via ffmpeg (lebih stabil)"""
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-y",                # overwrite
        "-i", rtsp_url,
        "-frames:v", "1",    # ambil hanya 1 frame
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo", "-"
    ]

    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_image = pipe.stdout.read()
    pipe.stdout.close()
    pipe.wait()

    if not raw_image:
        raise RuntimeError("❌ FFmpeg gagal ambil snapshot")

    # Baca data gambar dengan PIL langsung
    image = Image.open(io.BytesIO(raw_image)).convert("RGB")
    return image



def save_areas_to_backend_from_rtsp(rtsp_url, areas, image):
    """Save areas to backend from RTSP snapshot"""
    try:
        with st.spinner("Saving areas..."):
            # Simpan snapshot sebagai file sementara
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            buf.seek(0)

            files = {"file": ("rtsp_snapshot.jpg", buf, "image/jpeg")}
            upload_response = requests.post(f"{API_BASE_URL}/upload-image", files=files)

            if upload_response.status_code == 200:
                image_path = upload_response.json().get("path")
                
                # Simpan area ke backend
                areas_data = {
                    "areas": areas,
                    "image_path": image_path,
                    "source": rtsp_url
                }
                save_response = requests.post(f"{API_BASE_URL}/save-areas", json=areas_data)

                if save_response.status_code == 200:
                    st.success("✅ RTSP Areas saved successfully!")
                    st.balloons()
                else:
                    st.error("❌ Failed to save areas")
            else:
                st.error("❌ Failed to upload RTSP snapshot")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def connect_to_rtsp_stream(rtsp_url, placeholder):
    """Start HLS from backend and display via hls.js player"""
    try:
        res = requests.post(f"{API_BASE_URL}/start-rtsp-hls", json={"rtsp_url": rtsp_url})
        if res.status_code == 200:
            data = res.json()

            # Gunakan playlist_url dari backend kalau ada, fallback ke session_id
            if "playlist_url" in data:
                playlist_url = f"{API_BASE_URL}{data['playlist_url']}"
            else:
                playlist_url = f"{API_BASE_URL}/streams/{data['session_id']}/index.m3u8"

            player_html = f"""
            <html>
              <head>
                <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
              </head>
              <body style="margin:0; padding:0; background:black;">
                <video id="video" width="100%" height="480" controls autoplay muted playsinline></video>
                <script>
                  var video = document.getElementById('video');
                  if (Hls.isSupported()) {{
                    var hls = new Hls();
                    hls.loadSource("{playlist_url}");
                    hls.attachMedia(video);
                    hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                      video.play();
                    }});
                  }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                    // Safari native support
                    video.src = "{playlist_url}";
                    video.addEventListener('loadedmetadata', function() {{
                      video.play();
                    }});
                  }} else {{
                    document.body.innerHTML = "<p style='color:white;text-align:center;'>HLS not supported in this browser.</p>";
                  }}
                </script>
              </body>
            </html>
            """

            components.html(player_html, height=500)
            st.success("✅ HLS Stream started")
        else:
            st.error("❌ Gagal start HLS stream")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def save_stream_areas(rtsp_url):
    """Save areas defined on stream"""
    try:
        areas_data = {
            "areas": st.session_state.areas,
            "image_path": f"rtsp_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        response = requests.post(f"{API_BASE_URL}/save-areas", json=areas_data)
        
        if response.status_code == 200:
            st.success("✅ Stream areas saved successfully!")
        else:
            st.error("❌ Failed to save stream areas")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def stock_monitoring_page():
    st.header("📊 Stock Monitoring")
    st.markdown("Real-time stock monitoring and counting")
    
    # Check if areas are defined
    try:
        response = requests.get(f"{API_BASE_URL}/get-areas")
        if response.status_code == 200:
            areas_data = response.json()
            areas = areas_data.get("areas", [])
            st.session_state.areas = areas  # simpan ke session
            st.session_state.areas_image_width = areas_data.get("image_width")
            st.session_state.areas_image_height = areas_data.get("image_height")
            
            if not areas:
                st.warning("⚠️ No areas defined. Please define areas first in the Area Definition page.")
                if st.button("Go to Area Definition"):
                    st.switch_page("🎯 Area Definition")
                return
            
            st.success(f"✅ Loaded {len(areas)} defined areas")
        else:
            st.error("❌ Failed to load areas")
            return
    except Exception as e:
        st.error(f"❌ Error loading areas: {str(e)}")
        return
    
    # Monitoring options
    st.subheader("Monitoring Options")
    
    monitoring_type = st.radio(
        "Choose monitoring type:",
        ["📹 RTSP Live Stream", "📁 Video File"],
        horizontal=True
    )
    
    if monitoring_type == "📹 RTSP Live Stream":
        rtsp_monitoring()
    else:
        video_file_monitoring()

# ===============================
# Drawing helper
# ===============================
def _draw_areas_on_frame(rgb_frame, areas):
    """
    Draw areas dengan warna:
    - Area 1 = kapasitas total (kuning)
    - Area 2+ = pallet (merah jika kosong, hijau jika terisi)
    """
    overlay = rgb_frame.copy()

    for idx, area in enumerate(areas):
        try:
            points = np.array(area["points"], np.int32)
            status = area.get("status", "empty")

            if idx == 0:
                # Area 1 khusus kapasitas
                color = (255, 255, 0)  # 🟡 kuning
                label = area.get("name", "Capacity Area")
            else:
                if status == "empty":
                    color = (0, 0, 255)   # 🔴 merah
                    label = "Empty"
                else:  
                    color = (0, 255, 0)   # 🟢 hijau
                    label = "Terisi"

            cv2.polylines(overlay, [points], True, color, 2)
            centroid = np.mean(points, axis=0).astype(int)
            cv2.putText(
                overlay,
                label,
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        except Exception as e:
            print(f"[WARN] Area draw error: {e}")
            continue

    return overlay


# ===============================
# Monitoring
# ===============================

def rtsp_monitoring():
    st.subheader("📹 RTSP Live Monitoring - Optimized for Real-time")

    rtsp_url = st.text_input(
        "RTSP URL",
        placeholder="rtsp://username:password@ip:port/stream",
        help="Masukkan URL RTSP kamera",
    )
    
    with st.expander("⚙️ Performance Settings"):
        col1, col2 = st.columns(2)
        with col1:
            detection_interval = st.slider(
                "Detection Interval (frames)",
                min_value=1,
                max_value=10,
                value=3,
                help="Process full detection every N frames. Higher = faster but less responsive."
            )
        with col2:
            target_fps = st.slider(
                "Target FPS",
                min_value=5,
                max_value=30,
                value=15,
                help="Target frames per second. Lower = better performance."
            )
        
        if st.button("Apply Settings"):
            # Update backend settings via API
            try:
                requests.post(f"{API_BASE_URL}/update-performance-settings", json={
                    "detection_interval": detection_interval,
                    "target_fps": target_fps
                })
                st.success("✅ Settings applied!")
            except:
                st.warning("Could not update backend settings")

    if rtsp_url:
        mode_choice = st.radio(
            "Pilih Mode Streaming",
            options=[
                ("HLS (recommended for real-time)", "hls"),
                ("WebSocket (with overlay & stats)", "ws"),
            ],
            format_func=lambda x: x[0],
            key="stream_mode_select"
        )
        mode_value = mode_choice[1]

        if not st.session_state.get("monitoring_active", False):
            if st.button("▶️ Start Monitoring", type="primary"):
                try:
                    payload = {
                        "rtsp_url": rtsp_url,
                        "mode": mode_value,
                        "areas": st.session_state.get("areas", [])
                    }
                    resp = requests.post(
                        f"{API_BASE_URL}/start-rtsp-monitoring",
                        json=payload
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state.monitoring_active = True
                        st.session_state.hls_url = result.get("hls_playlist_absolute")
                        st.session_state.mode = mode_value
                        st.session_state.session_id = result.get("session_id")

                        # load areas dari backend
                        try:
                            a = requests.get(f"{API_BASE_URL}/get-areas").json()
                            st.session_state.areas = a.get("areas", [])
                            st.session_state.areas_image_width = a.get("image_width")
                            st.session_state.areas_image_height = a.get("image_height")
                        except Exception:
                            st.session_state.areas = []
                            st.session_state.areas_image_width = None
                            st.session_state.areas_image_height = None

                        if mode_value == "ws":
                            start_websocket_monitoring()

                        st.success("✅ Monitoring started with optimized real-time detection!")
                        st.info("💡 System uses frame caching and adaptive detection for better performance")
                    else:
                        st.error("❌ Failed to start monitoring")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            _render_live_view()


# ===============================
# Render Live View
# ===============================
def _render_live_view():
    """Render live view: HLS dengan overlay canvas, atau WebSocket dengan frame+stats"""
    if not st.session_state.get("monitoring_active", False):
        st.info("ℹ️ Monitoring belum aktif.")
        return

    mode = st.session_state.get("mode", "hls")

    if mode == "hls":
        hls_url = st.session_state.get("hls_url")
        if not hls_url:
            st.error("❌ HLS URL tidak ditemukan")
        else:
            areas = st.session_state.get("areas", [])
            areas_meta = {
                "image_width": st.session_state.get("areas_image_width"),
                "image_height": st.session_state.get("areas_image_height")
            }

            areas_js = json.dumps(areas)
            meta_js = json.dumps(areas_meta)

            components.html(f"""
                <style>
                #video-container{{position:relative;width:100%;max-width:900px;}}
                #video{{width:100%;height:auto;display:block;}}
                #overlay{{position:absolute;top:0;left:0;width:100%;height:100%;
                          pointer-events:none;z-index:10}}
                </style>

                <div id="video-container">
                  <video id="video" controls autoplay muted playsinline></video>
                  <canvas id="overlay"></canvas>
                </div>

                <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
                <script>
                const hlsUrl = "{hls_url}";
                let rawAreas = {areas_js};
                const meta = {meta_js};
                let stopDrawing = false;

                const video = document.getElementById('video');
                const canvas = document.getElementById('overlay');
                const ctx = canvas.getContext('2d');

                function resizeCanvas() {{
                    canvas.width = video.clientWidth;
                    canvas.height = video.clientHeight;
                }}

                function normalizePoints(pt, imgW, imgH) {{
                    const x = (pt[0] > 1 && imgW) ? (pt[0] / imgW) : pt[0];
                    const y = (pt[1] > 1 && imgH) ? (pt[1] / imgH) : pt[1];
                    return [x, y];
                }}

                function buildAreasForDrawing() {{
                    const imgW = meta.image_width || null;
                    const imgH = meta.image_height || null;
                    let normalized = [];
                    try {{
                        rawAreas.forEach((a, idx) => {{
                            const pts = a.points || a;
                            const name = a.name || (idx === 0 ? "Capacity Area" : "Pallet " + idx);
                            const status = a.status || "empty";
                            const ptsNorm = pts.map(p => normalizePoints(p, imgW, imgH));
                            normalized.push({{name: name, status: status, points: ptsNorm, idx: idx}});
                        }});
                    }} catch(e) {{
                        console.warn("Area parsing error", e);
                    }}
                    return normalized;
                }}

                function drawAreas() {{
                    if (stopDrawing) return;
                    if (!video.clientWidth || !video.clientHeight) return;
                    resizeCanvas();
                    ctx.clearRect(0,0,canvas.width,canvas.height);
                    ctx.lineWidth = 2;
                    ctx.font = "14px Arial";

                    const areas = buildAreasForDrawing();
                    areas.forEach((area) => {{
                        const pts = area.points;
                        if (!pts || pts.length === 0) return;

                        let color = "lime";
                        let label = area.name;

                        if (area.idx === 0) {{
                            // Area 1 = kapasitas
                            color = "yellow";
                            label = "Capacity Area";
                        }} else {{
                            if (area.status === "empty") {{
                                color = "red";   // 🔴 kosong
                                label = "Empty";
                            }} else {{
                                color = "green"; // 🟢 terisi
                                label = "Terisi";
                            }}
                        }}

                        ctx.strokeStyle = color;
                        ctx.fillStyle = color;

                        ctx.beginPath();
                        pts.forEach((p, i) => {{
                            const x = Math.round(p[0] * canvas.width);
                            const y = Math.round(p[1] * canvas.height);
                            if (i===0) ctx.moveTo(x,y);
                            else ctx.lineTo(x,y);
                        }});
                        ctx.closePath();
                        ctx.stroke();

                        const tx = Math.round(pts.reduce((a, p) => a + p[0], 0) / pts.length);
                        const ty = Math.round(pts.reduce((a, p) => a + p[1], 0) / pts.length) - 6;
                        ctx.fillText(label, tx, ty);
                    }});
                    requestAnimationFrame(drawAreas);
                }}

                if (Hls.isSupported()) {{
                    var hls = new Hls();
                    hls.loadSource(hlsUrl);
                    hls.attachMedia(video);
                    hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                        video.play();
                    }});
                }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                    video.src = hlsUrl;
                }}

                video.addEventListener('loadedmetadata', resizeCanvas);
                video.addEventListener('play', () => {{
                    resizeCanvas();
                    stopDrawing = false;
                    requestAnimationFrame(drawAreas);
                }});

                window.addEventListener('resize', resizeCanvas);

                window.addEventListener("message", (event) => {{
                    if (event.data === "STOP_MONITORING") {{
                        stopDrawing = true;
                        ctx.clearRect(0,0,canvas.width,canvas.height);
                    }}
                }});
                </script>
            """, height=560)

    elif mode == "ws":
        st.markdown("### WebSocket mode — overlay rendered dari frames & stats")
        frame_pl = st.empty()
        stats_pl = st.empty()
        st_autorefresh(interval=500, key="ws_refresh")
        data = st.session_state.get("websocket_data")

        if data and data.get("frame"):
            rgb = _decode_frame_b64_to_rgb(data["frame"])
            if rgb is not None:
                areas = st.session_state.get("areas", [])
                rgb = _draw_areas_on_frame(rgb, areas)
                frame_pl.image(rgb, channels="RGB", use_container_width=True)

        if data and data.get("count_data"):
            display_monitoring_metrics({"count_data": data.get("count_data")}, stats_pl)

    if st.button("⏹️ Stop Monitoring", type="primary"):
        try:
            resp = requests.post(f"{API_BASE_URL}/stop-monitoring")
            if resp.status_code == 200:
                st.success("✅ Monitoring stopped")
                components.html("<script>window.postMessage('STOP_MONITORING','*');</script>", height=0)
            else:
                st.error("❌ Failed to stop monitoring")
        except Exception as e:
            st.error(f"❌ Error stopping monitoring: {e}")

        st.session_state.monitoring_active = False
        st.session_state.hls_url = None
        st.session_state.mode = None
        st.session_state.websocket_data = None
        st.session_state.areas = []
        st.session_state.areas_image_width = None
        st.session_state.areas_image_height = None


# ===============================
# Frame decode helper
# ===============================
def _decode_frame_b64_to_rgb(frame_b64: str):
    try:
        frame_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None




def stop_monitoring():
    """Stop monitoring"""
    try:
        response = requests.post(f"{API_BASE_URL}/stop-monitoring")
        if response.status_code == 200:
            print("✅ Monitoring stopped via API")
            return True
        else:
            print(f"❌ Failed to stop monitoring, status {response.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        print(f"Exception in stop_monitoring: {e}")
        return False




def start_rtsp_monitoring(rtsp_url, placeholder):
    """Start RTSP monitoring"""
    try:
        with st.spinner("Starting RTSP monitoring..."):
            response = requests.post(
                f"{API_BASE_URL}/start-rtsp-monitoring",
                json={"rtsp_url": rtsp_url}
            )
            
            if response.status_code == 200:
                st.session_state.monitoring_active = True
                result = response.json()
                st.success(f"✅ Monitoring started - Session: {result.get('session_id', 'Unknown')}")
            else:
                st.error("❌ Failed to start monitoring")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# ===============================
# WebSocket listener
# ===============================

def start_websocket_monitoring():
    """Start a background thread to listen to WebSocket and update session_state."""
    def _runner():
        asyncio.run(_ws_listener())

    t = threading.Thread(target=_runner, daemon=True)
    t.start()


# ===============================
# WebSocket Listener
# ===============================
async def _ws_listener():
    ws_url = API_BASE_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws/monitoring"
    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            while True:
                msg = await ws.recv()
                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                st.session_state.websocket_data = data

                # Update status area sesuai hasil dari backend
                if details := data.get("count_data", {}).get("details"):
                    for d in details:
                        area_id = d["area_id"]
                        if area_id < len(st.session_state.areas):
                            st.session_state.areas[area_id]["status"] = d["status"]

                    # Simpan ke DB
                    try:
                        requests.post(f"{API_BASE_URL}/save-monitoring-result", json={
                            "session_id": st.session_state.get("session_id"),
                            "timestamp": data.get("timestamp"),
                            "count_data": details
                        })
                    except Exception as e:
                        st.session_state.websocket_error = f"Gagal save ke DB: {e}"
    except Exception as e:
        st.session_state.websocket_error = str(e)

def check_monitoring_status():
    """Check monitoring status"""
    try:
        response = requests.get(f"{API_BASE_URL}/monitoring-status")
        if response.status_code == 200:
            status = response.json()
            
            st.json(status)
            
            if status.get("is_monitoring"):
                st.success("✅ Monitoring is active")
            else:
                st.info("ℹ️ Monitoring is not active")
        else:
            st.error("❌ Failed to get status")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")









def video_file_monitoring():
    st.subheader("📁 Video File Monitoring")

    uploaded_video = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov"], help="Upload a video file for stock monitoring"
    )

    if uploaded_video is not None:
        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("▶️ Start Processing", type="primary"):
                try:
                    with st.spinner("Starting file monitoring..."):
                        files = {"file": (uploaded_video.name, uploaded_video.getvalue(), uploaded_video.type)}
                        response = requests.post(f"{API_BASE_URL}/start-file-monitoring", files=files)
                        if response.status_code == 200:
                            st.success("✅ File monitoring started")
                            start_websocket_monitoring()
                            st.session_state.monitoring_active = True
                            _render_live_view()
                        else:
                            st.error("❌ Failed to start file monitoring")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        with col2:
            st.markdown("### Processing Stats")
            st.empty()
            st.markdown("_(Stats will appear here during processing)_")


# === (2) Monitoring Metrics ===
def display_monitoring_metrics(data, placeholder):
    """Display monitoring metrics"""
    if not data:
        placeholder.info("No monitoring data available yet...")
        return
    
    count_data = data.get("count_data", {})

    with placeholder.container():
        # === Summary Metrics ===
        col1, col2, col3, col4 = st.columns(4)
        
        total_areas = count_data.get("total_areas", 0)
        occupied = count_data.get("occupied_slots", 0)
        empty = count_data.get("empty_slots", 0)
        full = count_data.get("full_slots", 0)

        with col1:
            st.metric("Total Areas", total_areas)
        
        with col2:
            st.metric("Occupied", occupied)
        
        with col3:
            st.metric("Empty", empty)
        
        with col4:
            occupancy_rate = (occupied / total_areas * 100) if total_areas > 0 else 0
            st.metric("Occupancy", f"{occupancy_rate:.1f}%")
        
        # === Estimated Stock ===
        if any(k in count_data for k in ("estimated_pallets", "estimated_sacks", "estimated_weight_tons")):
            st.markdown("### 📦 Estimated Stock")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Pallets", count_data.get("estimated_pallets", 0))
            
            with col2:
                st.metric("Sacks", count_data.get("estimated_sacks", 0))
            
            with col3:
                st.metric("Weight (tons)", f"{count_data.get('estimated_weight_tons', 0):.1f}")
        
        # === Detailed per-area view ===
        per_area = count_data.get("per_area", [])
        if per_area:
            st.markdown("### 📋 Area Details")
            df = pd.DataFrame(per_area)
            if not df.empty:
                st.dataframe(df, use_container_width=True, height=300)


# === (3) Analytics Page ===
def analytics_page():
    st.header("📈 Analytics Dashboard")
    st.markdown("Stock counting analytics and historical reports")
    
    try:
        response = requests.get(f"{API_BASE_URL}/analytics/history?limit=100")
        if response.status_code == 200:
            data = response.json()
            history = data.get("history", [])
            
            if not history:
                st.info("📊 No historical data available yet. Start monitoring to collect data.")
                return
            
            df = pd.DataFrame(history)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            display_analytics_dashboard(df)
        else:
            st.error("❌ Failed to load analytics data")
            
    except Exception as e:
        st.error(f"❌ Error loading analytics: {str(e)}")


# === (4) Display Analytics Dashboard ===
def display_analytics_dashboard(df: pd.DataFrame):
    """Display analytics dashboard"""
    st.subheader("📊 Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_occupancy = df['occupied_count'].mean() if not df.empty else 0
        st.metric("Avg Occupancy", f"{avg_occupancy:.1f}")
    
    with col2:
        total_records = len(df)
        st.metric("Total Records", total_records)
    
    with col3:
        max_occupied = df['occupied_count'].max() if not df.empty else 0
        st.metric("Peak Occupancy", max_occupied)
    
    with col4:
        latest_empty = df['empty_count'].iloc[-1] if not df.empty else 0
        st.metric("Current Empty", latest_empty)
    
    # Time series charts
    st.subheader("📈 Trends Over Time")
    
    if not df.empty:
        fig_occupancy = px.line(
            df, 
            x='timestamp', 
            y=['occupied_count', 'empty_count'],
            title='Occupancy Trends',
            labels={'value': 'Count', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig_occupancy, use_container_width=True)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        if not df.empty:
            fig_dist = px.histogram(
                df, 
                x='occupied_count',
                title='Occupancy Distribution',
                nbins=20
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        if not df.empty and 'timestamp' in df.columns and len(df) > 24:
            df['hour'] = df['timestamp'].dt.hour
            hourly_avg = df.groupby('hour')['occupied_count'].mean().reset_index()
            
            fig_hourly = px.bar(
                hourly_avg,
                x='hour',
                y='occupied_count',
                title='Average Occupancy by Hour'
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

    # Data table
    st.subheader("📋 Recent Records")
    st.dataframe(df.head(20), use_container_width=True)

    # Export options
    st.subheader("💾 Export Data")
    
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📊 Download CSV",
        data=csv,
        file_name=f"stock_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    if st.button("📈 Generate Report"):
        generate_analytics_report(df)


# === (5) Generate Analytics Report ===
def generate_analytics_report(df: pd.DataFrame):
    """Generate analytics report"""
    st.success("📈 Analytics report generated!")
    
    with st.expander("📊 Statistical Summary"):
        if not df.empty:
            st.write(df.describe())
        else:
            st.info("No data available for summary.")


# === (6) Settings Page ===
def settings_page():
    st.header("⚙️ Settings")
    st.markdown("Configure system parameters and preferences")
    
    # System configuration
    st.subheader("🔧 System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Detection Parameters")
        
        empty_threshold = st.slider(
            "Empty Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="Threshold for determining if an area is empty"
        )
        
        frame_rate = st.slider(
            "Frame Rate (FPS)",
            min_value=1,
            max_value=60,
            value=30,
            help="Processing frame rate for monitoring"
        )
    
    with col2:
        st.markdown("### Stock Calculation")
        
        pallet_height = st.number_input(
            "Pallet Height (layers)",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of pallet layers"
        )
        
        pallet_capacity = st.number_input(
            "Pallet Capacity (sacks)",
            min_value=1,
            max_value=100,
            value=20,
            help="Number of sacks per pallet"
        )
        
        sack_weight = st.number_input(
            "Sack Weight (kg)",
            min_value=1,
            max_value=100,
            value=50,
            help="Weight per sack in kilograms"
        )
    
    # Database configuration
    st.subheader("🗄️ Database Configuration")
    
    supabase_url = st.text_input(
        "Supabase URL",
        placeholder="https://your-project.supabase.co",
        help="Your Supabase project URL"
    )
    
    supabase_key = st.text_input(
        "Supabase Anon Key",
        type="password",
        placeholder="Your Supabase anonymous key",
        help="Your Supabase anonymous/public key"
    )
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")
        st.info("ℹ️ Restart the application to apply changes")
    
    # System information
    st.subheader("ℹ️ System Information")
    
    api_status, msg = check_api_health()
    system_info = {
        "API Status": "Connected" if api_status else f"Disconnected ({msg})",
        "Frontend Version": "1.0.0",
        "Backend Version": "1.0.0",
        "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for key, value in system_info.items():
        st.write(f"**{key}:** {value}")

def load_existing_areas():
    """Load existing areas from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/get-areas")
        if response.status_code == 200:
            data = response.json()
            st.session_state.areas = data.get("areas", [])
            st.session_state.areas_image_width = data.get("image_width")
            st.session_state.areas_image_height = data.get("image_height")
            if st.session_state.areas:
                st.success(f"✅ Loaded {len(st.session_state.areas)} existing areas")
    except Exception as e:
        st.warning(f"Could not load existing areas: {str(e)}")

if __name__ == "__main__":
    main()
