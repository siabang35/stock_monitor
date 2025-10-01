#from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import json
import sys, os
import pickle
from typing import List, Dict, Any
import asyncio
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import uuid

from fastapi.responses import FileResponse
from config import settings
from database import db_manager
from rtsp_handler import rtsp_handler
from hls_handler import hls_handler
from models import *

import logging

# pastikan project root (warehouse-stock-counting) masuk ke sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.area_picker import AreaPicker, create_picker_from_image_path

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="API for warehouse stock counting system"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static files
app.mount("/static", StaticFiles(directory=settings.upload_dir), name="static")

# Mount HLS public directory (served as /streams/{session}/index.m3u8)
try:
    hls_dir = getattr(settings, "hls_dir", os.path.join(settings.temp_dir, "hls"))
except Exception:
    hls_dir = os.path.join("hls")
os.makedirs(hls_dir, exist_ok=True)
app.mount("/streams", StaticFiles(directory=hls_dir), name="streams")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # naik dari backend ke root

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
# Global variables
current_areas = []
monitoring_sessions = {}
monitoring_mode = "none"  # one of: "none" | "rtsp" | "file"
monitoring_task = None
current_frame_b64 = None  # last processed frame (base64)
last_count_data = None

frame_cache = {
    "last_detection_frame": None,
    "last_detection_results": None,
    "frame_counter": 0,
    "detection_interval": 3,
    "last_areas_hash": None
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Simpan konfigurasi default
performance_settings = {
    "detection_interval": 3,
    "target_fps": 15
}

class PerformanceSettings(BaseModel):
    detection_interval: int
    target_fps: int

@app.get("/")
async def root():
    return {
        "message": "Warehouse Stock Counting API",
        "version": settings.api_version,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    db_status = db_manager.get_connection_status()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_status": db_status,
        "database_connected": bool(db_status.get("supabase_connected") or db_status.get("sqlite_available")),
        "rtsp_connected": rtsp_handler.is_connected(),
        "hls_sessions": hls_handler.get_status(),
        "areas_loaded": len(current_areas) > 0,
        "version": settings.api_version
    }

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read file
        contents = await file.read()
        if len(contents) > settings.max_file_size:
            raise HTTPException(status_code=400, detail="File too large")

        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1] or ".jpg"
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        image_path = os.path.join(settings.upload_dir, unique_filename)

        # Ensure upload directory exists
        os.makedirs(settings.upload_dir, exist_ok=True)

        # Validate image using Pillow
        try:
            img_pil = Image.open(BytesIO(contents))
            width, height = img_pil.size
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Save file
        with open(image_path, "wb") as f:
            f.write(contents)

        return {
            "message": "Image uploaded successfully",
            "path": image_path,
            "filename": unique_filename,
            "size": len(contents),
            "dimensions": {"width": width, "height": height}
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("Upload error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-interactive-picker")
async def run_interactive_picker(picker_config: dict):
    """Run interactive OpenCV picker"""
    try:
        image_path = picker_config.get("image_path")
        save_file = picker_config.get("save_file", "data/warehouse_areas.pkl")
        
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail="Invalid image path")
        
        # Create picker instance
        picker = AreaPicker(save_file=save_file)
        
        # This would typically run in a separate process/thread
        # For now, we'll return the picker configuration
        return {
            "message": "Interactive picker configuration ready",
            "image_path": image_path,
            "save_file": save_file,
            "instructions": {
                "left_click": "Add point (4 points = complete area)",
                "right_click": "Remove area",
                "middle_click": "Clear current points",
                "s_key": "Save manually",
                "c_key": "Clear all areas",
                "esc_key": "Exit picker"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- save_areas (update) ---
@app.post("/save-areas")
async def save_areas(areas_data: AreasData, background_tasks: BackgroundTasks):
    """Save defined areas with enhanced database integration (now storing image dims)"""
    try:
        global current_areas
        current_areas = areas_data.areas

        # Read image dimensions if image_path provided and exists
        image_width = None
        image_height = None
        try:
            if areas_data.image_path and os.path.exists(areas_data.image_path):
                pil = Image.open(areas_data.image_path)
                image_width, image_height = pil.size
        except Exception:
            # ignore, not critical
            image_width = None
            image_height = None

        # attach metadata for saving
        # store as dict with metadata if you prefer; for backward compatibility we still store list of polygons
        saved_payload = {
            "areas": current_areas,
            "image_path": areas_data.image_path,
            "image_width": image_width,
            "image_height": image_height
        }

        # Save picker file (improved)
        picker = AreaPicker(save_file="data/warehouse_areas.pkl")
        picker.set_areas(saved_payload)  # ensure AreaPicker supports dict or adjust accordingly

        # Save to legacy format for compatibility (pickle only areas list)
        areas_file = os.path.join(settings.temp_dir, "object_positions")
        os.makedirs(os.path.dirname(areas_file), exist_ok=True)
        with open(areas_file, "wb") as f:
            pickle.dump(current_areas, f)

        # Save to database
        db_save_success = await db_manager.save_area_definition(
            current_areas,
            areas_data.image_path,
            meta={"image_width": image_width, "image_height": image_height}
        )

        save_locations = ["data/warehouse_areas.pkl", areas_file]
        if db_save_success:
            db_status = db_manager.get_connection_status()
            save_locations.append(f"Database ({db_status['primary_database']})")

        return {
            "message": "Areas saved successfully",
            "count": len(current_areas),
            "timestamp": datetime.now().isoformat(),
            "save_locations": save_locations,
            "database_saved": db_save_success,
            "database_status": db_manager.get_connection_status(),
            "image_width": image_width,
            "image_height": image_height
        }
    except Exception as e:
        logger.error(f"Error saving areas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- get_areas (update) ---
@app.get("/get-areas")
async def get_areas():
    """Get saved areas with enhanced fallback logic (return image dims if available)"""
    try:
        areas = []
        source = "none"
        image_width = None
        image_height = None

        # Try improved picker first (we stored a dict there)
        picker = AreaPicker(save_file="data/warehouse_areas.pkl")
        try:
            saved = picker.get_areas()  # support returning dict saved_payload
            if isinstance(saved, dict) and saved.get("areas"):
                areas = saved.get("areas", [])
                image_width = saved.get("image_width")
                image_height = saved.get("image_height")
                source = "improved_picker"
            elif isinstance(saved, list) and saved:
                areas = saved
                source = "improved_picker_list"
        except Exception:
            # ignore picker errors
            pass

        # legacy file fallback
        if not areas:
            areas_file = os.path.join(settings.temp_dir, "object_positions")
            if os.path.exists(areas_file):
                try:
                    with open(areas_file, "rb") as f:
                        areas = pickle.load(f)
                        source = "legacy_file"
                except Exception:
                    areas = []

        # DB fallback
        if not areas:
            area_def = await db_manager.get_latest_area_definition()
            if area_def:
                areas = area_def.get("areas", [])
                meta = area_def.get("meta") or {}
                image_width = meta.get("image_width") or image_width
                image_height = meta.get("image_height") or image_height
                source = "database"

        # Update global
        global current_areas
        current_areas = areas

        return {
            "areas": areas,
            "count": len(areas),
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "image_width": image_width,
            "image_height": image_height,
            "database_status": db_manager.get_connection_status()
        }
    except Exception as e:
        logger.error(f"Error getting areas: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# --- _broadcast_frame: include areas + metadata into WS payload ---
async def _broadcast_frame(processed_frame, count_data, session_id: str):
    """Encode frame and broadcast to all active WS clients."""
    try:
        # Encode frame ke JPEG → base64
        _, buffer = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # --- Build areas_payload ---
        areas_payload = []
        try:
            # Map detail hasil deteksi
            details_map = {}
            if isinstance(count_data, dict) and "details" in count_data:
                for d in count_data["details"]:
                    # area_id dari process_frame_for_counting sudah dimulai dari 2 (slot)
                    details_map[int(d.get("area_id"))] = d

            for idx, a in enumerate(current_areas, start=1):  # mulai dari 1 biar konsisten
                # Normalisasi area object
                if isinstance(a, dict):
                    pts = a.get("points", [])
                    name = a.get("name", f"Area {idx}")
                else:
                    pts = a
                    name = f"Area {idx}"

                # Area 1 = kapasitas
                if idx == 1:
                    status = "capacity"
                else:
                    detail = details_map.get(idx)
                    status = detail["status"] if detail else "empty"

                areas_payload.append({
                    "area_id": idx,
                    "name": name,
                    "points": pts,
                    "status": status
                })

        except Exception as e:
            logger.exception(f"Failed to build areas_payload: {e}")
            areas_payload = current_areas  # fallback

        # --- Ambil data area + dims kalau ada (legacy) ---
        image_width = None
        image_height = None
        try:
            picker = AreaPicker(save_file="data/warehouse_areas.pkl")
            saved = picker.get_areas()
            if isinstance(saved, dict):
                image_width = saved.get("image_width")
                image_height = saved.get("image_height")
        except Exception:
            pass

        # --- Build payload untuk client ---
        data = {
            "type": "frame_data",
            "session_id": session_id,
            "frame": frame_base64,
            "count_data": count_data,
            "areas": areas_payload,
            "areas_image_width": image_width,
            "areas_image_height": image_height,
            "timestamp": datetime.now().isoformat(),
        }

        # --- Normalize count_data untuk disimpan ke DB ---
        try:
            normalized = dict(count_data) if isinstance(count_data, dict) else {}
            if "total_areas" not in normalized:
                normalized["total_areas"] = len(current_areas)
            if "total_slots" not in normalized and len(current_areas) >= 2:
                normalized["total_slots"] = max(0, len(current_areas) - 1)  # area1 = kapasitas
            normalized["session_id"] = session_id

            total_check = (normalized.get("total_areas", 0) or normalized.get("total_slots", 0))
            if total_check > 0 and not normalized.get("error"):
                try:
                    save_success = await db_manager.save_counting_result(normalized)
                    data["database_saved"] = bool(save_success)
                except Exception as e:
                    data["database_saved"] = False
                    logger.exception(f"DB save failed in _broadcast_frame: {e}")
        except Exception:
            logger.exception("Failed to normalize count_data for DB saving")

        # --- Cache terakhir untuk client baru ---
        global current_frame_b64, last_count_data
        current_frame_b64 = frame_base64
        last_count_data = count_data

        # --- Broadcast ke semua client ---
        await manager.broadcast(json.dumps(data))

    except Exception as e:
        logger.error(f"Broadcast error: {e}")


@app.get("/export-areas")
async def export_areas():
    """Export areas as JSON"""
    try:
        picker = AreaPicker(save_file="data/warehouse_areas.pkl")
        areas_data = picker.export_areas_json()
        
        return {
            "message": "Areas exported successfully",
            "data": areas_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Setting FPS#
@app.post("/import-areas")
async def import_areas(import_data: dict):
    """Import areas from JSON"""
    try:
        picker = AreaPicker(save_file="data/warehouse_areas.pkl")
        success = picker.import_areas_json(import_data)
        
        if success:
            global current_areas
            current_areas = picker.get_areas()
            
            return {
                "message": "Areas imported successfully",
                "count": len(current_areas),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to import areas")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-performance-settings")
async def get_performance_settings():
    return {"settings": performance_settings}
   
@app.post("/update-performance-settings")
async def update_performance_settings(settings: PerformanceSettings):
    global performance_settings
    try:
        performance_settings["detection_interval"] = settings.detection_interval
        performance_settings["target_fps"] = settings.target_fps
        return {"status": "success", "settings": performance_settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/start-rtsp-monitoring")
async def start_rtsp_monitoring(rtsp_config: RTSPConfig, request: Request):
    try:
        global monitoring_mode, monitoring_task

        # Stop any existing stream and HLS
        rtsp_handler.stop_streaming()
        monitoring_mode = "none"
        if monitoring_task and not monitoring_task.done():
            monitoring_task.cancel()
        # stop any prior HLS sessions
        for _, sess in list(monitoring_sessions.items()):
            hls_sid = sess.get("hls_session_id")
            if hls_sid:
                try:
                    hls_handler.stop_hls(hls_sid)
                except Exception:
                    pass
        monitoring_sessions.clear()

        # 1) Start HLS sidecar FIRST so frontend can play even if OpenCV fails
        hls_info = hls_handler.start_hls(rtsp_config.rtsp_url)
        hls_session_id = hls_info["session_id"]
        hls_playlist_url = f"/streams/{hls_session_id}/index.m3u8"
        base = str(request.base_url).rstrip("/")
        hls_playlist_absolute = f"{base}{hls_playlist_url}"

        # Create session container early
        session_id = str(uuid.uuid4())
        monitoring_sessions[session_id] = {
            "rtsp_url": rtsp_config.rtsp_url,
            "started_at": datetime.now().isoformat(),
            "frame_count": 0,
            "mode": "rtsp",
            "hls_session_id": hls_session_id,
            "hls_playlist_url": hls_playlist_url,
            "hls_playlist_absolute": hls_playlist_absolute,
            "started_hls": True,
            "started_rtsp_processing": False,
            "errors": [],
        }

        # 2) Try to connect RTSP via OpenCV (may fail for HEVC sources)
        started_rtsp_processing = False
        try:
            if rtsp_handler.connect(rtsp_config.rtsp_url):
                if rtsp_handler.start_streaming():
                    monitoring_mode = "rtsp"
                    monitoring_task = asyncio.create_task(_rtsp_loop(session_id))
                    started_rtsp_processing = True
                else:
                    monitoring_sessions[session_id]["errors"].append("OpenCV start_streaming() failed")
            else:
                monitoring_sessions[session_id]["errors"].append("OpenCV connect() failed")
        except Exception as conn_err:
            monitoring_sessions[session_id]["errors"].append(f"RTSP processing error: {conn_err}")

        monitoring_sessions[session_id]["started_rtsp_processing"] = started_rtsp_processing

        # 3) Always return 200 if HLS started, even if OpenCV failed
        return {
            "message": "RTSP monitoring initialized",
            "session_id": session_id,
            "rtsp_url": rtsp_config.rtsp_url,
            "hls_session_id": hls_session_id,
            "hls_playlist_url": hls_playlist_url,
            "hls_playlist_absolute": hls_playlist_absolute,
            "started_hls": True,
            "started_rtsp_processing": started_rtsp_processing,
            "errors": monitoring_sessions[session_id]["errors"],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        # If even HLS failed, then it's a real server error
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-file-monitoring")
async def start_file_monitoring(file: UploadFile = File(None), file_path: str | None = None):
    """Start video file monitoring"""
    try:
        global monitoring_mode, monitoring_task
        # stop previous
        rtsp_handler.stop_streaming()
        monitoring_mode = "none"
        if monitoring_task and not monitoring_task.done():
            monitoring_task.cancel()

        # resolve file path
        resolved_path = file_path
        if file and not file_path:
            os.makedirs(settings.upload_dir, exist_ok=True)
            ext = os.path.splitext(file.filename)[1] or ".mp4"
            unique_name = f"{uuid.uuid4()}{ext}"
            resolved_path = os.path.join(settings.upload_dir, unique_name)
            contents = await file.read()
            with open(resolved_path, "wb") as f:
                f.write(contents)

        if not resolved_path or not os.path.exists(resolved_path):
            raise HTTPException(status_code=400, detail="Video file not found")

        # Create monitoring session
        session_id = str(uuid.uuid4())
        monitoring_sessions[session_id] = {
            "file_path": resolved_path,
            "started_at": datetime.now().isoformat(),
            "frame_count": 0,
            "mode": "file",
        }

        monitoring_mode = "file"
        monitoring_task = asyncio.create_task(_file_loop(session_id, resolved_path))

        return {
            "message": "File monitoring started",
            "session_id": session_id,
            "file_path": resolved_path,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop-monitoring")
async def stop_monitoring():
    """Stop monitoring"""
    try:
        global monitoring_mode, monitoring_task
        # stop HLS sessions associated with monitoring
        for _, sess in list(monitoring_sessions.items()):
            hls_sid = sess.get("hls_session_id")
            if hls_sid:
                hls_handler.stop_hls(hls_sid)

        rtsp_handler.stop_streaming()
        monitoring_mode = "none"
        if monitoring_task and not monitoring_task.done():
            monitoring_task.cancel()
        monitoring_sessions.clear()
        return {"message": "Monitoring stopped", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring-status")
async def get_monitoring_status():
    """Get current monitoring status"""
    return {
        "is_monitoring": monitoring_mode != "none",
        "mode": monitoring_mode,
        "active_sessions": len(monitoring_sessions),
        "sessions": monitoring_sessions,
        "hls_sessions": hls_handler.get_status(),
        "last_metrics": last_count_data,
        "timestamp": datetime.now().isoformat(),
    }

@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket):
    """Enhanced WebSocket for real-time monitoring"""
    await manager.connect(websocket)
    session_id = str(uuid.uuid4())
    try:
        await manager.send_personal_message(json.dumps({
            "type": "connection",
            "session_id": session_id,
            "message": "Connected to monitoring",
            "database_status": db_manager.get_connection_status()
        }), websocket)

        # Keep alive while client is connected
        while True:
            await asyncio.sleep(15)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/analytics/history")
async def analytics_history(limit: int = 100):
    """
    Return counting history from database (Supabase preferred, SQLite fallback).
    Normalizes 'details' to JSON list if needed.
    """
    try:
        history = await db_manager.get_counting_history(limit=limit)

        # Normalize Supabase stringified 'details' into JSON
        normalized = []
        for rec in history:
            item = dict(rec)
            details = item.get("details")
            if isinstance(details, str):
                try:
                    item["details"] = json.loads(details)
                except Exception:
                    # keep as is if not valid JSON
                    pass
            normalized.append(item)

        source = "Supabase" if getattr(db_manager, "supabase", None) else "SQLite"
        return {
            "history": normalized,
            "count": len(normalized),
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-rtsp-hls")
async def start_rtsp_hls(request: Request):
    data = await request.json()
    rtsp_url = data.get("rtsp_url")
    if not rtsp_url:
        return JSONResponse({"error": "rtsp_url missing"}, status_code=400)

    session = hls_handler.start_hls(rtsp_url)
    session_id = session["session_id"]

    return {
        "session_id": session_id,
        "playlist_url": f"/streams/{session_id}/index.m3u8"
    }


# Tambahkan untuk serve file HTML
@app.get("/index")
async def get_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/player")
async def get_player():
    return FileResponse(os.path.join(FRONTEND_DIR, "player.html"))

@app.post("/stop-rtsp-hls")
async def stop_rtsp_hls(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    if not session_id:
        return JSONResponse({"error": "session_id missing"}, status_code=400)
    hls_handler.stop_hls(session_id)
    return {"stopped": True}

@app.get("/rtsp-snapshot")
async def rtsp_snapshot(overlay: bool = True):
    """
    Return current RTSP frame as JPEG.
    overlay=True will draw areas and analytics like the monitoring loop.
    """
    try:
        # Prefer current frame from handler
        frame = rtsp_handler.get_current_frame()
        if frame is None:
            # fall back to last broadcast if available
            if current_frame_b64:
                try:
                    img_bytes = base64.b64decode(current_frame_b64)
                    return Response(content=img_bytes, media_type="image/jpeg")
                except Exception:
                    pass
            raise HTTPException(status_code=404, detail="No frame available")

        if overlay and len(current_areas) > 0:
            processed, _count = process_frame_for_counting(frame)
            out = processed
        else:
            out = frame

        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode JPEG")
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def normalize_lighting(gray):
    """Normalisasi pencahayaan adaptif dengan CLAHE + auto gamma - OPTIMIZED"""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    gray_eq = clahe.apply(gray)

    mean_val = np.mean(gray_eq)
    if mean_val < 70:
        gamma = 1.6
    elif mean_val < 100:
        gamma = 1.3
    elif mean_val > 190:
        gamma = 0.6
    elif mean_val > 160:
        gamma = 0.8
    else:
        gamma = 1.0

    corrected = np.array(255 * ((gray_eq / 255.0) ** gamma), dtype="uint8")
    return corrected

def calculate_detection_score_fast(mask_region, gray_region, thresh_region, full_area):
    """Fast detection score calculation - optimized for real-time processing"""
    try:
        mean_brightness = np.mean(gray_region)
        brightness_score = min(mean_brightness / 150.0, 1.0)
        
        white_pixels = np.count_nonzero(thresh_region)
        white_ratio = white_pixels / full_area if full_area > 0 else 0
        
        std_dev = np.std(gray_region)
        texture_score = min(std_dev / 35.0, 1.0)
        
        final_score = (
            brightness_score * 0.45 +
            white_ratio * 0.40 +
            texture_score * 0.15
        )
        
        return final_score, {
            "brightness": brightness_score,
            "white_ratio": white_ratio,
            "texture": texture_score,
            "mean_brightness": mean_brightness,
            "std_dev": std_dev
        }
    except Exception as e:
        print(f"Error calculating detection score: {e}")
        return 0.0, {}

def preprocess_frame_optimized(frame):
    """Optimized preprocessing pipeline for real-time performance"""
    try:
        height, width = frame.shape[:2]
        if width > 1920:
            scale = 1920 / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_norm = normalize_lighting(img_gray)
        img_blur = cv2.GaussianBlur(img_norm, (5, 5), 0)
        
        img_thresh = cv2.adaptiveThreshold(
            img_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 10
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_combined = np.array([0, 0, 70])
        upper_combined = np.array([180, 60, 255])
        mask_hsv = cv2.inRange(img_hsv, lower_combined, upper_combined)
        
        img_thresh = cv2.bitwise_or(img_thresh, mask_hsv)
        
        return frame, img_gray, img_thresh
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return frame, None, None

def process_frame_for_counting(frame):
    """OPTIMIZED: Process frame with caching and adaptive detection"""
    try:
        global current_areas, frame_cache

        if not current_areas or len(current_areas) < 2:
            return frame, {
                "capacity_area": 0,
                "total_slots": 0,
                "empty_slots": 0,
                "occupied_slots": 0,
                "full_slots": 0,
                "details": [],
                "error": "Not enough areas defined"
            }

        use_cached = not should_run_full_detection()
        
        if use_cached and frame_cache["last_detection_results"] is not None:
            overlay = frame.copy()
            count_data = frame_cache["last_detection_results"]
            _draw_cached_overlay(overlay, count_data)
            alpha = 0.6
            result_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            return result_frame, count_data

        processed_frame, img_gray, img_thresh = preprocess_frame_optimized(frame)
        
        if img_gray is None or img_thresh is None:
            return frame, {"error": "Preprocessing failed"}

        overlay = processed_frame.copy()

        capacity_area = current_areas[0]
        try:
            cv2.polylines(
                overlay, [np.array(capacity_area, dtype=np.int32)],
                True, (0, 255, 255), 3
            )
            centroid_cap = tuple(np.mean(capacity_area, axis=0).astype(int))
            cv2.putText(
                overlay, "CAPACITY AREA",
                centroid_cap,
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
            )
        except Exception:
            pass

        empty_slots, filled_slots = 0, 0
        area_details = []
        detection_threshold = 0.28

        for i, polygon in enumerate(current_areas[1:], start=2):
            try:
                pts = np.array(polygon, dtype=np.int32)
                mask = np.zeros(img_thresh.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                x, y, w, h = cv2.boundingRect(pts)
                
                if w < 10 or h < 10:
                    continue
                
                gray_region = img_gray[y:y+h, x:x+w]
                thresh_region = cv2.bitwise_and(img_thresh[y:y+h, x:x+w], mask[y:y+h, x:x+w])
                
                full_area = cv2.countNonZero(mask)
                if full_area == 0:
                    continue

                detection_score, score_details = calculate_detection_score_fast(
                    mask[y:y+h, x:x+w], gray_region, thresh_region, full_area
                )

                if detection_score >= detection_threshold:
                    color = (0, 255, 0)  # GREEN for filled
                    status = "filled"
                    label_text = "Terisi"
                    filled_slots += 1
                else:
                    color = (0, 0, 255)  # RED for empty
                    status = "empty"
                    label_text = "Empty"
                    empty_slots += 1

                cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
                
                temp_overlay = overlay.copy()
                cv2.fillPoly(temp_overlay, [pts], color)
                cv2.addWeighted(temp_overlay, 0.25, overlay, 0.75, 0, overlay)

                centroid = np.mean(pts, axis=0).astype(int)
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = centroid[0] - text_size[0] // 2
                text_y = centroid[1] + text_size[1] // 2
                
                cv2.rectangle(
                    overlay,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0),
                    -1
                )
                
                cv2.putText(
                    overlay, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

                area_details.append({
                    "area_id": i,
                    "status": status,
                    "detection_score": float(detection_score),
                    "score_details": score_details,
                    "centroid": [int(centroid[0]), int(centroid[1])]
                })

            except Exception as e:
                print(f"Error processing slot {i}: {e}")
                continue

        try:
            alpha = 0.6
            result_frame = cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0)
        except Exception as e:
            print(f"Error blending frames: {e}")
            result_frame = processed_frame

        total_areas = len(current_areas)
        total_slots = max(0, len(current_areas) - 1)
        total_pallets = filled_slots * settings.pallet_height
        total_sacks = total_pallets * settings.pallet_capacity
        total_weight = total_sacks * settings.sack_weight / 1000

        try:
            cv2.rectangle(result_frame, (10, 10), (500, 200), (0, 0, 0), -1)
            cv2.rectangle(result_frame, (10, 10), (500, 200), (255, 255, 255), 2)
            
            analytics_text = [
                "=== WAREHOUSE MONITORING ===",
                f"Total Slots: {total_slots}",
                f"Filled: {filled_slots} (Green) | Empty: {empty_slots} (Red)",
                f"Occupancy: {(filled_slots/total_slots*100) if total_slots > 0 else 0:.1f}%",
                "",
                f"Est. Pallets: {total_pallets} | Sacks: {total_sacks}",
                f"Est. Weight: {total_weight:.1f} tons",
                "",
                f"FPS: {1.0/frame_cache['detection_interval']:.1f} (optimized)",
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            ]
            
            for i, text in enumerate(analytics_text):
                y_pos = 30 + (i * 18)
                cv2.putText(
                    result_frame, text, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )
        except Exception as e:
            print(f"Error adding analytics: {e}")

        count_data = {
            "capacity_area": 1,
            "total_areas": total_areas,
            "total_slots": total_slots,
            "empty_slots": empty_slots,
            "filled_slots": filled_slots,
            "occupied_slots": filled_slots,
            "full_slots": 0,
            "estimated_pallets": total_pallets,
            "estimated_sacks": total_sacks,
            "estimated_weight_tons": total_weight,
            "details": area_details,
            "processing_mode": "cached" if use_cached else "full"
        }
        
        frame_cache["last_detection_results"] = count_data
        
        return result_frame, count_data

    except Exception as e:
        print(f"Critical error in frame processing: {e}")
        import traceback
        traceback.print_exc()
        return frame, {
            "capacity_area": 0,
            "total_slots": 0,
            "empty_slots": 0,
            "occupied_slots": 0,
            "full_slots": 0,
            "details": [],
            "error": str(e)
        }

def should_run_full_detection():
    """Determine if we should run full detection or use cached results"""
    global frame_cache
    frame_cache["frame_counter"] += 1
    
    if frame_cache["frame_counter"] >= frame_cache["detection_interval"]:
        frame_cache["frame_counter"] = 0
        return True
    
    return False

def _draw_cached_overlay(overlay, count_data):
    """Quick overlay drawing using cached detection results"""
    try:
        global current_areas
        
        capacity_area = current_areas[0]
        cv2.polylines(
            overlay, [np.array(capacity_area, dtype=np.int32)],
            True, (0, 255, 255), 3
        )
        
        details = count_data.get("details", [])
        details_map = {d["area_id"]: d for d in details}
        
        for i, polygon in enumerate(current_areas[1:], start=2):
            pts = np.array(polygon, dtype=np.int32)
            detail = details_map.get(i, {})
            status = detail.get("status", "empty")
            
            if status in ["filled", "occupied", "full"]:
                color = (0, 255, 0)  # GREEN
                label_text = "Terisi"
            else:
                color = (0, 0, 255)  # RED
                label_text = "Empty"
            
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
            
            temp = overlay.copy()
            cv2.fillPoly(temp, [pts], color)
            cv2.addWeighted(temp, 0.25, overlay, 0.75, 0, overlay)
            
            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(
                overlay, label_text, tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
    except Exception as e:
        print(f"Error drawing cached overlay: {e}")

async def _rtsp_loop(session_id: str):
    """OPTIMIZED: Background loop with adaptive frame rate"""
    try:
        frame_skip_counter = 0
        target_fps = settings.frame_rate
        
        if target_fps > 15:
            frame_skip = 2
        else:
            frame_skip = 1
        
        while monitoring_mode == "rtsp" and rtsp_handler.is_connected():
            frame_skip_counter += 1
            
            if frame_skip_counter % frame_skip != 0:
                await asyncio.sleep(1.0 / target_fps)
                continue
            
            frame = rtsp_handler.get_current_frame()
            if frame is not None:
                processed_frame, count_data = process_frame_for_counting(frame)
                await _broadcast_frame(processed_frame, count_data, session_id)

                if session_id in monitoring_sessions:
                    monitoring_sessions[session_id]["frame_count"] += 1

            await asyncio.sleep(1.0 / target_fps)
            
    except Exception as e:
        logger.error(f"RTSP loop error: {e}")
    finally:
        logger.info("RTSP loop ended")

async def _file_loop(session_id: str, file_path: str):
    """Background loop reading frames from a video file and broadcasting."""
    cap = None
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return

        while monitoring_mode == "file":
            ret, frame = cap.read()
            if not ret:
                break  # end of file
            processed_frame, count_data = process_frame_for_counting(frame)
            await _broadcast_frame(processed_frame, count_data, session_id)

            if session_id in monitoring_sessions:
                monitoring_sessions[session_id]["frame_count"] += 1

            await asyncio.sleep(1.0 / settings.frame_rate)
    except Exception as e:
        logger.error(f"File loop error: {e}")
    finally:
        if cap:
            cap.release()
        logger.info("File loop ended")
            
# Removed duplicate normalize_lighting, calculate_detection_score, preprocess_frame_optimized, process_frame_for_counting, should_run_full_detection, _draw_cached_overlay, _rtsp_loop functions.
# Keeping the optimized versions as per the updates.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
