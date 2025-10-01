from fastapi import APIRouter, HTTPException, Request, Body
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import os

from .hls_handler import hls_handler

router = APIRouter(prefix="/hls", tags=["HLS"])


class StartHLSRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP URL of the camera")
    session_id: Optional[str] = Field(None, description="Custom session id. If omitted, auto-generated")
    segment_time: int = 2
    list_size: int = 6
    reencode: bool = True
    scale_even: bool = True


@router.post("/start")
def start_hls(req: StartHLSRequest, request: Request):
    try:
        result = hls_handler.start_hls(
            rtsp_url=req.rtsp_url,
            session_id=req.session_id,
            segment_time=req.segment_time,
            list_size=req.list_size,
            reencode=req.reencode,
            scale_even=req.scale_even,
            debug=True,
        )
        sid = result["session_id"]
        m3u8_url = f"/streams/{sid}/index.m3u8"
        base = str(request.base_url).rstrip("/")
        m3u8_url_absolute = f"{base}{m3u8_url}"
        return {
            "ok": True,
            "message": "HLS started",
            "session_id": sid,
            "playlist_url": m3u8_url,
            "playlist_url_absolute": m3u8_url_absolute,
            "segment_time": req.segment_time,
            "list_size": req.list_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StopHLSRequest(BaseModel):
    session_id: str


@router.post("/stop")
def stop_hls(req: StopHLSRequest):
    if not hls_handler.stop_hls(req.session_id):
        raise HTTPException(status_code=404, detail="Stream not found or already stopped")
    return {"ok": True, "message": "HLS stopped", "session_id": req.session_id}


@router.get("/status")
def status(session_id: str):
    return {"session_id": session_id, "running": hls_handler.is_running(session_id)}


@router.get("/list")
def list_streams():
    return hls_handler.list_streams()
