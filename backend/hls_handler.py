import os
import uuid
import shutil
import subprocess
import threading
import time
import platform
from typing import Dict, Optional


class HLSHandler:
    def __init__(self, base_dir: str, ffmpeg_path: str = "ffmpeg"):
        self.base_dir = base_dir
        self.ffmpeg_path = ffmpeg_path
        os.makedirs(self.base_dir, exist_ok=True)
        self.processes: Dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()

    def _session_dir(self, session_id: str) -> str:
        return os.path.join(self.base_dir, session_id)

    def start_hls(
        self,
        rtsp_url: str,
        session_id: Optional[str] = None,
        segment_time: int = 2,
        list_size: int = 6,
        reencode: bool = True,
        scale_even: bool = True,
        extra_args: Optional[list] = None,
        debug: bool = False,
    ) -> Dict[str, str]:
        """
        Start an ffmpeg process that reads RTSP and writes HLS playlist+segments on disk.
        Returns dict with session_id and playlist_path.
        """
        with self.lock:
            sid = session_id or str(uuid.uuid4())
            out_dir = self._session_dir(sid)
            os.makedirs(out_dir, exist_ok=True)

            # Stop old process if exists
            if sid in self.processes:
                self.stop_hls(sid)

            output = os.path.join(out_dir, "index.m3u8")

            # Build ffmpeg command
            cmd = [
                self.ffmpeg_path,
                "-rtsp_transport", "tcp",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
            ]

            # Tambahkan timeout hanya kalau bukan Windows
            if not platform.system().lower().startswith("win"):
                cmd += [
                    "-stimeout", "5000000",
                    "-rw_timeout", "5000000",
                ]

            cmd += [
                "-i", rtsp_url,
                "-an",  # no audio
            ]

            if reencode:
                # Transcode ke H.264 (kompatibel untuk semua device)
                cmd += [
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-tune", "zerolatency",
                    "-pix_fmt", "yuv420p",
                    "-profile:v", "baseline",
                    "-level", "3.1",
                    "-g", str(max(1, list_size * 2)),
                    "-keyint_min", "1",
                ]
            else:
                # Copy stream langsung
                cmd += ["-c:v", "copy"]

            if scale_even:
                # pastikan width/height genap
                cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]

            cmd += [
                "-f", "hls",
                "-hls_time", str(segment_time),
                "-hls_list_size", str(list_size),
                "-hls_flags", "delete_segments+append_list+independent_segments",
                "-hls_segment_type", "mpegts",
                output,  # path absolut
            ]

            if extra_args:
                cmd += extra_args

            # Start ffmpeg
            if debug:
                log_file = os.path.join(out_dir, "ffmpeg.log")
                log_f = open(log_file, "w", encoding="utf-8")
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
            else:
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            self.processes[sid] = proc

        # Wait for playlist muncul (max 10 detik)
        for _ in range(50):  # 50 x 0.2s = 10s
            if os.path.exists(output) and os.path.getsize(output) > 0:
                break
            time.sleep(0.2)

        return {"session_id": sid, "playlist_path": output}

    def stop_hls(self, session_id: str, cleanup: bool = True) -> bool:
        with self.lock:
            proc = self.processes.pop(session_id, None)
        if proc:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass

        if cleanup:
            try:
                shutil.rmtree(self._session_dir(session_id), ignore_errors=True)
            except Exception:
                pass
        return True

    def is_running(self, session_id: str) -> bool:
        with self.lock:
            proc = self.processes.get(session_id)
        if not proc:
            return False
        return proc.poll() is None

    def get_status(self) -> Dict[str, dict]:
        with self.lock:
            out = {}
            for sid, proc in self.processes.items():
                out[sid] = {
                    "running": proc.poll() is None,
                    "dir": self._session_dir(sid),
                }
            return out

    def list_streams(self) -> Dict[str, dict]:
        with self.lock:
            out: Dict[str, dict] = {}
            for sid, proc in self.processes.items():
                out[sid] = {
                    "running": proc.poll() is None,
                    "dir": self._session_dir(sid),
                    "playlist": f"/streams/{sid}/index.m3u8",
                }
            return out


# singleton accessor used by FastAPI app
def get_hls_handler():
    from config import settings
    base_dir = settings.hls_dir
    ffmpeg_path = getattr(settings, "ffmpeg_path", "ffmpeg")

    os.makedirs(base_dir, exist_ok=True)
    global _singleton
    try:
        _singleton
    except NameError:
        _singleton = HLSHandler(base_dir=base_dir, ffmpeg_path=ffmpeg_path)
    return _singleton


# convenience symbol
hls_handler = get_hls_handler()
