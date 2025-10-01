import cv2
import asyncio
import threading
from typing import Optional, Callable
import numpy as np
from datetime import datetime

class RTSPHandler:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.rtsp_url = None
        self.frame_callback: Optional[Callable] = None
        self.current_frame = None
        self.thread = None
        
    def set_frame_callback(self, callback: Callable):
        """Set callback function for frame processing"""
        self.frame_callback = callback
    
    def connect(self, rtsp_url: str) -> bool:
        """Connect to RTSP stream"""
        try:
            self.rtsp_url = rtsp_url
            self.cap = cv2.VideoCapture(rtsp_url)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try an initial read, but don't fail hard if first frame can't decode yet
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                return True
            # allow a short warm-up for HEVC streams
            warm_ok = False
            for _ in range(10):
                ret2, frame2 = self.cap.read()
                if ret2:
                    self.current_frame = frame2
                    warm_ok = True
                    break
                threading.Event().wait(0.1)
            if warm_ok:
                return True

            # if still not ok, release
            self.cap.release()
            self.cap = None
            return False
        except Exception as e:
            print(f"Error connecting to RTSP: {e}")
            return False
    
    def start_streaming(self):
        """Start streaming in a separate thread"""
        if self.cap and not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            return True
        return False
    
    def stop_streaming(self):
        """Stop streaming"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _stream_loop(self):
        """Main streaming loop"""
        while self.is_running and self.cap:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame
                    
                    # Call frame callback if set
                    if self.frame_callback:
                        self.frame_callback(frame)
                else:
                    # Try to reconnect
                    print("Lost RTSP connection, attempting to reconnect...")
                    self.cap.release()
                    if not self.connect(self.rtsp_url):
                        print("Failed to reconnect to RTSP stream")
                        break
                
                # Small delay to control frame rate
                threading.Event().wait(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Error in RTSP stream loop: {e}")
                break
        
        self.is_running = False
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame"""
        return self.current_frame
    
    def is_connected(self) -> bool:
        """Check if RTSP stream is connected"""
        return self.cap is not None and self.is_running

# Global RTSP handler instance
rtsp_handler = RTSPHandler()
