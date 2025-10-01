import asyncio
import json
import threading
import time
from typing import Callable, Optional

import streamlit as st

try:
    import websockets  # ensure 'websockets' package is installed
except Exception as e:
    st.error(f"Missing dependency 'websockets': {e}")
    websockets = None  # type: ignore

class AsyncWebSocketClient:
    def __init__(self, url: str):
        self.url = url
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.message_callback: Optional[Callable[[dict], None]] = None
        self.connected = False

    def set_message_callback(self, callback: Callable[[dict], None]):
        self.message_callback = callback

    async def _receiver(self):
        # Connect and receive messages forever until stop
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.url, max_size=8 * 1024 * 1024) as ws:  # 8MB
                    self.connected = True
                    backoff = 1.0
                    while not self._stop_event.is_set():
                        msg = await ws.recv()
                        try:
                            data = json.loads(msg)
                        except Exception:
                            data = {"type": "raw", "payload": msg}
                        if self.message_callback:
                            self.message_callback(data)
                self.connected = False
            except Exception as e:
                # transient failure, retry with backoff
                self.connected = False
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

    def start(self):
        if websockets is None:
            return False
        if self.thread and self.thread.is_alive():
            return True
        self._stop_event.clear()

        def _run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_until_complete(self._receiver())
            finally:
                try:
                    self.loop.run_until_complete(asyncio.sleep(0.05))
                except Exception:
                    pass
                self.loop.close()

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        # small wait for connection attempt
        time.sleep(0.3)
        return True

    def stop(self):
        self._stop_event.set()
        if self.loop and self.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(asyncio.sleep(0.01), self.loop)
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=2.0)
        self.thread = None
        self.loop = None
        self.connected = False


_ws_client: Optional[AsyncWebSocketClient] = None

def get_websocket_client(url: str) -> AsyncWebSocketClient:
    global _ws_client
    if _ws_client and _ws_client.url != url:
        # stop old one if URL changed
        _ws_client.stop()
        _ws_client = None
    if _ws_client is None:
        _ws_client = AsyncWebSocketClient(url)
    return _ws_client

def start_monitoring_websocket(callback: Callable[[dict], None], backend_base_url: str):
    # Build absolute ws url (ws or wss depending on backend_base_url)
    scheme = "wss" if backend_base_url.startswith("https") else "ws"
    ws_url = f"{scheme}://{backend_base_url.split('://', 1)[1].rstrip('/')}/ws/monitoring"
    client = get_websocket_client(ws_url)
    client.set_message_callback(callback)
    return client.start()

def stop_monitoring_websocket():
    global _ws_client
    if _ws_client:
        _ws_client.stop()
        _ws_client = None
