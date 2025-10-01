# from fastapi import FastAPI
# from .routes_hls import router as hls_router, mount_hls_static
# from .rtsp_handler import rtsp_handler
#
# app = FastAPI()
#
# # HLS: mount static and include router
# app.include_router(hls_router)
# mount_hls_static(app, mount_path="/streams", dir_path="hls")
#
# # Example: start your monitoring endpoint can also trigger HLS
# # @app.post("/monitoring/start")
# # def start_monitoring(rtsp_url: str):
# #     # 1) Start your RTSP frame loop (callbacks for overlays remain as-is)
# #     connected = rtsp_handler.connect(rtsp_url)
# #     if not connected:
# #         return {"ok": False, "error": "Failed to connect RTSP"}
# #     rtsp_handler.start_streaming()
# #
# #     # 2) Start HLS sidecar (returns playlist URL)
# #     from .hls_handler import hls_handler
# #     playlist_url = hls_handler.start_hls(rtsp_url=rtsp_url)
# #
# #     return {"ok": True, "hls_playlist_url": playlist_url}
#
# # Frontend (Streamlit) can play HLS via hls.js:
# # st.components.v1.html(f'''
# # <video id="video" controls autoplay muted playsinline style="width:100%;max-height:70vh;"></video>
# # <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
# # <script>
# #   const url = "{playlist_url_from_backend}";
# #   const video = document.getElementById("video");
# #   if (Hls.isSupported()) {{
# #     const hls = new Hls({{ maxLatency: 3, liveSyncDuration: 2 }});
# #     hls.loadSource(url);
# #     hls.attachMedia(video);
# #   }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
# #     video.src = url;
# #   }}
# # </script>
# # ''', height=460)
