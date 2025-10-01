import streamlit as st
import streamlit.components.v1 as components

def render_hls_player(playlist_url: str, width: int = 960, height: int = 540, autoplay: bool = True, muted: bool = True):
    if not playlist_url:
        st.warning("HLS playlist URL is empty.")
        return
    auto = "autoplay" if autoplay else ""
    mute = "muted" if muted else ""
    html = f"""
    <div style="position:relative;width:{width}px;height:{height}px;background:#000;">
      <video id="hlsVideo" {auto} {mute} controls playsinline
             style="width:100%;height:100%;object-fit:contain;background:#000;"></video>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <script>
      (function() {{
        const url = "{playlist_url}";
        const video = document.getElementById("hlsVideo");
        if (window.Hls && window.Hls.isSupported()) {{
          const hls = new Hls({{ maxBufferLength: 10, liveSyncDuration: 2 }});
          hls.loadSource(url);
          hls.attachMedia(video);
          hls.on(Hls.Events.MANIFEST_PARSED, function() {{
            try {{ video.play(); }} catch(e) {{}}
          }});
        }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
          video.src = url;
          video.addEventListener('loadedmetadata', function() {{
            try {{ video.play(); }} catch(e) {{}}
          }});
        }} else {{
          video.outerHTML = "<div style='color:white;padding:12px'>Browser tidak mendukung HLS.</div>";
        }}
      }})();
    </script>
    """
    components.html(html, height=height + 10, scrolling=False)
