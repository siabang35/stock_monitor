import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    api_title: str = "Warehouse Stock Counting API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database Settings
    supabase_url: Optional[str] = None
    supabase_anon_key: Optional[str] = None
    
    # File Storage Settings
    upload_dir: str = "data"
    temp_dir: str = "tmp"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    # Video Processing Settings
    empty_threshold: float = 0.15
    frame_rate: int = 30
    
    # Stock Calculation Settings
    pallet_height: int = 4
    pallet_capacity: int = 20
    sack_weight: int = 50  # kg

    # HLS Settings
    hls_dir: str = os.path.join("tmp", "hls")   # 🔑 semua pakai folder ini
    ffmpeg_path: str = "ffmpeg"                 # bisa diubah via .env

    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }

# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.temp_dir, exist_ok=True)
os.makedirs(settings.hls_dir, exist_ok=True)  # 🔑 pastikan hls dir ada
