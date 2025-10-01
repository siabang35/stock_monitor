from pydantic import BaseModel
from typing import List, Tuple, Optional
from datetime import datetime

class Point(BaseModel):
    x: int
    y: int

class Area(BaseModel):
    id: Optional[int] = None
    points: List[Tuple[int, int]]
    name: Optional[str] = None

class AreasData(BaseModel):
    areas: List[List[Tuple[int, int]]]
    image_path: Optional[str] = None

class RTSPConfig(BaseModel):
    rtsp_url: str
    
class CountingResult(BaseModel):
    timestamp: datetime
    empty_count: int
    occupied_count: int
    total_areas: int
    details: List[dict]
