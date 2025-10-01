from supabase import create_client, Client
import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        # Initialize database connections
        self.supabase: Optional[Client] = None
        self.sqlite_db = "data/warehouse_stock.db"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Initialize connections
        self.init_supabase()
        self.init_sqlite()
    
    def init_supabase(self):
        """Initialize Supabase client if credentials are available"""
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_ANON_KEY")
            
            if supabase_url and supabase_key:
                self.supabase = create_client(supabase_url, supabase_key)
                logger.info("✅ Supabase connected successfully")
                return True
            else:
                logger.warning("⚠️ Supabase credentials not found. Using SQLite fallback.")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            return False
    
    def init_sqlite(self):
        """Initialize SQLite database as fallback"""
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS counting_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    empty_count INTEGER DEFAULT 0,
                    occupied_count INTEGER DEFAULT 0,
                    total_areas INTEGER DEFAULT 0,
                    estimated_pallets INTEGER DEFAULT 0,
                    estimated_sacks INTEGER DEFAULT 0,
                    estimated_weight_tons REAL DEFAULT 0,
                    details TEXT,
                    session_id TEXT DEFAULT 'default'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS area_definitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    areas TEXT NOT NULL,
                    image_path TEXT,
                    meta TEXT,
                    created_at TEXT NOT NULL,
                    area_count INTEGER DEFAULT 0
                )
            ''')
            # 🔥 Extra check: if meta column is missing (old DB), add it
            cursor.execute("PRAGMA table_info(area_definitions)")
            columns = [col[1] for col in cursor.fetchall()]
            if "meta" not in columns:
                cursor.execute("ALTER TABLE area_definitions ADD COLUMN meta TEXT")
                logger.info("⚡ Added missing column 'meta' to area_definitions")
            
            conn.commit()
            conn.close()
            logger.info("✅ SQLite database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQLite: {e}")
            return False
    
    async def save_counting_result(self, result_data: Dict[str, Any]) -> bool:
        timestamp = datetime.now().isoformat()

        data = {
            "timestamp": timestamp,
            "empty_count": result_data.get("empty_count", 0),
            "occupied_count": result_data.get("occupied_count", 0),
            "total_areas": result_data.get("total_areas", 0),
            "estimated_pallets": result_data.get("estimated_pallets", 0),
            "estimated_sacks": result_data.get("estimated_sacks", 0),
            "estimated_weight_tons": result_data.get("estimated_weight_tons", 0),
            "details": json.dumps(result_data.get("details", [])),
            "session_id": result_data.get("session_id", "default"),
        }

        # Try Supabase first
        if self.supabase:
            try:
                response = self.supabase.table("counting_results").insert(data).execute()
                if hasattr(response, "data") and response.data:
                    logger.info(f"✅ Saved to Supabase: {response.data}")
                    return True
                else:
                    logger.warning("⚠️ Insert to Supabase returned no data, fallback to SQLite")
            except Exception as e:
                logger.error(f"❌ Supabase save failed: {e}")

        # Fallback to SQLite
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()

            cursor.execute(
                '''
                INSERT INTO counting_results 
                (timestamp, empty_count, occupied_count, total_areas, estimated_pallets, 
                 estimated_sacks, estimated_weight_tons, details, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    timestamp,
                    data["empty_count"],
                    data["occupied_count"],
                    data["total_areas"],
                    data["estimated_pallets"],
                    data["estimated_sacks"],
                    data["estimated_weight_tons"],
                    data["details"],
                    data["session_id"],
                ),
            )

            conn.commit()
            conn.close()
            logger.info("✅ Saved to SQLite")
            return True

        except Exception as e:
            logger.error(f"❌ SQLite save failed: {e}")
            return False

    
    async def get_counting_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get counting history from database"""
        # Try Supabase first
        if self.supabase:
            try:
                response = self.supabase.table("counting_results")\
                    .select("*")\
                    .order("timestamp", desc=True)\
                    .limit(limit)\
                    .execute()
                
                if response.data:
                    logger.info(f"✅ Retrieved {len(response.data)} records from Supabase")
                    return response.data
            except Exception as e:
                logger.error(f"❌ Supabase query failed: {e}")
        
        # Fallback to SQLite
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM counting_results 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            result = []
            for row in rows:
                record = dict(zip(columns, row))
                # Parse JSON details
                if record.get('details'):
                    try:
                        record['details'] = json.loads(record['details'])
                    except:
                        record['details'] = []
                result.append(record)
            
            conn.close()
            logger.info(f"✅ Retrieved {len(result)} records from SQLite")
            return result
            
        except Exception as e:
            logger.error(f"❌ SQLite query failed: {e}")
            return []
    
    async def save_area_definition(self, areas: List[List[tuple]], image_path: str = None, meta: Optional[Dict[str, Any]] = None) -> bool:
        """Save area definitions to database"""
        timestamp = datetime.now().isoformat()
        areas_json = json.dumps(areas)
        meta_json = json.dumps(meta) if meta else None
        
        # Try Supabase first
        if self.supabase:
            try:
                data = {
                    "areas": areas_json,
                    "image_path": image_path,
                    "meta": meta_json,
                    "created_at": timestamp,
                    "area_count": len(areas)
                }
                
                response = self.supabase.table("area_definitions").insert(data).execute()
                if response.data:
                    logger.info("✅ Area definition saved to Supabase")
                    return True
            except Exception as e:
                logger.error(f"❌ Supabase area save failed: {e}")
        
        # Fallback to SQLite
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO area_definitions (areas, image_path, meta, created_at, area_count)
                VALUES (?, ?, ?, ?, ?)
            ''', (areas_json, image_path, meta_json, timestamp, len(areas)))

            conn.commit()
            conn.close()
            logger.info("✅ Area definition saved to SQLite")
            return True
            
        except Exception as e:
            logger.error(f"❌ SQLite area save failed: {e}")
            return False
    
    async def get_latest_area_definition(self) -> Optional[Dict[str, Any]]:
        """Get the latest area definition"""
        # Try Supabase first
        if self.supabase:
            try:
                response = self.supabase.table("area_definitions")\
                    .select("*")\
                    .order("created_at", desc=True)\
                    .limit(1)\
                    .execute()
                
                if response.data:
                    area_def = response.data[0]
                    area_def["areas"] = json.loads(area_def["areas"])
                    logger.info("✅ Retrieved area definition from Supabase")
                    return area_def
            except Exception as e:
                logger.error(f"❌ Supabase area query failed: {e}")
        
        # Fallback to SQLite
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM area_definitions 
                ORDER BY created_at DESC 
                LIMIT 1
            ''')
            
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                area_def = dict(zip(columns, row))
                area_def["areas"] = json.loads(area_def["areas"])
                if area_def.get("meta"):
                    try:
                        area_def["meta"] = json.loads(area_def["meta"])
                    except:
                        area_def["meta"] = None
                conn.close()
                logger.info("✅ Retrieved area definition from SQLite")
                return area_def
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"❌ SQLite area query failed: {e}")
            return None
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get database connection status"""
        return {
            "supabase_connected": self.supabase is not None,
            "sqlite_available": os.path.exists(self.sqlite_db),
            "primary_database": "Supabase" if self.supabase else "SQLite"
        }

# Global database manager instance
db_manager = DatabaseManager()
