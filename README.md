# 📦 Warehouse Stock Counting System

A computer vision–based system for monitoring and counting fertilizer pallet stock in warehouses using RTSP cameras and real-time analytics.

---

## ✨ Features

- **Area Definition** – Define pallet areas via image upload or RTSP stream  
- **Real-time Monitoring** – Track stock levels from RTSP cameras or video files  
- **Stock Analytics** – Calculate empty/occupied areas with statistics  
- **Web Interface** – User-friendly frontend with Streamlit  
- **REST API** – Backend powered by FastAPI  
- **Database Integration** – Supabase support for persistent data storage  

---

## 🏗️ Project Structure

```
warehouse-stock-counting/
├── backend/                 # FastAPI backend
│   ├── main.py              # Main API server
│   └── models.py            # Pydantic models
├── frontend/                # Streamlit frontend
│   ├── app.py               # Main frontend app
│   └── components/          # UI components
├── utils/                   # Utility modules
│   └── video_processor.py   # Video processing utilities
├── scripts/                 # Run scripts
│   ├── run_backend.py       # Start backend server
│   └── run_frontend.py      # Start frontend app
├── data/                    # Data directory
├── tmp/                     # Temporary files
└── requirements.txt         # Python dependencies
```

---

## ⚙️ Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create necessary directories:
   ```bash
   mkdir -p data tmp
   ```

---

## 🚀 Usage

### Start the Backend Server
```bash
python scripts/run_backend.py
```
- API available at: `http://localhost:8000`  
- Docs: `http://localhost:8000/docs`

### Start the Frontend
```bash
python scripts/run_frontend.py
```
- Web interface: `http://localhost:8501`

### Workflow
1. **Define Areas** – Upload warehouse top-view image → draw pallet areas → save.  
2. **Monitor Stock** – Select video/RTSP stream → start monitoring → view stock count.  
3. **Analytics** – Check historical data, trends, and export reports.  

---

## 📡 API Endpoints

- `POST /upload-image` → Upload warehouse image  
- `POST /save-areas` → Save defined areas  
- `GET /get-areas` → Retrieve saved areas  
- `POST /start-rtsp-monitoring` → Start RTSP monitoring  
- `POST /stop-monitoring` → Stop monitoring  
- `WebSocket /ws/monitoring` → Real-time monitoring data  

---

## 🔧 Configuration

### RTSP Stream
```
rtsp://username:password@ip:port/stream
```

### Stock Parameters (constants in code)
- `PALLET_HEIGHT = 4` → Layers per pallet  
- `PALLET_CAPACITY = 20` → Sacks per pallet  
- `SACK_WEIGHT = 50` → Kg per sack  

---

## 🛠️ Troubleshooting

- **File path errors** → Ensure `data/` and `tmp/` exist  
- **RTSP issues** → Check camera URL & network connectivity  
- **OpenCV errors** → Validate image/video formats  

**Fixes included in this repo**:  
✔ Absolute paths instead of relative  
✔ Better error handling for missing files  
✔ Organized directory structure  
✔ Exception handling improvements  

---

## 🗄️ Database (Supabase)

1. Create a Supabase project  
2. Add environment variables in `.env`  
3. Implement database models & queries in backend  

---

## 🤝 Contributing

1. Fork this repo  
2. Create a feature branch  
3. Commit your changes  
4. Submit a Pull Request  

---

## 📄 License

MIT License © 2025
