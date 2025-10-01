# Warehouse Stock Counting System

A comprehensive system for counting fertilizer pallet stock in warehouses using computer vision and real-time monitoring.

## Features

- **Area Definition**: Define pallet areas using image upload or RTSP stream
- **Real-time Monitoring**: Monitor stock levels via RTSP cameras or video files
- **Stock Analytics**: Calculate empty/occupied areas with detailed statistics
- **Web Interface**: User-friendly Streamlit frontend
- **REST API**: FastAPI backend for integration
- **Database Integration**: Supabase integration for data storage

## Architecture

\`\`\`
warehouse-stock-counting/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API server
│   └── models.py           # Pydantic models
├── frontend/               # Streamlit frontend
│   ├── app.py              # Main frontend app
│   └── components/         # UI components
├── utils/                  # Utility modules
│   └── video_processor.py  # Video processing utilities
├── scripts/                # Run scripts
│   ├── run_backend.py      # Start backend server
│   └── run_frontend.py     # Start frontend app
├── data/                   # Data directory
├── tmp/                    # Temporary files
└── requirements.txt        # Python dependencies
\`\`\`

## Installation

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Create necessary directories:
\`\`\`bash
mkdir -p data tmp
\`\`\`

## Usage

### Start the Backend Server
\`\`\`bash
python scripts/run_backend.py
\`\`\`
The API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

### Start the Frontend
\`\`\`bash
python scripts/run_frontend.py
\`\`\`
The web interface will be available at `http://localhost:8501`

### Using the System

1. **Area Definition**:
   - Upload a top-view image of your warehouse
   - Define rectangular areas for each pallet location
   - Save the defined areas

2. **Stock Monitoring**:
   - Choose between video file or RTSP stream monitoring
   - Start monitoring to see real-time stock counts
   - View analytics and statistics

3. **Analytics**:
   - View historical data and trends
   - Export reports and statistics

## API Endpoints

- `POST /upload-image` - Upload warehouse image
- `POST /save-areas` - Save defined areas
- `GET /get-areas` - Retrieve saved areas
- `POST /start-rtsp-monitoring` - Start RTSP monitoring
- `POST /stop-monitoring` - Stop monitoring
- `WebSocket /ws/monitoring` - Real-time monitoring data

## Configuration

### RTSP Stream
Configure your RTSP camera URL in the format:
\`\`\`
rtsp://username:password@ip:port/stream
\`\`\`

### Stock Calculation Parameters
Modify these constants in the code as needed:
- `PALLET_HEIGHT = 4` - Number of pallet layers
- `PALLET_CAPACITY = 20` - Sacks per pallet
- `SACK_WEIGHT = 50` - Weight per sack (kg)

## Troubleshooting

### Common Issues

1. **File path errors**: Ensure `data/` and `tmp/` directories exist
2. **RTSP connection issues**: Check camera URL and network connectivity
3. **OpenCV errors**: Ensure proper image/video file formats

### Error Fixes from Original Code

- Fixed relative path issues by using absolute paths
- Added proper error handling for missing files
- Improved directory structure and file organization
- Added proper exception handling throughout the application

## Database Integration

The system is prepared for Supabase integration. To enable:

1. Set up Supabase project
2. Add environment variables for database connection
3. Implement database models and operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
