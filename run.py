#!/usr/bin/env python3
"""
Main runner script for the Warehouse Stock Counting System
"""
import subprocess
import sys
import threading
import time
import os

def run_backend():
    """Run the FastAPI backend"""
    print("Starting backend server...")
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)
    subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

def run_frontend():
    """Run the Streamlit frontend"""
    print("Starting frontend...")
    time.sleep(3)  # Wait for backend to start
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    os.chdir(frontend_dir)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])

def main():
    print("🚀 Starting Warehouse Stock Counting System")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("tmp", exist_ok=True)
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Start frontend in main thread
        run_frontend()
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
