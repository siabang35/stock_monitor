#!/usr/bin/env python3
"""
Script to run the Streamlit frontend
"""
import subprocess
import sys
import os

def main():
    # Change to frontend directory
    frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
    os.chdir(frontend_dir)
    
    # Run Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
    
    print("Starting Streamlit frontend...")
    print(f"Command: {' '.join(cmd)}")
    print("Frontend will be available at: http://localhost:8501")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nFrontend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
