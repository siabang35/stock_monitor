# Multi-stage Dockerfile for Warehouse Stock Counting System
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data tmp

# Expose ports
EXPOSE 8000 8501

# Backend stage
FROM base as backend
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend stage
FROM base as frontend
CMD ["python", "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# Development stage (default)
FROM base as development
CMD ["python", "run.py"]
