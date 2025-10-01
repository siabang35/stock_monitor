# Makefile for Warehouse Stock Counting System

.PHONY: help install run-backend run-frontend run-dev clean test docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  run-backend  - Run FastAPI backend"
	@echo "  run-frontend - Run Streamlit frontend"
	@echo "  run-dev      - Run both backend and frontend"
	@echo "  clean        - Clean temporary files"
	@echo "  test         - Run tests"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-run   - Run with Docker Compose"

# Install dependencies
install:
	pip install -r requirements.txt
	mkdir -p data tmp

# Run backend only
run-backend:
	python scripts/run_backend.py

# Run frontend only
run-frontend:
	python scripts/run_frontend.py

# Run development environment
run-dev:
	python run.py

# Clean temporary files
clean:
	rm -rf tmp/*
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Run tests (placeholder)
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Docker commands
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Database setup
setup-db:
	python scripts/run_sql_script.py scripts/create_tables.sql

# Development setup
setup-dev: install setup-db
	@echo "Development environment setup complete!"
	@echo "Run 'make run-dev' to start the application"
