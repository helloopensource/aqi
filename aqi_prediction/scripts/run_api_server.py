#!/usr/bin/env python
"""
Script to run the FastAPI server.
"""
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import application modules
import uvicorn
from src.api.app import app
from src.config.settings import API_HOST, API_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(f"Starting FastAPI server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info") 