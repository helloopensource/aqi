"""
Main entry point for the AQI prediction application.
"""
import os
import logging
import uvicorn
from dotenv import load_dotenv

from src.config.settings import API_PORT, API_HOST

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the API server."""
    logger.info("Starting AQI Prediction API server")
    
    # Load environment or default settings
    port = int(os.environ.get("API_PORT", API_PORT))
    host = os.environ.get("API_HOST", API_HOST)
    
    # Start Uvicorn server
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main() 