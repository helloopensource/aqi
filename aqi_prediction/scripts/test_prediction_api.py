#!/usr/bin/env python
"""
Script to test the prediction API with regression models.
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import date

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import application modules
from src.models.air_quality import AQParam, AQScenario, get_default_scenarios
from src.models.aqi_app import AQIApp
from src.models.model_trainer import ModelTrainer
from src.api.models import PredictionRequest
from src.api.endpoints import predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def create_mock_request(scenario_name="los-angeles_pm25"):
    """Create a mock request for testing."""
    return PredictionRequest(
        scenario_name=scenario_name,
        prediction_date=date.today(),
        weather_data={
            "TEMP": 75.0,
            "TEMP_ATTRIBUTES": 0,
            "DEWP": 60.0,
            "DEWP_ATTRIBUTES": 0,
            "SLP": 1015.0,
            "SLP_ATTRIBUTES": 0,
            "STP": 1013.0,
            "STP_ATTRIBUTES": 0,
            "VISIB": 10.0,
            "VISIB_ATTRIBUTES": 0,
            "WDSP": 5.0,
            "WDSP_ATTRIBUTES": 0,
            "MXSPD": 10.0,
            "GUST": 15.0,
            "MAX": 85.0,
            "MAX_ATTRIBUTES": 0,
            "MIN": 65.0,
            "MIN_ATTRIBUTES": 0,
            "PRCP": 0.0,
            "PRCP_ATTRIBUTES": "0",
            "SNDP": 0.0,
            "FRSHTT": "000000",
            "MONTH": 5,
            "DAYOFWEEK": 2,
            "SEASON": "Spring",
            "TEMP_RANGE": 20.0,
            "TEMP_AVG": 75.0,
            "TEMP_DEWP_DIFF": 15.0,
            "WDSP_TEMP": 375.0,
            "isUnhealthy": 0
        }
    )

async def test_predict_endpoint():
    """Test the predict endpoint."""
    logger.info("Testing predict endpoint with regression model")
    
    # Create request
    request = create_mock_request("los-angeles_pm25")
    
    try:
        # Call the predict endpoint directly
        from fastapi import BackgroundTasks
        background_tasks = BackgroundTasks()
        response = await predict(request, background_tasks)
        
        # Print the response
        logger.info(f"Predict endpoint response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error testing predict endpoint: {str(e)}")
        return None

def test_model_trainer():
    """Test the ModelTrainer directly."""
    logger.info("Testing ModelTrainer with regression model")
    
    # Set up
    scenarios = get_default_scenarios()
    scenario = scenarios["los-angeles_pm25"]
    trainer = ModelTrainer(scenario, "pm25_value", is_regression=True)
    
    # Create test data
    import pandas as pd
    data = pd.DataFrame({
        "TEMP": [75.0],
        "TEMP_ATTRIBUTES": [0],
        "DEWP": [60.0],
        "DEWP_ATTRIBUTES": [0],
        "SLP": [1015.0],
        "SLP_ATTRIBUTES": [0],
        "STP": [1013.0],
        "STP_ATTRIBUTES": [0],
        "VISIB": [10.0],
        "VISIB_ATTRIBUTES": [0],
        "WDSP": [5.0],
        "WDSP_ATTRIBUTES": [0],
        "MXSPD": [10.0],
        "GUST": [15.0],
        "MAX": [85.0],
        "MAX_ATTRIBUTES": [0],
        "MIN": [65.0],
        "MIN_ATTRIBUTES": [0],
        "PRCP": [0.0],
        "PRCP_ATTRIBUTES": ["0"],
        "SNDP": [0.0],
        "FRSHTT": ["000000"],
        "MONTH": [5],
        "DAYOFWEEK": [2],
        "SEASON": ["Spring"],
        "TEMP_RANGE": [20.0],
        "TEMP_AVG": [75.0],
        "TEMP_DEWP_DIFF": [15.0],
        "WDSP_TEMP": [375.0],
        "isUnhealthy": [0]
    })
    
    # Make prediction
    try:
        result = trainer.predict(data)
        logger.info(f"Prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Running tests")
    result = test_model_trainer()
    
    if result is not None and not result.empty:
        logger.info("ModelTrainer test passed")
    else:
        logger.error("ModelTrainer test failed")
        
    # We can't run the async test directly
    logger.info("To test the predict endpoint, you'll need to run an async test or use the API directly") 