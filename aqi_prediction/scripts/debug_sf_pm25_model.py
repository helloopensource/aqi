#!/usr/bin/env python
"""
Script to debug the San Francisco PM2.5 regression model prediction issue.
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import application modules
from src.models.air_quality import AQScenario, get_default_scenarios
from src.models.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create a sample data point for prediction testing."""
    # Create a dataframe with one row of sample weather data
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
    return data

def test_model_prediction():
    """Test the San Francisco PM2.5 regression model prediction."""
    # Get scenarios
    scenarios = get_default_scenarios()
    if "san-francisco_pm25" not in scenarios:
        logger.error("San Francisco PM2.5 scenario not found")
        return False
    
    scenario = scenarios["san-francisco_pm25"]
    logger.info(f"Testing prediction for scenario: {scenario.name}")
    
    # Create model trainer for regression
    trainer = ModelTrainer(scenario, "pm25_value", is_regression=True)
    
    # Create sample data
    data = create_sample_data()
    logger.info(f"Sample data shape: {data.shape}")
    
    # Time the model loading
    logger.info("Loading model...")
    start_time = time.time()
    model = trainer.load_model()
    load_time = time.time() - start_time
    logger.info(f"Model loading time: {load_time:.2f} seconds")
    
    if model is None:
        logger.error("Could not load model")
        return False
    
    # Get model features
    if hasattr(model, 'features'):
        model_features = model.features()
        logger.info(f"Model features: {model_features}")
        
        # Check for missing features in sample data
        missing_features = [f for f in model_features if f not in data.columns and f != "pm25_value"]
        if missing_features:
            logger.warning(f"Missing features in sample data: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                logger.info(f"Adding missing feature {feature} with default value 0")
                data[feature] = 0
    
    # Try prediction with timeout
    logger.info("Making prediction...")
    try:
        start_time = time.time()
        # Set a timeout for the prediction
        import signal
        
        class TimeoutError(Exception):
            pass
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Prediction timed out")
        
        # Set the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        # Make the prediction
        result = trainer.predict(data)
        
        # Cancel the timeout
        signal.alarm(0)
        
        prediction_time = time.time() - start_time
        logger.info(f"Prediction time: {prediction_time:.2f} seconds")
        
        if result.empty:
            logger.error("Prediction result is empty")
            return False
        
        logger.info(f"Prediction result columns: {result.columns.tolist()}")
        logger.info(f"Prediction result: {result.iloc[0].to_dict()}")
        return True
        
    except TimeoutError:
        logger.error("Prediction timed out after 30 seconds")
        return False
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("Starting debug script for San Francisco PM2.5 model")
    success = test_model_prediction()
    
    if success:
        logger.info("Debug completed successfully")
        sys.exit(0)
    else:
        logger.error("Debug failed")
        sys.exit(1) 