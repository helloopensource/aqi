#!/usr/bin/env python
"""
Quick test script for regression prediction.
Usage: python quick_predict_test.py <scenario_name> <target>
Example: python quick_predict_test.py san-francisco_pm25 pm25_value
"""
import os
import sys
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import application modules
from src.models.air_quality import AQParam, AQScenario, get_default_scenarios
from src.models.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "san-francisco_pm25"
    target = sys.argv[2] if len(sys.argv) > 2 else "pm25_value"
    
    logger.info(f"Testing prediction for scenario: {scenario_name}, target: {target}")
    
    # Get scenarios
    scenarios = get_default_scenarios()
    if scenario_name not in scenarios:
        logger.error(f"Scenario '{scenario_name}' not found")
        return 1
    
    scenario = scenarios[scenario_name]
    
    # Create model trainer
    trainer = ModelTrainer(scenario, target, is_regression=True)
    
    # Create sample data
    data = pd.DataFrame({
        "TEMP": [75.0],
        "TEMP_ATTRIBUTES": ["0"],
        "DEWP": [60.0],
        "DEWP_ATTRIBUTES": ["0"],
        "SLP": [1015.0],
        "SLP_ATTRIBUTES": ["0"],
        "STP": [1013.0],
        "STP_ATTRIBUTES": ["0"],
        "VISIB": [10.0],
        "VISIB_ATTRIBUTES": ["0"],
        "WDSP": [5.0],
        "WDSP_ATTRIBUTES": ["0"],
        "MXSPD": [10.0],
        "GUST": [15.0],
        "MAX": [85.0],
        "MAX_ATTRIBUTES": ["0"],
        "MIN": [65.0],
        "MIN_ATTRIBUTES": ["0"],
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
    result = trainer.predict(data)
    
    if result.empty:
        logger.error("Prediction failed")
        return 1
    
    logger.info(f"Prediction result: {result.iloc[0].to_dict()}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 