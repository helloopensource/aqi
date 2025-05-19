#!/usr/bin/env python
"""
Script to train a model for a specific scenario.
Usage: python train_model.py --scenario los-angeles_pm25 --time-limit 900
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import application modules
from src.models.air_quality import AQParam, AQScenario, get_default_scenarios
from src.models.aqi_app import AQIApp
from src.models.model_trainer import ModelTrainer
from src.config.settings import DEFAULT_ML_TARGET_LABEL, DATA_DIR, MODEL_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an AQI prediction model")
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="los-angeles_pm25",
        help="Scenario to train model for"
    )
    parser.add_argument( 
        "--metric", type=str, default="accuracy", help="Metric to optimize" 
    )
    parser.add_argument(
        "--time-limit", 
        type=int, 
        default=900,
        help="Time limit for training in seconds"
    )
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Test size for training"
    )
    parser.add_argument(
        "--validation-size", 
        type=float, 
        default=0.2,
        help="Validation size for training"
    )
    return parser.parse_args()


def main():
    """Main function to train a model."""
    args = parse_args()
    
    # Ensure data directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Create app and load default parameters and scenarios
    app = AQIApp()
    default_params = AQParam.get_default_params()
    default_scenarios = get_default_scenarios()
    
    # Add params and scenarios to app
    for param in default_params.values():
        app.add_aq_param(param)
    
    for scenario in default_scenarios.values():
        app.add_aq_scenario(scenario)
    
    # Select scenario
    if args.scenario not in app.aq_scenarios:
        logger.error(f"Scenario '{args.scenario}' not found")
        logger.info(f"Available scenarios: {list(app.aq_scenarios.keys())}")
        return 1
    
    app.select_scenario(args.scenario)
    scenario = app.aq_scenarios[args.scenario]
    
    logger.info(f"Training model for scenario: {scenario.name}")
    logger.info(f"Time limit: {args.time_limit} seconds")
    
    # Get data
    try:
        logger.info("Retrieving NOAA GSOD data...")
        noaa_df = app.get_noaa_data()
        
        logger.info("Retrieving OpenAQ data...")
        openaq_df = app.get_openaq_data()
        
        if noaa_df.empty or openaq_df.empty:
            logger.error("Failed to retrieve data")
            return 1
        
        logger.info("Merging data...")
        merged_df = app.get_merged_data(noaa_df, openaq_df)
        
        if merged_df.empty:
            logger.error("Failed to merge data")
            return 1
        
        logger.info("Preparing data...")
        train_df, val_df, test_df = app.prepare_train_test_data(
            merged_df,
            test_size=args.test_size,
            validation_size=args.validation_size
        )
        
        logger.info("Training model...")
        trainer = ModelTrainer(scenario, app.ml_target_label)
        predictor = trainer.train_model(train_df, val_df, args.time_limit, eval_metric=args.metric)
        
        # Display model info
        model_info = trainer.get_model_info()
        logger.info(f"Model training completed: {model_info}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 