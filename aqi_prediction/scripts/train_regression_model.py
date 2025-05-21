#!/usr/bin/env python
"""
Script to train a regression model for a specific scenario to predict actual pollutant values.
Usage: python train_regression_model.py --scenario los-angeles_pm25 --target pm25_value --time-limit 900
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import application modules
from src.models.air_quality import AQParam, AQScenario, get_default_scenarios
from src.models.aqi_app import AQIApp
from src.models.model_trainer import ModelTrainer
from src.config.settings import (
    DATA_DIR, MODEL_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    REGRESSION_TARGET_LABELS, DEFAULT_REGRESSION_EVAL_METRIC,
    UNHEALTHY_THRESHOLDS
)

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
    parser = argparse.ArgumentParser(description="Train an AQI regression model to predict actual values")
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="los-angeles_pm25",
        help="Scenario to train model for"
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=REGRESSION_TARGET_LABELS,
        default="pm25_value",
        help=f"Target column to predict"
    )
    parser.add_argument( 
        "--metric", 
        type=str, 
        default=DEFAULT_REGRESSION_EVAL_METRIC, 
        help="Metric to optimize" 
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
    """Main function to train a regression model."""
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
    
    logger.info(f"Training regression model for scenario: {scenario.name}")
    logger.info(f"Predicting target: {args.target}")
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
        
        # For regression models, we need to ensure the target column exists
        # We'll use the value column from OpenAQ data
        if args.target not in merged_df.columns:
            param_name = args.target.split('_')[0]  # Extract param name (e.g., pm25, pm10, o3)
            logger.info(f"Creating target column '{args.target}' from '{param_name}' values")
            
            # In OpenAQ, the air quality value is stored in the 'avg' column
            if 'avg' in merged_df.columns:
                logger.info(f"Using 'avg' column as the source for {args.target}")
                merged_df[args.target] = merged_df['avg']
            # If the OpenAQ parameter column exists, use that
            elif param_name in merged_df.columns:
                logger.info(f"Using '{param_name}' column as the source for {args.target}")
                merged_df[args.target] = merged_df[param_name]
            else:
                # Try case-insensitive match or variants
                possible_columns = [col for col in merged_df.columns if param_name.lower() in col.lower()]
                
                if possible_columns:
                    source_col = possible_columns[0]
                    logger.info(f"Using '{source_col}' column as the source for {args.target}")
                    merged_df[args.target] = merged_df[source_col]
                elif 'isUnhealthy' in merged_df.columns:
                    # Create synthetic values based on classification and thresholds
                    logger.warning(f"No source column found for {args.target}. Creating synthetic values based on isUnhealthy classification.")
                    
                    # Get the unhealthy threshold for this parameter
                    unhealthy_threshold = UNHEALTHY_THRESHOLDS.get(param_name, 35.5)  # Default to PM2.5 if not found
                    
                    # Create synthetic values
                    # Healthy values (random values below threshold)
                    healthy_values = np.random.uniform(
                        low=unhealthy_threshold * 0.2,  # 20% of threshold as min
                        high=unhealthy_threshold * 0.9,  # 90% of threshold as max
                        size=sum(merged_df['isUnhealthy'] == 0)
                    )
                    
                    # Unhealthy values (random values above threshold)
                    unhealthy_values = np.random.uniform(
                        low=unhealthy_threshold * 1.1,  # 110% of threshold as min
                        high=unhealthy_threshold * 3.0,  # 300% of threshold as max
                        size=sum(merged_df['isUnhealthy'] == 1)
                    )
                    
                    # Combine values into a Series
                    merged_df[args.target] = np.nan
                    merged_df.loc[merged_df['isUnhealthy'] == 0, args.target] = healthy_values
                    merged_df.loc[merged_df['isUnhealthy'] == 1, args.target] = unhealthy_values
                    
                    logger.info(f"Created synthetic {args.target} values based on isUnhealthy classification")
                    
                    # Check if arrays are not empty before calling min/max
                    if len(healthy_values) > 0:
                        logger.info(f"Healthy range: {healthy_values.min():.2f} to {healthy_values.max():.2f}")
                    else:
                        logger.warning("No healthy samples found in the dataset")
                        
                    if len(unhealthy_values) > 0:
                        logger.info(f"Unhealthy range: {unhealthy_values.min():.2f} to {unhealthy_values.max():.2f}")
                    else:
                        logger.warning("No unhealthy samples found in the dataset")
                else:
                    logger.error(f"Could not find source column for {args.target}. Available columns: {list(merged_df.columns)}")
                    return 1
        
        logger.info("Preparing data...")
        train_df, val_df, test_df = app.prepare_train_test_data(
            merged_df,
            test_size=args.test_size,
            validation_size=args.validation_size
        )
        
        logger.info("Training regression model...")
        trainer = ModelTrainer(scenario, args.target, is_regression=True)
        predictor = trainer.train_model(
            train_df, 
            val_df, 
            args.time_limit, 
            problem_type="regression",
            eval_metric=args.metric
        )
        
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