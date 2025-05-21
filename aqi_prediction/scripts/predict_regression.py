#!/usr/bin/env python
"""
Script to make predictions using a trained regression model.
Usage: python predict_regression.py --scenario los-angeles_pm25 --target pm25_value
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import application modules
from src.models.air_quality import AQParam, AQScenario, get_default_scenarios
from src.models.aqi_app import AQIApp
from src.models.model_trainer import ModelTrainer
from src.config.settings import REGRESSION_TARGET_LABELS

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
    parser = argparse.ArgumentParser(description="Make predictions with a trained regression model")
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="los-angeles_pm25",
        help="Scenario to use model for"
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=REGRESSION_TARGET_LABELS,
        default="pm25_value",
        help=f"Target column that was predicted"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default=None,
        help="Path to input CSV file with features (optional)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save prediction results CSV (optional)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of recent data to predict on if no input data provided"
    )
    return parser.parse_args()


def main():
    """Main function to make predictions with a regression model."""
    args = parse_args()
    
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
    
    logger.info(f"Making predictions for scenario: {scenario.name}")
    logger.info(f"Using regression model for target: {args.target}")
    
    # Create model trainer for the regression model
    trainer = ModelTrainer(scenario, args.target, is_regression=True)
    
    try:
        # Load model
        model = trainer.load_model()
        if model is None:
            logger.error(f"No model found for {scenario.name} with target {args.target}")
            return 1
        
        # Get input data
        if args.input_data and os.path.exists(args.input_data):
            logger.info(f"Loading input data from {args.input_data}")
            input_df = pd.read_csv(args.input_data)
        else:
            logger.info(f"Retrieving recent data for the past {args.days} days")
            # Get recent NOAA GSOD data
            noaa_df = app.get_noaa_data()
            if noaa_df.empty:
                logger.error("Failed to retrieve NOAA data")
                return 1
            
            # Filter for only the most recent days
            if 'DATE' in noaa_df.columns:
                noaa_df['DATE'] = pd.to_datetime(noaa_df['DATE'])
                cutoff_date = pd.to_datetime('today') - pd.Timedelta(days=args.days)
                recent_df = noaa_df[noaa_df['DATE'] >= cutoff_date]
                
                # If no recent data is available, use the most recent 30 days of data from the dataset
                if recent_df.empty:
                    logger.warning(f"No data available in the past {args.days} days. Using the most recent 30 days from the dataset.")
                    # Sort by date and take the most recent 30 days
                    noaa_df = noaa_df.sort_values('DATE', ascending=False).head(30)
                    logger.info(f"Using {len(noaa_df)} rows of the most recent data available")
                else:
                    noaa_df = recent_df
                    logger.info(f"Filtered data to {len(noaa_df)} rows from the past {args.days} days")
            
            # For prediction, we don't need OpenAQ data since we're predicting it
            # But we do need a properly formatted DataFrame with all necessary features
            try:
                input_df = app.prepare_features(noaa_df)
                
                # Add the isUnhealthy column which is required by the model
                if 'isUnhealthy' not in input_df.columns:
                    logger.info("Adding required 'isUnhealthy' column with default value 0")
                    input_df['isUnhealthy'] = 0  # Default to healthy
                
                # Handle missing categorical features - ensure SEASON is a string not a float/int
                if 'SEASON' in input_df.columns and not isinstance(input_df['SEASON'].iloc[0], str):
                    season_map = {
                        1: 'Winter', 2: 'Winter', 3: 'Spring', 
                        4: 'Spring', 5: 'Spring', 6: 'Summer',
                        7: 'Summer', 8: 'Summer', 9: 'Fall', 
                        10: 'Fall', 11: 'Fall', 12: 'Winter'
                    }
                    input_df['SEASON'] = input_df['SEASON'].map(season_map)
                    
                # Handle attribute columns that should be strings
                for col in input_df.columns:
                    if col.endswith('_ATTRIBUTES') and not isinstance(input_df[col].iloc[0], str):
                        input_df[col] = input_df[col].astype(str)
                
            except AttributeError:
                # If prepare_features doesn't exist, create a simple version here
                logger.info("Using basic feature preparation")
                input_df = noaa_df.copy()
                
                # Add derived features if they don't exist
                if 'TEMP_RANGE' not in input_df.columns and 'MAX' in input_df.columns and 'MIN' in input_df.columns:
                    input_df['TEMP_RANGE'] = input_df['MAX'] - input_df['MIN']
                
                if 'TEMP_AVG' not in input_df.columns and 'MAX' in input_df.columns and 'MIN' in input_df.columns:
                    input_df['TEMP_AVG'] = (input_df['MAX'] + input_df['MIN']) / 2
                
                if 'TEMP_DEWP_DIFF' not in input_df.columns and 'TEMP_AVG' in input_df.columns and 'DEWP' in input_df.columns:
                    input_df['TEMP_DEWP_DIFF'] = input_df['TEMP_AVG'] - input_df['DEWP']
                
                if 'WDSP_TEMP' not in input_df.columns and 'WDSP' in input_df.columns and 'TEMP_AVG' in input_df.columns:
                    input_df['WDSP_TEMP'] = input_df['WDSP'] * input_df['TEMP_AVG']
                
                # Add month, day of week if not present
                if 'DATE' in input_df.columns and 'MONTH' not in input_df.columns:
                    if input_df['DATE'].dtype != 'datetime64[ns]':
                        input_df['DATE'] = pd.to_datetime(input_df['DATE'])
                    input_df['MONTH'] = input_df['DATE'].dt.month
                    input_df['DAYOFWEEK'] = input_df['DATE'].dt.dayofweek
                
                # Add season if not present
                if 'SEASON' not in input_df.columns and 'MONTH' in input_df.columns:
                    season_map = {
                        1: 'Winter', 2: 'Winter', 3: 'Spring', 
                        4: 'Spring', 5: 'Spring', 6: 'Summer',
                        7: 'Summer', 8: 'Summer', 9: 'Fall', 
                        10: 'Fall', 11: 'Fall', 12: 'Winter'
                    }
                    input_df['SEASON'] = input_df['MONTH'].map(season_map)
                
                # Add the isUnhealthy column which is required by the model
                if 'isUnhealthy' not in input_df.columns:
                    logger.info("Adding required 'isUnhealthy' column with default value 0")
                    input_df['isUnhealthy'] = 0  # Default to healthy
        
        logger.info(f"Making predictions on {len(input_df)} data points")
        
        # Make predictions
        prediction_results = trainer.predict(input_df)
        
        if prediction_results.empty:
            logger.error("Failed to make predictions")
            return 1
        
        # Add date and location information to predictions if available
        if 'DATE' in input_df.columns:
            prediction_results['DATE'] = input_df['DATE']
        
        # Display prediction summary
        logger.info("\nPrediction Summary:")
        logger.info(f"Total predictions: {len(prediction_results)}")
        
        for col in prediction_results.columns:
            if col.endswith('_predicted') or col.endswith('_lower_bound') or col.endswith('_upper_bound'):
                logger.info(f"{col} - Mean: {prediction_results[col].mean():.2f}, Min: {prediction_results[col].min():.2f}, Max: {prediction_results[col].max():.2f}")
        
        # Save results to file if requested
        if args.output_file:
            output_path = args.output_file
            logger.info(f"Saving predictions to {output_path}")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            prediction_results.to_csv(output_path, index=False)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 