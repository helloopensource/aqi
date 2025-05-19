#!/usr/bin/env python
"""
Script to make predictions using a trained model.
Usage: python predict.py --scenario los-angeles_pm25 --weather-file weather_data.csv
"""
import os
import sys
import argparse
import logging
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import application modules
from src.models.air_quality import AQParam, AQScenario, get_default_scenarios
from src.models.aqi_app import AQIApp
from src.models.model_trainer import ModelTrainer
from src.utils.aqi_calculator import calculate_aqi, get_category_from_aqi
from src.config.settings import DEFAULT_ML_TARGET_LABEL, UNHEALTHY_THRESHOLDS

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
    parser = argparse.ArgumentParser(description="Make predictions using a trained model")
    parser.add_argument(
        "--scenario", 
        type=str, 
        default="los-angeles_pm25",
        help="Scenario to use for prediction"
    )
    parser.add_argument(
        "--weather-file", 
        type=str, 
        help="CSV file with weather data (must include all required features)"
    )
    parser.add_argument(
        "--weather-json", 
        type=str, 
        help="JSON string with weather data (alternative to weather-file)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file for predictions (default: stdout)"
    )
    return parser.parse_args()


def load_weather_data(args):
    """Load weather data from file or JSON string."""
    if args.weather_file:
        if not os.path.exists(args.weather_file):
            logger.error(f"Weather file not found: {args.weather_file}")
            return None
        
        try:
            weather_df = pd.read_csv(args.weather_file)
            logger.info(f"Loaded weather data from {args.weather_file}: {len(weather_df)} rows")
            return weather_df
        except Exception as e:
            logger.error(f"Error loading weather file: {str(e)}")
            return None
    
    elif args.weather_json:
        try:
            weather_data = json.loads(args.weather_json)
            weather_df = pd.DataFrame([weather_data])
            logger.info("Loaded weather data from JSON string")
            return weather_df
        except Exception as e:
            logger.error(f"Error parsing weather JSON: {str(e)}")
            return None
    
    else:
        # Example weather data
        logger.info("No weather data provided, using example data")
        weather_data = {
            # Original data
            "DEWP": 50.5,
            "WDSP": 5.2,
            "MAX": 78.3,
            "MIN": 62.1,
            "PRCP": 0.0,
            "MONTH": 6,
            "DAYOFWEEK": 3,
            "TEMP_RANGE": 16.2,
            "TEMP_AVG": 70.2,
            "TEMP_DEWP_DIFF": 19.7,
            "WDSP_TEMP": 365.04,
            
            # Missing columns
            "TEMP": 70.2,                # Same as TEMP_AVG
            "TEMP_ATTRIBUTES": 0,        # Default attribute values
            "DEWP_ATTRIBUTES": 0,
            "SLP": 1013.0,              # Standard sea level pressure in hPa
            "SLP_ATTRIBUTES": 0,
            "STP": 1013.0,              # Standard station pressure in hPa
            "STP_ATTRIBUTES": 0,
            "VISIB": 10.0,              # Visibility in miles
            "WDSP_ATTRIBUTES": 0,
            "MXSPD": 10.0,              # Max sustained wind speed in knots
            "GUST": 0.0,                # Wind gust in knots
            "MAX_ATTRIBUTES": "Valid",  # String categorical
            "MIN_ATTRIBUTES": "Valid",  # String categorical
            "PRCP_ATTRIBUTES": "Valid",  # String categorical
            "FRSHTT": "000000",         # Flags for weather conditions
            "SEASON": "Summer"          # String categorical based on MONTH
        }
        return pd.DataFrame([weather_data])


def main():
    """Main function to make predictions."""
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
    
    # Load weather data
    weather_df = load_weather_data(args)
    if weather_df is None:
        return 1
    
    # Make predictions
    try:
        trainer = ModelTrainer(scenario, app.ml_target_label)
        
        # Check if model exists
        predictor = trainer.load_model()
        if predictor is None:
            logger.error("Model not found. Train the model first.")
            return 1
        
        # Handle feature types in a more robust way
        try:
            # Different versions of AutoGluon have different ways of accessing feature metadata
            if hasattr(predictor, 'feature_metadata') and hasattr(predictor.feature_metadata, 'get_type_map_raw'):
                features_info = predictor.feature_metadata.get_type_map_raw()
                # Convert categorical columns
                for feature, dtype in features_info.items():
                    if feature in weather_df.columns and 'category' in str(dtype).lower():
                        weather_df[feature] = weather_df[feature].astype('category')
            elif hasattr(predictor, 'feature_metadata') and hasattr(predictor.feature_metadata, 'type_map_raw'):
                features_info = predictor.feature_metadata.type_map_raw
                # Convert categorical columns
                for feature, dtype in features_info.items():
                    if feature in weather_df.columns and 'category' in str(dtype).lower():
                        weather_df[feature] = weather_df[feature].astype('category')
            elif hasattr(predictor, 'load_metadata') and hasattr(predictor, 'transform_features'):
                # Some versions use a preprocessor for feature transformation
                logger.info("Using predictor's built-in transformation pipeline")
                # No manual conversions needed
            else:
                logger.warning("Could not determine feature types, continuing with raw features")
        except Exception as e:
            logger.warning(f"Error handling feature types: {str(e)}. Continuing with raw features.")
        
        # Make predictions
        logger.info("Making predictions...")
        result = trainer.predict(weather_df)
        
        # Process and display results
        predictions = []
        
        for i, row in result.iterrows():
            prediction = row["prediction"]
            is_unhealthy = bool(prediction)
            
            # Get probability if available
            probability = None
            if "probability_unhealthy" in row:
                probability = float(row["probability_unhealthy"])
            
            # Calculate AQI value based on probability and parameter
            param_name = scenario.aq_param_target.name
            
            # Simplified approach: derive a concentration estimate from probability
            aqi_value = None
            category = None
            
            if probability is not None:
                # Calculate a more reasonable concentration based on the probability
                # Use actual unhealthy thresholds from the config
                if param_name == "pm25":
                    # For PM2.5, unhealthy threshold is 35.5 µg/m³
                    unhealthy_threshold = UNHEALTHY_THRESHOLDS["pm25"]
                    if is_unhealthy:
                        # If predicted unhealthy, concentration should be above threshold
                        estimated_conc = unhealthy_threshold + (probability * 50)
                    else:
                        # If predicted healthy, concentration should be below threshold
                        estimated_conc = probability * unhealthy_threshold * 0.8
                elif param_name == "pm10":
                    # For PM10, unhealthy threshold is 155 µg/m³
                    unhealthy_threshold = UNHEALTHY_THRESHOLDS["pm10"]
                    if is_unhealthy:
                        # If predicted unhealthy, concentration should be above threshold
                        estimated_conc = unhealthy_threshold + (probability * 100)
                    else:
                        # If predicted healthy, concentration should be below threshold
                        estimated_conc = probability * unhealthy_threshold * 0.8
                elif param_name == "o3":
                    # For O3, unhealthy threshold is 0.070 ppm
                    unhealthy_threshold = UNHEALTHY_THRESHOLDS["o3"]
                    if is_unhealthy:
                        estimated_conc = unhealthy_threshold + (probability * 0.05)
                    else:
                        estimated_conc = probability * unhealthy_threshold * 0.8
                elif param_name == "no2":
                    # For NO2, unhealthy threshold is 0.100 ppm
                    unhealthy_threshold = UNHEALTHY_THRESHOLDS["no2"]
                    if is_unhealthy:
                        estimated_conc = unhealthy_threshold + (probability * 0.05)
                    else:
                        estimated_conc = probability * unhealthy_threshold * 0.8
                elif param_name == "so2":
                    # For SO2, unhealthy threshold is 0.075 ppm
                    unhealthy_threshold = UNHEALTHY_THRESHOLDS["so2"]
                    if is_unhealthy:
                        estimated_conc = unhealthy_threshold + (probability * 0.05)
                    else:
                        estimated_conc = probability * unhealthy_threshold * 0.8
                elif param_name == "co":
                    # For CO, unhealthy threshold is 9.5 ppm
                    unhealthy_threshold = UNHEALTHY_THRESHOLDS["co"]
                    if is_unhealthy:
                        estimated_conc = unhealthy_threshold + (probability * 5)
                    else:
                        estimated_conc = probability * unhealthy_threshold * 0.8
                else:
                    # Default conservative estimate
                    estimated_conc = probability * 20
                
                # Ensure logical consistency - cap maximum concentration if not unhealthy
                if not is_unhealthy and estimated_conc > scenario.unhealthy_threshold:
                    estimated_conc = scenario.unhealthy_threshold * 0.9
                
                logger.info(f"Estimated {param_name} concentration: {estimated_conc:.2f} based on probability {probability:.4f}")
                
                # Calculate AQI
                aqi_value, category = calculate_aqi(estimated_conc, param_name)
                
                # Ensure consistency between prediction and AQI
                if not is_unhealthy and aqi_value > 100:
                    logger.warning(f"Inconsistency detected: prediction is healthy but AQI is {aqi_value}. Adjusting...")
                    # Recalculate with a lower concentration
                    estimated_conc = scenario.unhealthy_threshold * 0.7
                    aqi_value, category = calculate_aqi(estimated_conc, param_name)
            
            # Create prediction result
            prediction_result = {
                "scenario": scenario.name,
                "date": date.today().isoformat(),
                "is_unhealthy": is_unhealthy,
                "probability": probability,
                "aqi_value": aqi_value,
                "category": category["name"] if category else None,
                "health_implications": category["health_implications"] if category else None,
                "prediction_time": datetime.now().isoformat()
            }
            
            predictions.append(prediction_result)
        
        # Output results
        output_json = json.dumps(predictions, indent=2)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
            logger.info(f"Predictions saved to {args.output}")
        else:
            print(output_json)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 