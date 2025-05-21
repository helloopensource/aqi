"""
FastAPI endpoints for the AQI prediction service.
"""
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, date

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
import asyncio

from ..models.air_quality import AQParam, AQScenario, get_default_scenarios
from ..models.aqi_app import AQIApp
from ..models.model_trainer import ModelTrainer
from ..utils.aqi_calculator import calculate_aqi, get_category_from_aqi, is_unhealthy
from ..config.settings import REGRESSION_TARGET_LABELS
from .models import (
    AQParamModel, AQScenarioModel, HealthResponse, ErrorResponse,
    PredictionRequest, PredictionResponse, ModelInfoResponse,
    ScenarioListResponse, AQICategory, AQIResponse
)

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Create app and load default parameters and scenarios
app = AQIApp()
default_params = AQParam.get_default_params()
default_scenarios = get_default_scenarios()

# Add params and scenarios to app
for param in default_params.values():
    app.add_aq_param(param)

for scenario in default_scenarios.values():
    app.add_aq_scenario(scenario)

# If no scenario selected, select the first one
if app.selected_scenario is None and len(app.aq_scenarios) > 0:
    app.selected_scenario = app.aq_scenarios[next(iter(app.aq_scenarios))]


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now()
    }


@router.get("/params", response_model=List[AQParamModel], tags=["Parameters"])
async def get_parameters():
    """Get all available air quality parameters."""
    return [
        AQParamModel(
            id=param.id,
            name=param.name,
            unit=param.unit,
            unhealthy_threshold_default=param.unhealthy_threshold_default,
            desc=param.desc
        )
        for param in app.aq_params.values()
    ]


@router.get("/scenarios", response_model=ScenarioListResponse, tags=["Scenarios"])
async def get_scenarios():
    """Get all available scenarios."""
    scenario_models = []
    
    for scenario in app.aq_scenarios.values():
        scenario_models.append(
            AQScenarioModel(
                location=scenario.location,
                name=scenario.name,
                noaa_station_id=scenario.noaa_station_id,
                noaa_station_lat=scenario.noaa_station_lat,
                noaa_station_lng=scenario.noaa_station_lng,
                open_aq_sensor_ids=scenario.open_aq_sensor_ids,
                unhealthy_threshold=scenario.unhealthy_threshold,
                year_start=scenario.year_start,
                year_end=scenario.year_end,
                aq_radius_miles=scenario.aq_radius_miles,
                target_param=scenario.aq_param_target.name,
                target_param_desc=scenario.aq_param_target.desc
            )
        )
    
    return {"scenarios": scenario_models}


@router.get("/scenarios/{scenario_name}", response_model=AQScenarioModel, tags=["Scenarios"])
async def get_scenario(scenario_name: str):
    """Get a specific scenario by name."""
    if scenario_name not in app.aq_scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_name}' not found")
    
    scenario = app.aq_scenarios[scenario_name]
    
    return AQScenarioModel(
        location=scenario.location,
        name=scenario.name,
        noaa_station_id=scenario.noaa_station_id,
        noaa_station_lat=scenario.noaa_station_lat,
        noaa_station_lng=scenario.noaa_station_lng,
        open_aq_sensor_ids=scenario.open_aq_sensor_ids,
        unhealthy_threshold=scenario.unhealthy_threshold,
        year_start=scenario.year_start,
        year_end=scenario.year_end,
        aq_radius_miles=scenario.aq_radius_miles,
        target_param=scenario.aq_param_target.name,
        target_param_desc=scenario.aq_param_target.desc
    )


@router.get("/models/{scenario_name}", response_model=ModelInfoResponse, tags=["Models"])
async def get_model_info(scenario_name: str, background_tasks: BackgroundTasks):
    """Get information about a trained model."""
    if scenario_name not in app.aq_scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_name}' not found")
    
    scenario = app.aq_scenarios[scenario_name]
    trainer = ModelTrainer(scenario, app.ml_target_label)
    
    try:
        # Run model info retrieval in a thread pool with timeout
        try:
            model_info = await asyncio.wait_for(
                run_in_threadpool(trainer.get_model_info),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Getting model info timed out")
        
        if "error" in model_info:
            if model_info["error"] == "Model not found":
                raise HTTPException(status_code=404, detail="Model not found")
            else:
                raise HTTPException(status_code=500, detail=model_info["error"])
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make a prediction for the given scenario and weather data."""
    if request.scenario_name not in app.aq_scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{request.scenario_name}' not found")
    
    scenario = app.aq_scenarios[request.scenario_name]
    trainer = ModelTrainer(scenario, app.ml_target_label)
    
    # Check if model exists
    try:
        logger.info("Loading model...")
        # Check if we should use regression model - look for target values with _value suffix
        regression_target = None
        for target in REGRESSION_TARGET_LABELS:
            if target.startswith(scenario.aq_param_target.name):
                regression_target = target
                break
        
        # Initialize ModelTrainer with regression flag if target found
        if regression_target:
            logger.info(f"Using regression model with target: {regression_target}")
            trainer = ModelTrainer(scenario, regression_target, is_regression=True)
        else:
            # Fall back to classification model
            logger.info(f"Using classification model with target: {app.ml_target_label}")
            trainer = ModelTrainer(scenario, app.ml_target_label)
        
        predictor = await asyncio.wait_for(
            run_in_threadpool(trainer.load_model),
            timeout=10.0  # 10 second timeout for loading
        )
        if predictor is None:
            raise HTTPException(status_code=404, detail="Model not found. Train the model first.")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Model loading timed out")
    
    # Create DataFrame from weather data
    try:
        logger.info("Creating DataFrame from weather data...")
        # Create a DataFrame with the weather data
        data = pd.DataFrame([request.weather_data])
        
        # Ensure all required NOAA features are present
        required_features = [
            'TEMP', 'TEMP_ATTRIBUTES', 'DEWP_ATTRIBUTES', 'SLP', 'SLP_ATTRIBUTES',
            'STP', 'STP_ATTRIBUTES', 'VISIB', 'VISIB_ATTRIBUTES', 'WDSP_ATTRIBUTES', 'MXSPD', 'GUST',
            'MAX', 'MAX_ATTRIBUTES', 'MIN', 'MIN_ATTRIBUTES', 'PRCP', 'PRCP_ATTRIBUTES', 'FRSHTT', 'SEASON', 'SNDP'
        ]
        
        # Add missing features with default values
        for feature in required_features:
            if feature not in data.columns:
                if feature == 'SEASON':
                    # Calculate season based on date
                    if 'DATE' in data.columns:
                        date = pd.to_datetime(data['DATE'].iloc[0])
                        month = date.month
                        season = 'Winter' if month in [12, 1, 2] else \
                                'Spring' if month in [3, 4, 5] else \
                                'Summer' if month in [6, 7, 8] else 'Fall'
                        data[feature] = season
                    else:
                        data[feature] = 'Unknown'
                elif feature.endswith('_ATTRIBUTES'):
                    data[feature] = '0'  # Default attribute value
                elif feature in ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP']:
                    data[feature] = 0.0  # Default numeric value
                elif feature == 'FRSHTT':
                    data[feature] = '000000'  # Default weather phenomena code
                else:
                    data[feature] = 0.0  # Default value for other numeric features
        
        # Add isUnhealthy column which is required by the model 
        # (trained models require this field even for regression prediction)
        if 'isUnhealthy' not in data.columns:
            logger.info("Adding required 'isUnhealthy' column with default value 0")
            data['isUnhealthy'] = 0  # Default to healthy
        
        # Handle feature types in a more robust way
        try:
            # Different versions of AutoGluon have different ways of accessing feature metadata
            if hasattr(predictor, 'feature_metadata') and hasattr(predictor.feature_metadata, 'get_type_map_raw'):
                features_info = predictor.feature_metadata.get_type_map_raw()
                # Convert categorical columns
                for feature, dtype in features_info.items():
                    if feature in data.columns and 'category' in str(dtype).lower():
                        data[feature] = data[feature].astype('category')
            elif hasattr(predictor, 'feature_metadata') and hasattr(predictor.feature_metadata, 'type_map_raw'):
                features_info = predictor.feature_metadata.type_map_raw
                # Convert categorical columns
                for feature, dtype in features_info.items():
                    if feature in data.columns and 'category' in str(dtype).lower():
                        data[feature] = data[feature].astype('category')
            else:
                logger.warning("Could not determine feature types, continuing with raw features")
        except Exception as e:
            logger.warning(f"Error handling feature types: {str(e)}. Continuing with raw features.")
        
        # Make prediction with timeout
        try:
            logger.info("Starting prediction...")
            predict_func = lambda: trainer.predict(data)
            result = await asyncio.wait_for(
                run_in_threadpool(predict_func),
                timeout=30.0  # 30 second timeout for prediction, increased based on debug results
            )
            
            logger.info("Prediction completed, processing results...")
            
            # Log result structure for debugging
            logger.debug(f"Prediction result shape: {result.shape if not result.empty else 'empty'}")
            logger.debug(f"Prediction result columns: {result.columns.tolist() if not result.empty else []}")
            logger.debug(f"Prediction result first row: {result.iloc[0].to_dict() if not result.empty else {}}")
            
            # Get prediction from result
            try:
                # Check if result is empty
                if result.empty:
                    raise ValueError("Empty prediction result returned")
                
                # Check if prediction column exists
                if "prediction" in result.columns:
                    prediction = result["prediction"].iloc[0]
                    is_unhealthy_pred = bool(prediction)
                elif "isUnhealthy_predicted" in result.columns:
                    is_unhealthy_pred = bool(result["isUnhealthy_predicted"].iloc[0])
                else:
                    # Default to non-unhealthy if we can't determine
                    logger.warning("Could not find prediction column, defaulting to healthy")
                    is_unhealthy_pred = False
            except Exception as e:
                logger.error(f"Error handling prediction result: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error handling prediction result: {str(e)}")
            
            # Get probability if available
            probability = None
            try:
                if "probability_unhealthy" in result.columns:
                    probability = float(result["probability_unhealthy"].iloc[0])
                elif "unhealthy_probability" in result.columns:
                    probability = float(result["unhealthy_probability"].iloc[0])
            except Exception as e:
                logger.warning(f"Could not get probability: {str(e)}")
            
            # Calculate AQI value based on probability and parameter
            param_name = scenario.aq_param_target.name
            
            # Check if this is a regression model prediction (with direct value prediction)
            # Try multiple possible column naming patterns for regression prediction
            regression_prediction_cols = [
                f"{param_name}_value_predicted",  # Standard format: pm25_value_predicted
                f"{param_name}_predicted",        # Alternative format: pm25_predicted
                f"predicted_{param_name}_value",  # Another format: predicted_pm25_value
                f"predicted_{param_name}"         # Simple format: predicted_pm25
            ]
            regression_prediction_col = None
            for col in regression_prediction_cols:
                if col in result.columns:
                    regression_prediction_col = col
                    logger.info(f"Found regression prediction column: {col}")
                    break
            
            if regression_prediction_col in result.columns:
                # Use regression model's predicted value directly
                logger.info(f"Using regression model prediction from column: {regression_prediction_col}")
                estimated_conc = float(result[regression_prediction_col].iloc[0])
                
                # Set is_unhealthy based on whether the predicted value exceeds the threshold
                is_unhealthy_pred = estimated_conc > scenario.unhealthy_threshold
                
                # No need for probability in regression models
                probability = None
            else:
                # Fall back to classification model behavior - calculate estimated concentration
                logger.info("Using classification model to estimate concentration")
                # Calculate a more reasonable concentration based on the probability
                # Use actual unhealthy thresholds from the config
                if param_name == "pm25":
                    # For PM2.5, the unhealthy threshold is around 35.5 µg/m³
                    # Map probability to a more reasonable concentration range
                    unhealthy_threshold = scenario.unhealthy_threshold
                    if is_unhealthy_pred:
                        # If predicted unhealthy, concentration should be above threshold
                        estimated_conc = unhealthy_threshold + (probability * 50)
                    else:
                        # If predicted healthy, ensure a minimum concentration for reasonable AQI
                        min_conc = 3.0  # Minimum concentration to ensure meaningful AQI
                        max_healthy_conc = unhealthy_threshold * 0.9  # Max concentration for healthy prediction (around 32 µg/m³)
                        
                        # Scale the probability to a realistic concentration range
                        estimated_conc = min_conc + (probability * (max_healthy_conc - min_conc))
                elif param_name == "pm10":
                    # For PM10, the unhealthy threshold is around 155 µg/m³
                    unhealthy_threshold = scenario.unhealthy_threshold
                    if is_unhealthy_pred:
                        # If predicted unhealthy, concentration should be above threshold
                        estimated_conc = unhealthy_threshold + (probability * 100)
                    else:
                        # If predicted healthy, ensure a minimum concentration for reasonable AQI
                        min_conc = 10.0  # Minimum concentration to ensure meaningful AQI
                        max_healthy_conc = unhealthy_threshold * 0.9  # Max concentration for healthy prediction (around 140 µg/m³)
                        
                        # Scale the probability to a realistic concentration range
                        estimated_conc = min_conc + (probability * (max_healthy_conc - min_conc))
                elif param_name == "o3":
                    # For O3, the unhealthy threshold is typically around 0.070 ppm
                    unhealthy_threshold = scenario.unhealthy_threshold
                    if is_unhealthy_pred:
                        # If predicted unhealthy, concentration should be above threshold
                        estimated_conc = unhealthy_threshold + (probability * 0.15)  # Add up to 0.15 ppm
                    else:
                        # If predicted healthy, concentration should be below threshold but still realistic
                        # Ensure a minimum concentration (0.02 ppm) and scale up to 90% of threshold
                        min_conc = 0.02  # Minimum concentration to ensure non-zero AQI
                        max_healthy_conc = unhealthy_threshold * 0.9  # Max concentration for healthy prediction
                        
                        # Scale the probability to a realistic concentration range
                        estimated_conc = min_conc + (probability * (max_healthy_conc - min_conc))
                else:
                    # Default conservative estimate
                    estimated_conc = probability * 20
            
            # For classification models only - ensure logical consistency
            # For regression models, we trust the model's output
            is_classification_model = regression_prediction_col is None
            
            if is_classification_model and not is_unhealthy_pred and estimated_conc > scenario.unhealthy_threshold:
                estimated_conc = scenario.unhealthy_threshold * 0.9
            
            # Calculate AQI
            aqi_value, category_info = calculate_aqi(estimated_conc, param_name)
            
            return {
                "scenario": scenario.name,
                "date": request.prediction_date,
                "is_unhealthy": is_unhealthy_pred,
                "probability": probability,
                "concentration": float(estimated_conc),
                "aqi_value": aqi_value,
                "category": category_info["name"] if category_info else None,
                "health_implications": category_info["health_implications"] if category_info else None,
                "prediction_time": datetime.now()
            }
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Prediction timed out")
        except ValueError as ve:
            logger.error(f"Bad request for prediction: {str(ve)}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    except ValueError as ve:
        logger.error(f"Bad request for prediction: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.post("/train/{scenario_name}", response_model=ModelInfoResponse, tags=["Training"])
async def train_model(scenario_name: str, time_limit_secs: Optional[int] = Query(None), eval_metric: Optional[str] = Query(None)):
    """Train a model for the specified scenario."""
    if scenario_name not in app.aq_scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_name}' not found")
    
    scenario = app.aq_scenarios[scenario_name]
    app.select_scenario(scenario_name)
    
    try:
        # Get data with timeout handling
        try:
            noaa_df = app.get_noaa_data()
            if noaa_df.empty:
                raise HTTPException(status_code=500, detail="Failed to retrieve NOAA data")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving NOAA data: {str(e)}")
            
        try:
            openaq_df = app.get_openaq_data()
            if openaq_df.empty:
                raise HTTPException(status_code=500, detail="Failed to retrieve OpenAQ data")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving OpenAQ data: {str(e)}")
        
        # Merge data
        try:
            merged_df = app.get_merged_data(noaa_df, openaq_df)
            if merged_df.empty:
                raise HTTPException(status_code=500, detail="Failed to merge data")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error merging data: {str(e)}")
        
        # Prepare data
        try:
            train_df, val_df, test_df = app.prepare_train_test_data(merged_df)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error preparing data: {str(e)}")
        
        # Train model with timeout
        try:
            trainer = ModelTrainer(scenario, app.ml_target_label)
            time_limit = time_limit_secs or app.ml_time_limit_secs
            predictor = trainer.train_model(train_df, val_df, time_limit, eval_metric=eval_metric)
            
            # Return model info
            model_info = trainer.get_model_info()
            return model_info
        except TimeoutError:
            raise HTTPException(status_code=504, detail="Model training timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/aqi/{scenario_name}", response_model=AQIResponse, tags=["AQI"])
async def get_aqi(scenario_name: str, concentration: float):
    """
    Calculate AQI for a given concentration and scenario.
    
    Args:
        scenario_name: Name of the scenario (determines parameter)
        concentration: Pollutant concentration
    """
    if scenario_name not in app.aq_scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_name}' not found")
    
    scenario = app.aq_scenarios[scenario_name]
    param_name = scenario.aq_param_target.name
    
    try:
        aqi_value, category = calculate_aqi(concentration, param_name)
        
        return AQIResponse(
            aqi=aqi_value,
            concentration=concentration,
            parameter=param_name,
            unit=scenario.aq_param_target.unit,
            category=AQICategory(
                name=category["name"],
                range=category["range"],
                color=category["color"],
                health_implications=category["health_implications"],
                cautionary_statement=category["cautionary_statement"]
            ),
            date=date.today()
        )
    
    except Exception as e:
        logger.error(f"Error calculating AQI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating AQI: {str(e)}")


@router.delete("/models/{scenario_name}", tags=["Models"])
async def delete_model(scenario_name: str):
    """Delete a trained model."""
    if scenario_name not in app.aq_scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_name}' not found")
    
    scenario = app.aq_scenarios[scenario_name]
    trainer = ModelTrainer(scenario, app.ml_target_label)
    
    result = trainer.delete_model()
    
    if result:
        return {"message": f"Model for scenario '{scenario_name}' deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model for scenario '{scenario_name}' not found") 