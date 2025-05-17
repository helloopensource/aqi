"""
FastAPI endpoints for the AQI prediction service.
"""
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, date

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from ..models.air_quality import AQParam, AQScenario, get_default_scenarios
from ..models.aqi_app import AQIApp
from ..models.model_trainer import ModelTrainer
from ..utils.aqi_calculator import calculate_aqi, get_category_from_aqi, is_unhealthy
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
async def get_model_info(scenario_name: str):
    """Get information about a trained model."""
    if scenario_name not in app.aq_scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_name}' not found")
    
    scenario = app.aq_scenarios[scenario_name]
    trainer = ModelTrainer(scenario, app.ml_target_label)
    
    model_info = trainer.get_model_info()
    
    if "error" in model_info:
        raise HTTPException(status_code=404, detail=model_info["error"])
    
    return model_info


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a prediction for the given scenario and weather data.
    
    Example request:
    ```json
    {
        "scenario": "los-angeles_pm25",
        "date": "2023-06-15",
        "weather_data": {
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
            "WDSP_TEMP": 365.04
        }
    }
    ```
    """
    if request.scenario_name not in app.aq_scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{request.scenario_name}' not found")
    
    scenario = app.aq_scenarios[request.scenario_name]
    trainer = ModelTrainer(scenario, app.ml_target_label)
    
    # Check if model exists
    predictor = trainer.load_model()
    if predictor is None:
        raise HTTPException(status_code=404, detail="Model not found. Train the model first.")
    
    # Create DataFrame from weather data
    try:
        # Create a DataFrame with the weather data
        data = pd.DataFrame([request.weather_data])
        
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
        
        # Make prediction
        result = trainer.predict(data)
        
        # Get prediction from result
        prediction = result["prediction"].iloc[0]
        is_unhealthy_pred = bool(prediction)
        
        # Get probability if available
        probability = None
        if "probability_unhealthy" in result.columns:
            probability = float(result["probability_unhealthy"].iloc[0])
        
        # Calculate AQI value based on probability and parameter
        param_name = scenario.aq_param_target.name
        
        # Simplified approach: derive a concentration estimate from probability
        aqi_value = None
        category = None
        
        if probability is not None:
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
                    # If predicted healthy, concentration should be below threshold
                    estimated_conc = probability * unhealthy_threshold * 0.8
            elif param_name == "pm10":
                # For PM10, the unhealthy threshold is around 155 µg/m³
                unhealthy_threshold = scenario.unhealthy_threshold
                if is_unhealthy_pred:
                    # If predicted unhealthy, concentration should be above threshold
                    estimated_conc = unhealthy_threshold + (probability * 100)
                else:
                    # If predicted healthy, concentration should be below threshold
                    estimated_conc = probability * unhealthy_threshold * 0.8
            else:
                # Default conservative estimate
                estimated_conc = probability * 20
            
            # Ensure logical consistency - cap maximum concentration if not unhealthy
            if not is_unhealthy_pred and estimated_conc > scenario.unhealthy_threshold:
                estimated_conc = scenario.unhealthy_threshold * 0.9
            
            # Calculate AQI
            aqi_value, category_info = calculate_aqi(estimated_conc, param_name)
        
        return {
            "scenario": scenario.name,
            "date": request.prediction_date,  # Use prediction_date (mapped from 'date' in JSON)
            "is_unhealthy": is_unhealthy_pred,
            "probability": probability,
            "aqi_value": aqi_value,
            "category": category_info["name"] if category_info else None,
            "health_implications": category_info["health_implications"] if category_info else None,
            "prediction_time": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/{scenario_name}", response_model=ModelInfoResponse, tags=["Training"])
async def train_model(scenario_name: str, time_limit_secs: Optional[int] = Query(None)):
    """Train a model for the specified scenario."""
    if scenario_name not in app.aq_scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_name}' not found")
    
    scenario = app.aq_scenarios[scenario_name]
    app.select_scenario(scenario_name)
    
    try:
        # Get data
        noaa_df = app.get_noaa_data()
        openaq_df = app.get_openaq_data()
        
        if noaa_df.empty or openaq_df.empty:
            raise HTTPException(status_code=500, detail="Failed to retrieve data")
        
        # Merge data
        merged_df = app.get_merged_data(noaa_df, openaq_df)
        
        if merged_df.empty:
            raise HTTPException(status_code=500, detail="Failed to merge data")
        
        # Prepare data
        train_df, val_df, test_df = app.prepare_train_test_data(merged_df)
        
        # Train model
        trainer = ModelTrainer(scenario, app.ml_target_label)
        
        time_limit = time_limit_secs or app.ml_time_limit_secs
        predictor = trainer.train_model(train_df, val_df, time_limit)
        
        # Return model info
        model_info = trainer.get_model_info()
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during model training: {str(e)}")


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