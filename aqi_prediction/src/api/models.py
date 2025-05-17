"""
API data models for the AQI prediction system.
"""
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import date, datetime


class AQParamModel(BaseModel):
    """Air quality parameter model for API."""
    id: int
    name: str
    unit: str
    unhealthy_threshold_default: float
    desc: str


class AQScenarioModel(BaseModel):
    """Air quality scenario model for API."""
    location: str
    name: str
    noaa_station_id: str
    noaa_station_lat: float = 0.0
    noaa_station_lng: float = 0.0
    open_aq_sensor_ids: List[int] = []
    unhealthy_threshold: float
    year_start: int
    year_end: int
    aq_radius_miles: int
    target_param: str
    target_param_desc: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


class PredictionRequest(BaseModel):
    """Request model for making predictions."""
    scenario_name: str = Field(..., description="Name of the scenario to use", alias="scenario")
    prediction_date: date = Field(..., description="Date for prediction", alias="date")
    weather_data: Dict[str, Union[float, str]] = Field(..., description="Weather data for prediction")

    class Config:
        populate_by_name = True
        allow_population_by_field_name = True


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    scenario: str
    date: date
    is_unhealthy: bool
    probability: Optional[float] = None
    aqi_value: Optional[int] = None
    category: Optional[str] = None
    health_implications: Optional[str] = None
    prediction_time: datetime = Field(default_factory=datetime.now)


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    scenario: str
    target: str
    path: str
    problem_type: str
    eval_metric: str
    model_types: List[str]
    best_model: Optional[str] = None
    features: List[str]


class ScenarioListResponse(BaseModel):
    """Response model for listing available scenarios."""
    scenarios: List[AQScenarioModel]


class AQICategory(BaseModel):
    """AQI category information."""
    name: str
    range: str
    color: str
    health_implications: str
    cautionary_statement: str


class AQIResponse(BaseModel):
    """Response model for AQI information."""
    aqi: int
    concentration: float
    parameter: str
    unit: str
    category: AQICategory
    date: date 