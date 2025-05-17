"""
Configuration settings for the AQI prediction system.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# NOAA GSOD settings
NOAA_BUCKET = "noaa-gsod-pds"

# OpenAQ settings
OPENAQ_API_URL = "https://api.openaq.org/v3"

# Model settings
DEFAULT_ML_TARGET_LABEL = "isUnhealthy"
DEFAULT_ML_EVAL_METRIC = "accuracy"
DEFAULT_ML_TIME_LIMIT_SECS = 300  # 5 minutes

# AQI settings - EPA Breakpoints for PM2.5 (μg/m³, 24-hour average)
AQI_BREAKPOINTS = {
    "pm25": {
        (0, 12.0): (0, 50),       # Good
        (12.1, 35.4): (51, 100),  # Moderate
        (35.5, 55.4): (101, 150), # Unhealthy for Sensitive Groups
        (55.5, 150.4): (151, 200),# Unhealthy
        (150.5, 250.4): (201, 300),# Very Unhealthy
        (250.5, 500.4): (301, 500) # Hazardous
    },
    "pm10": {
        (0, 54): (0, 50),         # Good
        (55, 154): (51, 100),     # Moderate
        (155, 254): (101, 150),   # Unhealthy for Sensitive Groups
        (255, 354): (151, 200),   # Unhealthy
        (355, 424): (201, 300),   # Very Unhealthy
        (425, 604): (301, 500)    # Hazardous
    }
}

# Default unhealthy thresholds for AQ parameters - Concentration at which AQI > 100
UNHEALTHY_THRESHOLDS = {
    "pm25": 35.5,  # µg/m³
    "pm10": 155,   # µg/m³
    "o3": 0.070,   # ppm
    "no2": 0.100,  # ppm
    "so2": 0.075,  # ppm
    "co": 9.5      # ppm
}

# Feature engineering settings
DEFAULT_WEATHER_FEATURES = [
    'DEWP',     # Dew point temperature
    'WDSP',     # Wind speed
    'MAX',      # Maximum temperature
    'MIN',      # Minimum temperature
    'PRCP',     # Precipitation
    'MONTH',    # Month
    'DAYOFWEEK',# Day of week
    'SEASON',   # Season
    'TEMP_RANGE',# Temperature range
    'TEMP_AVG', # Average temperature
    'TEMP_DEWP_DIFF',# Temperature-dewpoint difference
    'WDSP_TEMP'# Wind speed * temperature
]

# API settings
API_PORT = 8000
API_HOST = "0.0.0.0" 