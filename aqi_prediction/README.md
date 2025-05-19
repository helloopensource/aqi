# AQI Prediction System

A machine learning-based system for predicting Air Quality Index (AQI) values based on weather data.

## Overview

This project implements a system to predict air quality using weather data from NOAA's Global Surface Summary of the Day (GSOD) and air quality measurements from OpenAQ. The system uses machine learning models to establish correlations between weather patterns and air quality, enabling forecasting of AQI values.

## Features

- Data collection from NOAA GSOD and OpenAQ
- Feature engineering for weather data
- Machine learning model training using AutoGluon
- REST API for model training and predictions
- AQI calculation according to EPA standards
- Support for multiple air quality parameters (PM2.5, PM10, etc.)
- Pre-configured scenarios for major US cities

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd aqi_prediction
```

2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAQ API key:
```
OPENAQ_API_KEY=your_api_key_here
```

## Usage

### Starting the API

Run the API server:

```bash
python main.py
```

This will start the FastAPI server on http://localhost:8000 by default.

### API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Training a Model

To train a model for a specific scenario:

```bash
curl -X POST "http://localhost:8000/api/v1/train/los-angeles_pm25?time_limit_secs=900&eval_metric=f1"
```

### Making Predictions

To make a prediction using a trained model:

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "scenario_name": "los-angeles_pm25",
           "prediction_date": "2023-06-15",
           "weather_data": {
             "DATE": "2023-06-15",
             "TEMP": 70.2,
             "TEMP_ATTRIBUTES": "0",
             "DEWP": 50.5,
             "DEWP_ATTRIBUTES": "0",
             "SLP": 1013.2,
             "SLP_ATTRIBUTES": "0",
             "STP": 1013.2,
             "STP_ATTRIBUTES": "0",
             "VISIB": 10.0,
             "WDSP": 5.2,
             "WDSP_ATTRIBUTES": "0",
             "MXSPD": 8.0,
             "GUST": 12.0,
             "MAX": 78.3,
             "MAX_ATTRIBUTES": "0",
             "MIN": 62.1,
             "PRCP": 0.0,
             "PRCP_ATTRIBUTES": "0",
             "SNDP": 0.0,
             "FRSHTT": "000000",
             "MONTH": 6,
             "DAYOFWEEK": 3,
             "SEASON": "Summer",
             "TEMP_RANGE": 16.2,
             "TEMP_AVG": 70.2,
             "TEMP_DEWP_DIFF": 19.7,
             "WDSP_TEMP": 365.04
           }
         }'
```

Note: The API will automatically add any missing required features with default values if they are not provided in the request.

## Project Structure

```
aqi_prediction/
├── data/                  # Data storage
│   ├── models/            # Trained models
│   ├── processed/         # Processed data files
│   └── raw/               # Raw data from NOAA and OpenAQ
├── src/                   # Source code
│   ├── api/               # API endpoints and models
│   ├── config/            # Configuration settings
│   ├── data_processors/   # Data processing modules
│   ├── models/            # ML model definitions
│   └── utils/             # Utility functions
├── tests/                 # Test modules
├── .env                   # Environment variables
├── main.py                # Application entry point
└── requirements.txt       # Project dependencies
```

## Data Sources

- NOAA Global Surface Summary of the Day: https://registry.opendata.aws/noaa-gsod/
- OpenAQ: https://registry.opendata.aws/openaq/

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 