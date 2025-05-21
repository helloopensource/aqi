#!/bin/bash
# Script to test the API with curl

echo "Testing the predict endpoint with regression model..."

# Create a request JSON file
cat > request.json << EOL
{
  "scenario": "los-angeles_pm25",
  "date": "2023-05-21",
  "weather_data": {
    "TEMP": 75.0,
    "TEMP_ATTRIBUTES": 0,
    "DEWP": 60.0,
    "DEWP_ATTRIBUTES": 0,
    "SLP": 1015.0,
    "SLP_ATTRIBUTES": 0,
    "STP": 1013.0,
    "STP_ATTRIBUTES": 0,
    "VISIB": 10.0,
    "VISIB_ATTRIBUTES": 0,
    "WDSP": 5.0,
    "WDSP_ATTRIBUTES": 0,
    "MXSPD": 10.0,
    "GUST": 15.0,
    "MAX": 85.0,
    "MAX_ATTRIBUTES": 0,
    "MIN": 65.0,
    "MIN_ATTRIBUTES": 0,
    "PRCP": 0.0,
    "PRCP_ATTRIBUTES": "0",
    "SNDP": 0.0,
    "FRSHTT": "000000",
    "MONTH": 5,
    "DAYOFWEEK": 2,
    "SEASON": "Spring",
    "TEMP_RANGE": 20.0,
    "TEMP_AVG": 75.0,
    "TEMP_DEWP_DIFF": 15.0,
    "WDSP_TEMP": 375.0,
    "isUnhealthy": 0
  }
}
EOL

# Send request to predict endpoint
curl -X POST -H "Content-Type: application/json" -d @request.json http://localhost:8000/api/v1/predict | jq

# Clean up
rm request.json

echo "Done!" 