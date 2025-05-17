import axios from 'axios';
import { format } from 'date-fns';
import { AQScenario, PredictionRequest, PredictionResponse, WeatherData } from '../types';

// Base URL for the API
const API_URL = 'http://localhost:8000/api/v1';

// Default weather data - this would typically come from real weather data
const getDefaultWeatherData = (): WeatherData => {
  return {
    DEWP: 58.5,
    WDSP: 6.2,
    MAX: 75.3,
    MIN: 60.1,
    PRCP: 0.0,
    MONTH: new Date().getMonth() + 1,
    DAYOFWEEK: new Date().getDay(),
    TEMP_RANGE: 15.2,
    TEMP_AVG: 67.7,
    TEMP_DEWP_DIFF: 9.2,
    WDSP_TEMP: 419.74
  };
};

// Get all available scenarios
export const getScenarios = async (): Promise<{ scenarios: AQScenario[] }> => {
  try {
    const response = await axios.get(`${API_URL}/scenarios`);
    return response.data;
  } catch (error) {
    console.error('Error fetching scenarios:', error);
    throw error;
  }
};

// Get a specific scenario by name
export const getScenario = async (scenarioName: string): Promise<AQScenario> => {
  try {
    const response = await axios.get(`${API_URL}/scenarios/${scenarioName}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching scenario ${scenarioName}:`, error);
    throw error;
  }
};

// Get prediction for a scenario
export const getPrediction = async (
  scenarioName: string,
  date: Date = new Date(),
  weatherData: WeatherData = getDefaultWeatherData()
): Promise<PredictionResponse> => {
  try {
    const formattedDate = format(date, 'yyyy-MM-dd');
    
    const requestData: PredictionRequest = {
      scenario: scenarioName,
      date: formattedDate,
      weather_data: weatherData
    };
    
    const response = await axios.post(`${API_URL}/predict`, requestData);
    return response.data;
  } catch (error) {
    console.error(`Error getting prediction for ${scenarioName}:`, error);
    throw error;
  }
};

// Health check endpoint
export const healthCheck = async (): Promise<{ status: string, version: string, timestamp: string }> => {
  try {
    const response = await axios.get(`${API_URL}/health`);
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
}; 