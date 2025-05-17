import React from 'react';
import { PredictionResponse, aqiCategoryInfo } from '../types';
import AQIGauge from './AQIGauge';
import HealthInfo from './HealthInfo';

interface AQIInfoCardProps {
  prediction: PredictionResponse | null;
  loading: boolean;
}

const AQIInfoCard: React.FC<AQIInfoCardProps> = ({ prediction, loading }) => {
  if (loading) {
    return (
      <div className="aqi-card bg-white p-6 flex flex-col items-center">
        <div className="animate-pulse flex flex-col items-center w-full">
          <div className="rounded-full bg-gray-200 h-32 w-32 mb-4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
          <div className="h-24 bg-gray-200 rounded w-full"></div>
        </div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="aqi-card bg-white p-6">
        <p className="text-center text-gray-500">No prediction data available</p>
      </div>
    );
  }

  const { aqi_value, category, health_implications, probability } = prediction;
  
  // Background color gradient based on AQI category
  const bgColor = category && aqiCategoryInfo[category] 
    ? `linear-gradient(to bottom, white, ${aqiCategoryInfo[category].color}10)`
    : 'white';

  return (
    <div 
      className="aqi-card bg-white p-6"
      style={{ background: bgColor }}
    >
      <div className="flex flex-col md:flex-row">
        <div className="flex-1">
          <AQIGauge aqi={aqi_value} category={category} />
        </div>
        
        <div className="flex-1 mt-6 md:mt-0 md:ml-6">
          <h3 className="text-lg font-medium text-gray-900 mb-2">Air Quality Details</h3>
          
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-sm text-gray-500">AQI Value</p>
              <p className="text-lg font-medium">{aqi_value !== null ? aqi_value : 'N/A'}</p>
            </div>
            
            <div>
              <p className="text-sm text-gray-500">Category</p>
              <p className="text-lg font-medium">{category || 'Unknown'}</p>
            </div>
            
            <div>
              <p className="text-sm text-gray-500">Unhealthy Probability</p>
              <p className="text-lg font-medium">
                {probability !== null ? `${(probability * 100).toFixed(1)}%` : 'N/A'}
              </p>
            </div>
            
            <div>
              <p className="text-sm text-gray-500">Prediction Status</p>
              <p className="text-lg font-medium">
                {prediction.is_unhealthy ? 'Unhealthy' : 'Healthy'}
              </p>
            </div>
          </div>
          
          <HealthInfo 
            category={category} 
            healthImplications={health_implications} 
          />
        </div>
      </div>
    </div>
  );
};

export default AQIInfoCard; 