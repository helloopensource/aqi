import React from 'react';
import { aqiCategoryInfo } from '../types';

interface HealthInfoProps {
  category: string | null;
  healthImplications: string | null;
}

const HealthInfo: React.FC<HealthInfoProps> = ({ 
  category, 
  healthImplications 
}) => {
  if (!category || !healthImplications) {
    return (
      <div className="p-4 bg-gray-100 rounded-lg mt-4">
        <p className="text-gray-600">Health information not available</p>
      </div>
    );
  }

  const categoryClass = aqiCategoryInfo[category]?.class || 'good';

  return (
    <div className={`health-info ${categoryClass} p-4 bg-gray-50 rounded-lg mt-4`}>
      <h3 className="font-bold text-lg mb-2">Health Implications</h3>
      <p className="text-gray-700">{healthImplications}</p>
      
      {category === 'Good' && (
        <div className="mt-4 flex items-center">
          <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
            Safe for Outdoor Activities
          </span>
        </div>
      )}

      {category === 'Moderate' && (
        <div className="mt-4 flex items-center">
          <span className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm font-medium">
            Generally Safe for Most People
          </span>
        </div>
      )}

      {category === 'Unhealthy for Sensitive Groups' && (
        <div className="mt-4 flex items-center">
          <span className="bg-orange-100 text-orange-800 px-3 py-1 rounded-full text-sm font-medium">
            Consider Limiting Outdoor Activities
          </span>
        </div>
      )}

      {category === 'Unhealthy' && (
        <div className="mt-4 flex items-center">
          <span className="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-medium">
            Limit Outdoor Activities
          </span>
        </div>
      )}

      {category === 'Very Unhealthy' && (
        <div className="mt-4 flex items-center">
          <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm font-medium">
            Avoid Outdoor Activities
          </span>
        </div>
      )}

      {category === 'Hazardous' && (
        <div className="mt-4 flex items-center">
          <span className="bg-red-200 text-red-900 px-3 py-1 rounded-full text-sm font-medium">
            Stay Indoors - Health Emergency
          </span>
        </div>
      )}
    </div>
  );
};

export default HealthInfo; 