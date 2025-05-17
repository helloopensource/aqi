import React, { useEffect, useState } from 'react';
import { aqiCategoryInfo } from '../types';

interface AQIGaugeProps {
  aqi: number | null;
  category: string | null;
}

const AQIGauge: React.FC<AQIGaugeProps> = ({ aqi, category }) => {
  const [rotation, setRotation] = useState(0);
  const [color, setColor] = useState('#ccc');

  useEffect(() => {
    if (aqi === null) {
      setRotation(0);
      setColor('#ccc');
      return;
    }

    // Calculate rotation angle based on AQI (0-500)
    // 0 AQI = -90 degrees, 500 AQI = 90 degrees
    const newRotation = (aqi / 500) * 180 - 90;
    setRotation(newRotation);

    // Set color based on AQI category
    if (category && aqiCategoryInfo[category]) {
      setColor(aqiCategoryInfo[category].color);
    } else {
      setColor('#ccc');
    }
  }, [aqi, category]);

  // Generate the segments for the gauge background
  const renderGaugeSegments = () => {
    const categories = Object.entries(aqiCategoryInfo);
    return (
      <svg width="200" height="200" viewBox="0 0 100 100" className="absolute top-0 left-0">
        <defs>
          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            {categories.map(([name, { color }], index) => (
              <stop
                key={name}
                offset={`${(index / (categories.length - 1)) * 100}%`}
                stopColor={color}
              />
            ))}
          </linearGradient>
        </defs>
        <path
          d="M 10 50 A 40 40 0 1 1 90 50"
          fill="none"
          stroke="url(#gaugeGradient)"
          strokeWidth="12"
          strokeLinecap="round"
        />
      </svg>
    );
  };

  return (
    <div className="flex flex-col items-center pt-8">
      <div className="aqi-gauge">
        {renderGaugeSegments()}
        <div
          className="aqi-needle"
          style={{ transform: `translateX(-50%) rotate(${rotation}deg)` }}
        />
        <div className="aqi-needle-cap" />
      </div>
      <div className="mt-4 text-center">
        <h3 className="text-xl font-bold mt-2">{aqi !== null ? aqi : '-'}</h3>
        <div 
          className="category-label" 
          style={{ color: color }}
        >
          {category || 'Unknown'}
        </div>
      </div>
    </div>
  );
};

export default AQIGauge; 