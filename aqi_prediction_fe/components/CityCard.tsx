import React from 'react';
import Image from 'next/image';
import { cityImages } from '../types';

interface CityCardProps {
  scenarioName: string;
  date: string;
}

const CityCard: React.FC<CityCardProps> = ({ scenarioName, date }) => {
  // Extract the city name from the scenario (e.g., "los-angeles_pm25" -> "los-angeles")
  const cityKey = scenarioName.split('_')[0];
  const cityData = cityImages[cityKey] || {
    city: 'Unknown City',
    image: 'https://images.unsplash.com/photo-1576502200916-3808e07386a5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=80',
    altText: 'Generic city skyline'
  };

  // Format the date
  const formattedDate = new Date(date).toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  return (
    <div className="city-image relative rounded-lg overflow-hidden mb-6">
      <div className="relative h-48 md:h-64 w-full">
        <Image
          src={cityData.image}
          alt={cityData.altText}
          fill
          style={{ objectFit: 'cover' }}
          priority
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />
      </div>
      <div className="absolute bottom-0 left-0 p-4 text-white">
        <h2 className="text-2xl font-bold">{cityData.city}</h2>
        <p className="text-sm opacity-90">{formattedDate}</p>
      </div>
    </div>
  );
};

export default CityCard; 