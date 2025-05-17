import React, { useState, useEffect } from 'react';
import { NextPage } from 'next';
import Link from 'next/link';
import Image from 'next/image';
import { AQScenario } from '../types';
import { getScenarios } from '../services/api';
import Layout from '../components/Layout';
import { cityImages } from '../types';

const Cities: NextPage = () => {
  const [scenarios, setScenarios] = useState<AQScenario[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Group scenarios by location
  const scenariosByLocation = scenarios.reduce((acc, scenario) => {
    const location = scenario.location;
    if (!acc[location]) {
      acc[location] = [];
    }
    acc[location].push(scenario);
    return acc;
  }, {} as { [key: string]: AQScenario[] });

  // Fetch scenarios on component mount
  useEffect(() => {
    const fetchScenarios = async () => {
      try {
        setLoading(true);
        const data = await getScenarios();
        
        if (data.scenarios && data.scenarios.length > 0) {
          setScenarios(data.scenarios);
        } else {
          setError('No scenarios available');
        }
      } catch (err) {
        console.error('Error fetching scenarios:', err);
        setError('Failed to load scenarios');
      } finally {
        setLoading(false);
      }
    };

    fetchScenarios();
  }, []);

  // Get city key from location
  const getCityKey = (location: string): string => {
    return location.toLowerCase().replace(/\s+/g, '-');
  };

  return (
    <Layout title="AQI Prediction - Cities">
      <div className="px-4 py-6 sm:px-0">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Available Cities</h1>
          <p className="text-gray-600">
            Select a city to view current air quality predictions.
          </p>
        </div>

        {loading ? (
          <div className="animate-pulse grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3, 4].map((item) => (
              <div key={item} className="bg-gray-200 rounded-lg h-64"></div>
            ))}
          </div>
        ) : error ? (
          <div className="bg-red-50 border-l-4 border-red-400 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(scenariosByLocation).map(([location, locScenarios]) => {
              const cityKey = getCityKey(location);
              const cityData = cityImages[cityKey] || {
                city: location,
                image: 'https://images.unsplash.com/photo-1576502200916-3808e07386a5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=80',
                altText: `${location} skyline`
              };
              
              return (
                <Link 
                  href={`/?scenario=${locScenarios[0].name}`} 
                  key={location}
                  legacyBehavior
                >
                  <a className="block rounded-lg overflow-hidden shadow-lg hover:shadow-xl transition duration-300 transform hover:-translate-y-1 city-image">
                    <div className="relative h-48 w-full">
                      <Image
                        src={cityData.image}
                        alt={cityData.altText}
                        fill
                        style={{ objectFit: 'cover' }}
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent" />
                    </div>
                    <div className="absolute bottom-0 left-0 p-4 text-white">
                      <h2 className="text-xl font-bold">{location}</h2>
                      <p className="text-sm opacity-90">
                        {locScenarios.map(s => s.target_param).join(', ')}
                      </p>
                    </div>
                  </a>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </Layout>
  );
};

export default Cities; 