import React, { useState, useEffect } from 'react';
import { NextPage } from 'next';
import { useRouter } from 'next/router';
import { AQScenario, PredictionResponse } from '../types';
import { getScenarios, getPrediction } from '../services/api';
import Layout from '../components/Layout';
import CityCard from '../components/CityCard';
import ScenarioSelector from '../components/ScenarioSelector';
import AQIInfoCard from '../components/AQIInfoCard';

const Home: NextPage = () => {
  const router = useRouter();
  const [scenarios, setScenarios] = useState<AQScenario[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<string>('');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch scenarios on component mount
  useEffect(() => {
    const fetchScenarios = async () => {
      try {
        setLoading(true);
        const data = await getScenarios();
        
        if (data.scenarios && data.scenarios.length > 0) {
          setScenarios(data.scenarios);
          
          // Check if there's a scenario in the URL query params
          const scenarioFromQuery = router.query.scenario as string;
          
          if (scenarioFromQuery && data.scenarios.some(s => s.name === scenarioFromQuery)) {
            // If the scenario from the query exists in our scenarios, select it
            setSelectedScenario(scenarioFromQuery);
          } else {
            // Otherwise select the first scenario by default
            setSelectedScenario(data.scenarios[0].name);
          }
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

    if (router.isReady) {
      fetchScenarios();
    }
  }, [router.isReady, router.query.scenario]);

  // Fetch prediction when scenario changes
  useEffect(() => {
    if (!selectedScenario) return;

    const fetchPrediction = async () => {
      try {
        setLoading(true);
        const data = await getPrediction(selectedScenario);
        setPrediction(data);
      } catch (err) {
        console.error('Error fetching prediction:', err);
        setError('Failed to load prediction');
        setPrediction(null);
      } finally {
        setLoading(false);
      }
    };

    fetchPrediction();
  }, [selectedScenario]);

  const handleScenarioChange = (scenarioName: string) => {
    // Update the URL when the scenario changes without full page refresh
    router.push(`/?scenario=${scenarioName}`, undefined, { shallow: true });
    setSelectedScenario(scenarioName);
  };

  return (
    <Layout title="AQI Prediction - Home">
      <div className="px-4 py-6 sm:px-0">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Air Quality Prediction</h1>
          <p className="text-gray-600">
            Get real-time air quality predictions for major cities.
          </p>
        </div>

        {error ? (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-6">
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
        ) : null}

        {scenarios.length > 0 && (
          <div className="mb-6">
            <ScenarioSelector
              scenarios={scenarios}
              selectedScenario={selectedScenario}
              onScenarioChange={handleScenarioChange}
            />
          </div>
        )}

        {selectedScenario && prediction && (
          <CityCard 
            scenarioName={selectedScenario}
            date={prediction.date}
          />
        )}

        <AQIInfoCard prediction={prediction} loading={loading} />
      </div>
    </Layout>
  );
};

export default Home; 