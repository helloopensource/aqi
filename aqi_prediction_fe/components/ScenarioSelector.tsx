import React from 'react';
import { AQScenario } from '../types';

interface ScenarioSelectorProps {
  scenarios: AQScenario[];
  selectedScenario: string;
  onScenarioChange: (scenario: string) => void;
}

const ScenarioSelector: React.FC<ScenarioSelectorProps> = ({
  scenarios,
  selectedScenario,
  onScenarioChange
}) => {
  // Group scenarios by location
  const scenariosByLocation: { [key: string]: AQScenario[] } = scenarios.reduce((acc, scenario) => {
    const location = scenario.location;
    if (!acc[location]) {
      acc[location] = [];
    }
    acc[location].push(scenario);
    return acc;
  }, {} as { [key: string]: AQScenario[] });

  return (
    <div className="mb-6">
      <label htmlFor="scenario-select" className="block text-sm font-medium text-gray-700 mb-2">
        Select Location & Parameter
      </label>
      <select
        id="scenario-select"
        value={selectedScenario}
        onChange={(e) => onScenarioChange(e.target.value)}
        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md shadow-sm"
      >
        {Object.entries(scenariosByLocation).map(([location, locScenarios]) => (
          <optgroup key={location} label={location}>
            {locScenarios.map((scenario) => (
              <option key={scenario.name} value={scenario.name}>
                {scenario.location} - {scenario.target_param} ({scenario.target_param_desc})
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      
      <div className="mt-2 text-sm text-gray-500">
        <p>View air quality predictions based on location and pollutant</p>
      </div>
    </div>
  );
};

export default ScenarioSelector; 