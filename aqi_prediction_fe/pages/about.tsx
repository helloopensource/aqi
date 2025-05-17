import React from 'react';
import { NextPage } from 'next';
import Link from 'next/link';
import Layout from '../components/Layout';

const About: NextPage = () => {
  return (
    <Layout title="AQI Prediction - About">
      <div className="px-4 py-6 sm:px-0">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">About AQI Prediction</h1>
          <p className="text-gray-600">
            Learn about our air quality prediction system and how it works.
          </p>
        </div>

        <div className="bg-white shadow overflow-hidden sm:rounded-lg">
          <div className="px-4 py-5 sm:px-6">
            <h2 className="text-lg leading-6 font-medium text-gray-900">
              Air Quality Index (AQI) Information
            </h2>
            <p className="mt-1 max-w-2xl text-sm text-gray-500">
              Understanding air quality categories and health implications.
            </p>
          </div>
          <div className="border-t border-gray-200">
            <dl>
              <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">Data Sources</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  <ul className="list-disc pl-5 space-y-1">
                    <li>National Oceanic and Atmospheric Administration (NOAA): Global Surface Summary of the Day</li>
                    <li>OpenAQ: Global aggregated physical air quality data</li>
                  </ul>
                </dd>
              </div>
              <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">AQI Categories</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  <div className="overflow-hidden rounded-lg">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">AQI Range</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Health Implications</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">0-50</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                              Good
                            </span>
                          </td>
                          <td className="px-6 py-4 text-sm text-gray-500">Air quality is considered satisfactory, and air pollution poses little or no risk.</td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">51-100</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-yellow-100 text-yellow-800">
                              Moderate
                            </span>
                          </td>
                          <td className="px-6 py-4 text-sm text-gray-500">Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people.</td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">101-150</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-orange-100 text-orange-800">
                              Unhealthy for Sensitive Groups
                            </span>
                          </td>
                          <td className="px-6 py-4 text-sm text-gray-500">Members of sensitive groups may experience health effects. The general public is not likely to be affected.</td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">151-200</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">
                              Unhealthy
                            </span>
                          </td>
                          <td className="px-6 py-4 text-sm text-gray-500">Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.</td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">201-300</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-purple-100 text-purple-800">
                              Very Unhealthy
                            </span>
                          </td>
                          <td className="px-6 py-4 text-sm text-gray-500">Health warnings of emergency conditions. The entire population is more likely to be affected.</td>
                        </tr>
                        <tr>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">301-500</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-200 text-red-900">
                              Hazardous
                            </span>
                          </td>
                          <td className="px-6 py-4 text-sm text-gray-500">Health alert: everyone may experience more serious health effects.</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </dd>
              </div>
              <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">Machine Learning Models</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  <p>
                    Our system uses advanced machine learning models trained on historical air quality and weather data to predict AQI values. The models learn patterns between weather conditions and air quality, allowing for accurate predictions based on current meteorological data.
                  </p>
                </dd>
              </div>
              <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">Additional Resources</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  <ul className="list-disc pl-5 space-y-1">
                    <li><a href="https://www.airnow.gov/" className="text-blue-600 hover:text-blue-800" target="_blank" rel="noopener noreferrer">AirNow.gov</a> - Official US Government AQI information</li>
                    <li><a href="https://www.epa.gov/air-quality-air-quality-index" className="text-blue-600 hover:text-blue-800" target="_blank" rel="noopener noreferrer">US EPA Air Quality Index</a> - Detailed information about AQI</li>
                    <li><a href="https://openaq.org/" className="text-blue-600 hover:text-blue-800" target="_blank" rel="noopener noreferrer">OpenAQ</a> - Global air quality data platform</li>
                  </ul>
                </dd>
              </div>
            </dl>
          </div>
        </div>

        <div className="mt-8 text-center">
          <Link href="/" legacyBehavior>
            <a className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
              Return to Home
            </a>
          </Link>
        </div>
      </div>
    </Layout>
  );
};

export default About; 