"""
Main AQ by Weather application class.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import boto3
import requests
import random
from botocore import UNSIGNED
from botocore.config import Config
from io import StringIO

from .air_quality import AQParam, AQScenario
from ..config.settings import (
    NOAA_BUCKET, 
    OPENAQ_API_URL, 
    DEFAULT_ML_TARGET_LABEL,
    DEFAULT_ML_EVAL_METRIC,
    DEFAULT_ML_TIME_LIMIT_SECS,
    DEFAULT_WEATHER_FEATURES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR
)

logger = logging.getLogger(__name__)


class AQIApp:
    """
    Main application class for AQI prediction system with data access methods
    and model management.
    """
    @staticmethod
    def _calculate_distance(lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two points using the Haversine formula.
        
        Args:
            lat1: Latitude of point 1 in decimal degrees
            lon1: Longitude of point 1 in decimal degrees
            lat2: Latitude of point 2 in decimal degrees
            lon2: Longitude of point 2 in decimal degrees
            
        Returns:
            Distance in kilometers
        """
        import math
        
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r
        
    def _create_synthetic_data(self, filename: str):
        """
        Create synthetic air quality data for training when no real data is available.
        
        Args:
            filename: Path to save the synthetic data
        """
        logger.info(f"Creating synthetic {self.selected_scenario.aq_param_target.name} data for model training")
        
        # Parameter-specific configurations
        param_name = self.selected_scenario.aq_param_target.name
        
        # Define typical ranges based on parameter type
        if param_name == "pm25":
            # PM2.5 typically ranges from 0-100 μg/m³, with spikes up to 300
            # Healthy threshold is around 35.5 μg/m³
            mean_value = 20.0
            std_dev = 15.0
            max_value = 300.0
            min_value = 2.0
        elif param_name == "pm10":
            # PM10 typically ranges from 0-150 μg/m³, with spikes up to 500
            # Healthy threshold is around 155 μg/m³
            mean_value = 60.0
            std_dev = 40.0
            max_value = 500.0
            min_value = 5.0
        elif param_name == "o3":
            # O3 typically ranges from 0-0.2 ppm, with spikes up to 0.5
            # Healthy threshold is around 0.07 ppm
            mean_value = 0.04
            std_dev = 0.025
            max_value = 0.5
            min_value = 0.01
        else:
            # Generic range for other parameters
            mean_value = self.selected_scenario.unhealthy_threshold * 0.6
            std_dev = self.selected_scenario.unhealthy_threshold * 0.3
            max_value = self.selected_scenario.unhealthy_threshold * 5.0
            min_value = self.selected_scenario.unhealthy_threshold * 0.1
            
        # Create date range for the entire scenario period
        start_date = date(self.selected_scenario.year_start, 1, 1)
        end_date = date(self.selected_scenario.year_end, 12, 31)
        days = (end_date - start_date).days + 1
        
        # Create dates list
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Generate random values with seasonal patterns
        values = []
        for d in dates:
            # Add seasonal effect - higher values in summer, lower in winter for PM2.5
            month = d.month
            if param_name in ["pm25", "pm10"]:
                # PM is often worse in winter due to inversions/wood burning
                seasonal_effect = 1.0 + 0.5 * (abs(month - 1) / 11.0) 
            elif param_name == "o3":
                # O3 is often worse in summer due to heat and sunlight
                seasonal_effect = 1.0 + 0.6 * (1.0 - abs(month - 7) / 6.0)
            else:
                seasonal_effect = 1.0
                
            # Random value with normal distribution
            base_value = np.random.normal(mean_value * seasonal_effect, std_dev)
            
            # Ensure value is within realistic bounds
            value = max(min_value, min(base_value, max_value))
            values.append(value)
        
        # Create DataFrame
        df = pd.DataFrame({
            'day': dates,
            'avg': values
        })
        
        # Add classification label based on unhealthy threshold
        df[self.ml_target_label] = np.where(
            df['avg'] <= self.selected_scenario.unhealthy_threshold, 
            0,  # Healthy
            1   # Unhealthy
        )
        
        # Ensure reasonable class distribution (about 25% unhealthy days)
        unhealthy_count = df[self.ml_target_label].sum()
        unhealthy_ratio = unhealthy_count / len(df)
        
        logger.info(f"Initial synthetic data has {unhealthy_count} unhealthy days ({unhealthy_ratio:.1%})")
        
        if unhealthy_ratio < 0.10:
            # Not enough unhealthy days, adjust some values to be higher
            healthy_indices = df[df[self.ml_target_label] == 0].index
            num_to_adjust = int(0.15 * len(df) - unhealthy_count)
            
            if num_to_adjust > 0 and len(healthy_indices) > 0:
                indices_to_adjust = np.random.choice(
                    healthy_indices, 
                    size=min(num_to_adjust, len(healthy_indices)), 
                    replace=False
                )
                
                for idx in indices_to_adjust:
                    # Set to a value just above the unhealthy threshold
                    df.at[idx, 'avg'] = self.selected_scenario.unhealthy_threshold * random.uniform(1.05, 1.5)
                    df.at[idx, self.ml_target_label] = 1
                
                logger.info(f"Adjusted {len(indices_to_adjust)} values to ensure sufficient unhealthy days")
        
        # Save to file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        logger.info(f"Created synthetic data with {len(df)} days and saved to {filename}")
        
        # Add a dummy sensor ID to the scenario to indicate we have data
        self.selected_scenario.open_aq_sensor_ids.append(-1)  # Use -1 to indicate synthetic data
        
        return df
    
    def __init__(
        self, 
        ml_target_label: str = DEFAULT_ML_TARGET_LABEL, 
        ml_eval_metric: str = DEFAULT_ML_EVAL_METRIC, 
        ml_time_limit_secs: Optional[int] = DEFAULT_ML_TIME_LIMIT_SECS
    ):
        """
        Initialize the AQI prediction application.
        
        Args:
            ml_target_label: Target label for ML model
            ml_eval_metric: Evaluation metric for model
            ml_time_limit_secs: Time limit for model training
        """
        self.ml_target_label = ml_target_label
        self.ml_eval_metric = ml_eval_metric
        self.ml_time_limit_secs = ml_time_limit_secs
        self.ml_ignore_columns = ['DATE', 'NAME', 'LATITUDE', 'LONGITUDE', 'day', 'avg']
        
        self.default_columns_noaa = [
            'DATE', 'NAME', 'LATITUDE', 'LONGITUDE',
            'DEWP', 'WDSP', 'MAX', 'MIN', 'PRCP', 'MONTH'
        ]
        
        self.aq_params: Dict[str, AQParam] = {}
        self.aq_scenarios: Dict[str, AQScenario] = {}
        
        self.selected_scenario: Optional[AQScenario] = None
        
        # OpenAQ API key from environment
        self.api_key = os.environ.get('OPENAQ_API_KEY', '')
    
    def add_aq_param(self, aq_param: AQParam) -> bool:
        """
        Add an air quality parameter to the application.
        
        Args:
            aq_param: AQParam object to add
            
        Returns:
            True if successfully added, False otherwise
        """
        if aq_param and aq_param.is_valid():
            self.aq_params[aq_param.name] = aq_param
            return True
        else:
            return False
    
    def add_aq_scenario(self, aq_scenario: AQScenario) -> bool:
        """
        Add an air quality scenario to the application.
        
        Args:
            aq_scenario: AQScenario object to add
            
        Returns:
            True if successfully added, False otherwise
        """
        if aq_scenario and aq_scenario.is_valid():
            self.aq_scenarios[aq_scenario.name] = aq_scenario
            if self.selected_scenario is None:
                self.selected_scenario = self.aq_scenarios[next(iter(self.aq_scenarios))]
            return True
        else:
            return False
    
    def select_scenario(self, scenario_name: str) -> bool:
        """
        Select a scenario by name.
        
        Args:
            scenario_name: Name of the scenario to select
            
        Returns:
            True if successful, False otherwise
        """
        if scenario_name in self.aq_scenarios:
            self.selected_scenario = self.aq_scenarios[scenario_name]
            return True
        return False
    
    def get_filename_noaa(self) -> str:
        """Get the filename for NOAA data."""
        if self and self.selected_scenario and self.selected_scenario.is_valid():
            return os.path.join(
                RAW_DATA_DIR, 
                f"noaa_{self.selected_scenario.name}_{self.selected_scenario.year_start}-"
                f"{self.selected_scenario.year_end}_{self.selected_scenario.noaa_station_id}.csv"
            )
        else:
            return ""
    
    def get_filename_openaq(self) -> str:
        """Get the filename for OpenAQ data."""
        if self and self.selected_scenario and self.selected_scenario.is_valid():
            # Use a temporary filename when no sensors are selected yet
            if len(self.selected_scenario.open_aq_sensor_ids) == 0:
                return os.path.join(
                    RAW_DATA_DIR,
                    f"openaq_{self.selected_scenario.name}_{self.selected_scenario.year_start}-"
                    f"{self.selected_scenario.year_end}_temp.csv"
                )
            
            id_string = "-".join(str(id) for id in self.selected_scenario.open_aq_sensor_ids)
            return os.path.join(
                RAW_DATA_DIR,
                f"openaq_{self.selected_scenario.name}_{self.selected_scenario.year_start}-"
                f"{self.selected_scenario.year_end}_{id_string}.csv"
            )
        else:
            return os.path.join(RAW_DATA_DIR, "empty_openaq.csv")
    
    def get_filename_other(self, prefix: str) -> str:
        """Get a filename with custom prefix."""
        if self and self.selected_scenario and self.selected_scenario.is_valid():
            return os.path.join(
                PROCESSED_DATA_DIR,
                f"{prefix}_{self.selected_scenario.name}_{self.selected_scenario.year_start}-"
                f"{self.selected_scenario.year_end}.csv"
            )
        else:
            return ""
    
    def get_noaa_data(self) -> pd.DataFrame:
        """
        Get NOAA GSOD data for the selected scenario.
        
        Returns:
            DataFrame with NOAA weather data
        """
        noaagsod_df = pd.DataFrame()
        filename_noaa = self.get_filename_noaa()

        if os.path.exists(filename_noaa):
            # Use local data file already accessed + prepared
            logger.info(f'Loading NOAA GSOD data from local file: {filename_noaa}')
            noaagsod_df = pd.read_csv(filename_noaa)
        else:
            if not self.selected_scenario:
                logger.error("No scenario selected")
                return noaagsod_df
                
            # Access + prepare data and save to a local data file
            noaagsod_bucket = NOAA_BUCKET
            logger.info(f'Accessing NOAA GSOD dataset from AWS Open Data Registry (bucket: {noaagsod_bucket})...')
            s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED, connect_timeout=5, read_timeout=10))

            os.makedirs(os.path.dirname(filename_noaa), exist_ok=True)
            
            try:
                for year in range(self.selected_scenario.year_start, self.selected_scenario.year_end + 1):
                    key = f'{year}/{self.selected_scenario.noaa_station_id}.csv'
                    logger.info(f"Retrieving NOAA data for year {year}")
                    csv_obj = s3.get_object(Bucket=noaagsod_bucket, Key=key)
                    csv_string = csv_obj['Body'].read().decode('utf-8')
                    noaagsod_df = pd.concat(
                        [noaagsod_df, pd.read_csv(StringIO(csv_string))], 
                        ignore_index=True
                    )

                # Feature Engineering
                # Extract date components for seasonality
                noaagsod_df['MONTH'] = pd.to_datetime(noaagsod_df['DATE']).dt.month
                noaagsod_df['DAYOFWEEK'] = pd.to_datetime(noaagsod_df['DATE']).dt.dayofweek
                noaagsod_df['SEASON'] = pd.to_datetime(noaagsod_df['DATE']).dt.month.map({
                    1: 'Winter', 2: 'Winter', 3: 'Spring', 
                    4: 'Spring', 5: 'Spring', 6: 'Summer',
                    7: 'Summer', 8: 'Summer', 9: 'Fall', 
                    10: 'Fall', 11: 'Fall', 12: 'Winter'
                })
                
                # Calculate temperature differences and averages
                noaagsod_df['TEMP_RANGE'] = noaagsod_df['MAX'] - noaagsod_df['MIN']
                noaagsod_df['TEMP_AVG'] = (noaagsod_df['MAX'] + noaagsod_df['MIN']) / 2
                
                # Create interaction features
                noaagsod_df['TEMP_DEWP_DIFF'] = noaagsod_df['TEMP_AVG'] - noaagsod_df['DEWP']
                noaagsod_df['WDSP_TEMP'] = noaagsod_df['WDSP'] * noaagsod_df['TEMP_AVG']
                
                # Add VISIB_ATTRIBUTES to the feature engineering logic
                noaagsod_df['VISIB_ATTRIBUTES'] = '0'  # Default attribute value for visibility
                
                # Update scenario with station coordinates
                if not noaagsod_df.empty and 'LATITUDE' in noaagsod_df.columns and 'LONGITUDE' in noaagsod_df.columns:
                    first_row = noaagsod_df.iloc[0]
                    self.selected_scenario.update_noaa_station_coords(
                        first_row['LATITUDE'], 
                        first_row['LONGITUDE']
                    )
                
                # Save to file
                noaagsod_df.to_csv(filename_noaa, index=False)
                logger.info(f"NOAA data saved to {filename_noaa}")
            
            except Exception as e:
                logger.error(f"Error fetching NOAA data: {str(e)}")
            
        return noaagsod_df
    
    def get_openaq_data(self) -> pd.DataFrame:
        """
        Get OpenAQ data for the selected scenario.
        
        Returns:
            DataFrame with OpenAQ air quality data
        """
        if not self.api_key:
            logger.warning("OPENAQ_API_KEY not set in environment variables")
            
        aq_df = pd.DataFrame()
        if not self.selected_scenario:
            logger.error("No scenario selected")
            return aq_df
            
        headers = {
            'accept': 'application/json',
            'x-api-key': self.api_key
        }

        if self.selected_scenario.noaa_station_lat == 0.0 or self.selected_scenario.noaa_station_lng == 0.0:
            # Try to get coordinates from NOAA data
            logger.info("NOAA station coordinates not defined. Attempting to fetch from NOAA data...")
            noaa_data = self.get_noaa_data()
            
            # Check if we now have coordinates
            if self.selected_scenario.noaa_station_lat == 0.0 or self.selected_scenario.noaa_station_lng == 0.0:
                logger.error("NOAA Station Lat/Lng STILL NOT DEFINED. Cannot proceed")
                return aq_df
            
            logger.info(f"Retrieved coordinates: {self.selected_scenario.noaa_station_lat}, {self.selected_scenario.noaa_station_lng}")
        
        filename_openaq = self.get_filename_openaq()
        
        if len(self.selected_scenario.open_aq_sensor_ids) == 0:
            # Find OpenAQ sensors near the NOAA station location
            logger.info('Finding OpenAQ sensors near NOAA station location...')
            
            # Try with initial radius
            initial_radius = 25000  # 25km radius
            
            # Initialize sensors dictionary to store sensor info including coordinates
            sensor_info = {}
            
            # Try progressively larger search radii if no sensors found
            for search_radius in [initial_radius, 50000, 100000, 200000]:  # 25km, 50km, 100km, 200km
                # Query OpenAQ locations API with coordinates
                aq_req_params = {
                    'coordinates': f"{self.selected_scenario.noaa_station_lat},{self.selected_scenario.noaa_station_lng}",
                    'radius': search_radius,
                    'parameter': self.selected_scenario.aq_param_target.name,
                    'limit': 100
                }
                
                try:
                    logger.info(f"Searching with radius: {search_radius/1000}km")
                    aq_resp = requests.get(f"{OPENAQ_API_URL}/locations", params=aq_req_params, headers=headers, timeout=10)
                    aq_data = aq_resp.json()
                    
                    if 'results' in aq_data and aq_data['results']:
                        for location in aq_data['results']:
                            # Check each location's sensors for our target parameter
                            for sensor in location['sensors']:
                                if sensor['parameter']['name'] == self.selected_scenario.aq_param_target.name:
                                    sensor_id = sensor['id']
                                    if sensor_id not in sensor_info:
                                        # Store sensor info with coordinates for later spatial interpolation
                                        sensor_info[sensor_id] = {
                                            'id': sensor_id,
                                            'latitude': location['coordinates']['latitude'],
                                            'longitude': location['coordinates']['longitude'],
                                            'distance': self._calculate_distance(
                                                self.selected_scenario.noaa_station_lat, 
                                                self.selected_scenario.noaa_station_lng,
                                                location['coordinates']['latitude'], 
                                                location['coordinates']['longitude']
                                            )
                                        }
                        
                        # If we found sensors, no need to try larger radius
                        if sensor_info:
                            break
                except Exception as e:
                    logger.error(f"Error finding OpenAQ sensors with radius {search_radius/1000}km: {str(e)}")
            
            # Add the sensors to our scenario, sorted by distance
            if sensor_info:
                # Sort sensors by distance
                sorted_sensors = sorted(sensor_info.values(), key=lambda x: x['distance'])
                
                # Store the distance info for later weighting
                self.selected_scenario.sensor_distances = {s['id']: s['distance'] for s in sorted_sensors}
                
                # Add sensor IDs to the scenario
                for sensor in sorted_sensors:
                    self.selected_scenario.open_aq_sensor_ids.append(sensor['id'])
                
                logger.info(
                    f'Found {len(self.selected_scenario.open_aq_sensor_ids)} OpenAQ '
                    f'locations with {self.selected_scenario.aq_param_target.name} sensors '
                    f'within {search_radius/1000}km'
                )
                
                # Log the sensors found with distances
                for sensor_id, info in self.selected_scenario.sensor_distances.items():
                    logger.info(f"Sensor ID: {sensor_id}, Distance: {info:.2f}km")
            else:
                logger.warning(
                    f"No OpenAQ sensors with {self.selected_scenario.aq_param_target.name} parameter "
                    f"found within 200km. Using fallback method with synthetic data."
                )
                # Create synthetic data based on typical values for this parameter
                self._create_synthetic_data(filename_openaq)
        
        # Check if we have a local file first
        if os.path.exists(filename_openaq):
            # Use local data file already accessed + prepared
            logger.info(f'Loading OpenAQ data from local file: {filename_openaq}')
            aq_df = pd.read_csv(filename_openaq)
        # Otherwise, if we found sensors or need to generate synthetic data
        elif len(self.selected_scenario.open_aq_sensor_ids) >= 1:
                # Access + prepare data (one year at a time to avoid timeouts)
                logger.info('Accessing OpenAQ Measurements API...')
                
                os.makedirs(os.path.dirname(filename_openaq), exist_ok=True)
                
                try:
                    # Dictionary to store all measurements by date, sensor
                    # Structure: {day: {sensor_id: value}}
                    all_measurements = {}
                    
                    for year in range(self.selected_scenario.year_start, self.selected_scenario.year_end + 1):
                        for sensor_id in self.selected_scenario.open_aq_sensor_ids:
                            # Get daily measurements for this sensor and year
                            aq_req_url = f"{OPENAQ_API_URL}/sensors/{sensor_id}/days"
                            aq_req_params = {
                                'date_from': f'{year}-01-01',
                                'date_to': f'{year}-12-31',
                                'limit': 366
                            }
                            
                            logger.info(f'Fetching data for sensor {sensor_id} in {year}')
                            aq_resp = requests.get(aq_req_url, params=aq_req_params, headers=headers, timeout=10)
                            aq_data = aq_resp.json()
                            
                            if 'results' in aq_data:
                                for measurement in aq_data['results']:
                                    dt = datetime.strptime(
                                        measurement['period']['datetimeFrom']['utc'], 
                                        '%Y-%m-%dT%H:%M:%SZ'
                                    )
                                    day = dt.date()
                                    
                                    if measurement['value'] is not None:
                                        # Initialize day if not already in dictionary
                                        if day not in all_measurements:
                                            all_measurements[day] = {}
                                            
                                        # Store this measurement
                                        all_measurements[day][sensor_id] = measurement['value']
                    
                    # Apply inverse distance weighting (IDW) for spatial interpolation
                    if all_measurements:
                        logger.info("Applying spatial interpolation with inverse distance weighting (IDW)")
                        
                        # Power parameter for IDW (adjusts how quickly influence decreases with distance)
                        p = 2
                        
                        for day, sensors_data in all_measurements.items():
                            if sensors_data:  # If we have any sensor data for this day
                                # If we have multiple sensors, use IDW
                                if len(sensors_data) > 1:
                                    weights_sum = 0
                                    weighted_values_sum = 0
                                    
                                    for sensor_id, value in sensors_data.items():
                                        # Get distance from scenario's sensor_distances dictionary
                                                                                    # Make sure sensor_distances exists
                                            if not hasattr(self.selected_scenario, 'sensor_distances'):
                                                self.selected_scenario.sensor_distances = {}
                                                
                                            if sensor_id in self.selected_scenario.sensor_distances:
                                                distance = self.selected_scenario.sensor_distances[sensor_id]
                                            
                                            # Avoid division by zero for very small distances
                                            if distance < 0.1:
                                                distance = 0.1
                                                
                                            # Calculate weight as 1/(distance^p)
                                            weight = 1.0 / (distance ** p)
                                            weights_sum += weight
                                            weighted_values_sum += value * weight
                                    
                                    # Calculate the IDW value
                                    if weights_sum > 0:
                                        interpolated_value = weighted_values_sum / weights_sum
                                        
                                        # Add to DataFrame
                                        date_df = pd.DataFrame({
                                            'day': [day],
                                            'avg': [interpolated_value]
                                        })
                                        aq_df = pd.concat([aq_df, date_df], ignore_index=True)
                                else:
                                    # If only one sensor, use its value directly
                                    sensor_id = next(iter(sensors_data))
                                    value = sensors_data[sensor_id]
                                    
                                    date_df = pd.DataFrame({
                                        'day': [day],
                                        'avg': [value]
                                    })
                                    aq_df = pd.concat([aq_df, date_df], ignore_index=True)
                    
                    # Add classification label
                    if not aq_df.empty:
                        # Add classification label
                        aq_df[self.ml_target_label] = np.where(
                            aq_df['avg'] <= self.selected_scenario.unhealthy_threshold, 
                            0,  # Healthy
                            1   # Unhealthy
                        )
                        
                        # Save to file
                        aq_df.to_csv(filename_openaq, index=False)
                        logger.info(f"OpenAQ data with spatial interpolation saved to {filename_openaq}")
                
                except Exception as e:
                    logger.error(f"Error fetching OpenAQ data: {str(e)}")
        
        return aq_df
    
    def get_merged_data(self, noaagsod_df: pd.DataFrame, aq_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge NOAA and OpenAQ data.
        
        Args:
            noaagsod_df: NOAA GSOD data
            aq_df: OpenAQ data
            
        Returns:
            Merged DataFrame
        """
        if len(noaagsod_df) > 0 and len(aq_df) > 0:
            # Print shapes before merge for debugging
            logger.info(f"NOAA GSOD shape before merge: {noaagsod_df.shape}")
            logger.info(f"AQ data shape before merge: {aq_df.shape}")
            
            # Convert DATE to datetime if it's not already
            if noaagsod_df['DATE'].dtype != 'datetime64[ns]':
                noaagsod_df['DATE'] = pd.to_datetime(noaagsod_df['DATE'])
            
            # Convert day to datetime if it's not already
            if aq_df['day'].dtype != 'datetime64[ns]':
                aq_df['day'] = pd.to_datetime(aq_df['day'])
            
            # Merge the data
            merged_df = pd.merge(noaagsod_df, aq_df, how="inner", left_on="DATE", right_on="day")
            
            # Print shape after merge
            logger.info(f"Merged shape: {merged_df.shape}")
            
            if len(merged_df) == 0:
                logger.warning("Merge resulted in empty DataFrame. No matching dates between datasets.")
                logger.info(f"DATE dtype: {noaagsod_df['DATE'].dtype}")
                logger.info(f"day dtype: {aq_df['day'].dtype}")
                
                # Try to convert string date formats and merge again
                try:
                    noaagsod_df['DATE_str'] = noaagsod_df['DATE'].dt.strftime('%Y-%m-%d')
                    aq_df['day_str'] = aq_df['day'].dt.strftime('%Y-%m-%d')
                    merged_df = pd.merge(noaagsod_df, aq_df, how="inner", left_on="DATE_str", right_on="day_str")
                    
                    if len(merged_df) > 0:
                        logger.info(f"Merge successful after string conversion: {merged_df.shape}")
                    else:
                        logger.warning("Merge still resulted in empty DataFrame after string conversion.")
                        return pd.DataFrame()
                except Exception as e:
                    logger.error(f"Error during string date conversion: {str(e)}")
                    return pd.DataFrame()
                    
            # Save merged data
            merged_filename = self.get_filename_other("merged")
            if merged_filename:
                os.makedirs(os.path.dirname(merged_filename), exist_ok=True)
                merged_df.to_csv(merged_filename, index=False)
                logger.info(f"Merged data saved to {merged_filename}")
            
            # Drop columns not needed for modeling, but only if they exist
            columns_to_drop = [col for col in self.ml_ignore_columns if col in merged_df.columns]
            if columns_to_drop:
                logger.info(f"Dropping columns: {columns_to_drop}")
                merged_df = merged_df.drop(columns=columns_to_drop)
            return merged_df
        else:
            logger.warning("Cannot merge: one or both DataFrames are empty")
            return pd.DataFrame()
    
    def prepare_train_test_data(
        self, 
        merged_df: pd.DataFrame, 
        test_size: float = 0.2, 
        validation_size: float = 0.25
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            merged_df: Merged data
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            
        Returns:
            Tuple of (train_df, validation_df, test_df)
        """
        if merged_df.empty:
            logger.warning("Cannot prepare data: DataFrame is empty")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Shuffle the data
        shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into training and test
        train_test_split_idx = int(len(shuffled_df) * (1 - test_size))
        train_validation_df = shuffled_df.iloc[:train_test_split_idx]
        test_df = shuffled_df.iloc[train_test_split_idx:]
        
        # Split training into train and validation
        train_validation_split_idx = int(len(train_validation_df) * (1 - validation_size))
        train_df = train_validation_df.iloc[:train_validation_split_idx]
        validation_df = train_validation_df.iloc[train_validation_split_idx:]
        
        # Log class distribution
        logger.info(f"Training set shape: {train_df.shape}")
        logger.info(f"Validation set shape: {validation_df.shape}")
        logger.info(f"Test set shape: {test_df.shape}")
        
        if self.ml_target_label in train_df.columns:
            logger.info(
                f"Training set class distribution: "
                f"{train_df[self.ml_target_label].value_counts(normalize=True)}"
            )
        
        # Save split data
        train_filename = self.get_filename_other("train")
        validation_filename = self.get_filename_other("validation")
        test_filename = self.get_filename_other("test")
        
        if train_filename and validation_filename and test_filename:
            train_df.to_csv(train_filename, index=False)
            validation_df.to_csv(validation_filename, index=False)
            test_df.to_csv(test_filename, index=False)
            logger.info(f"Split data saved to {os.path.dirname(train_filename)}")
        
        return train_df, validation_df, test_df 
    
    def prepare_features(self, noaa_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction by adding derived features and ensuring all 
        necessary columns are present.
        
        Args:
            noaa_df: DataFrame with NOAA weather data
            
        Returns:
            DataFrame with prepared features ready for prediction
        """
        logger.info("Preparing features for prediction")
        input_df = noaa_df.copy()
        
        # Add derived features
        if 'TEMP_RANGE' not in input_df.columns and 'MAX' in input_df.columns and 'MIN' in input_df.columns:
            input_df['TEMP_RANGE'] = input_df['MAX'] - input_df['MIN']
            logger.info("Added TEMP_RANGE feature")
        
        if 'TEMP_AVG' not in input_df.columns and 'MAX' in input_df.columns and 'MIN' in input_df.columns:
            input_df['TEMP_AVG'] = (input_df['MAX'] + input_df['MIN']) / 2
            logger.info("Added TEMP_AVG feature")
        
        if 'TEMP_DEWP_DIFF' not in input_df.columns and 'TEMP_AVG' in input_df.columns and 'DEWP' in input_df.columns:
            input_df['TEMP_DEWP_DIFF'] = input_df['TEMP_AVG'] - input_df['DEWP']
            logger.info("Added TEMP_DEWP_DIFF feature")
        
        if 'WDSP_TEMP' not in input_df.columns and 'WDSP' in input_df.columns and 'TEMP_AVG' in input_df.columns:
            input_df['WDSP_TEMP'] = input_df['WDSP'] * input_df['TEMP_AVG']
            logger.info("Added WDSP_TEMP feature")
        
        # Add date-related features
        if 'DATE' in input_df.columns:
            # Convert to datetime if not already
            if input_df['DATE'].dtype != 'datetime64[ns]':
                input_df['DATE'] = pd.to_datetime(input_df['DATE'])
                logger.info("Converted DATE to datetime")
            
            # Add month and day of week
            if 'MONTH' not in input_df.columns:
                input_df['MONTH'] = input_df['DATE'].dt.month
                logger.info("Added MONTH feature")
            
            if 'DAYOFWEEK' not in input_df.columns:
                input_df['DAYOFWEEK'] = input_df['DATE'].dt.dayofweek
                logger.info("Added DAYOFWEEK feature")
        
        # Add season if not present
        if 'SEASON' not in input_df.columns and 'MONTH' in input_df.columns:
            season_map = {
                1: 'Winter', 2: 'Winter', 3: 'Spring', 
                4: 'Spring', 5: 'Spring', 6: 'Summer',
                7: 'Summer', 8: 'Summer', 9: 'Fall', 
                10: 'Fall', 11: 'Fall', 12: 'Winter'
            }
            input_df['SEASON'] = input_df['MONTH'].map(season_map)
            logger.info("Added SEASON feature")
        
        # Add missing attribute columns that may be needed by the model
        for attr_col in ['TEMP_ATTRIBUTES', 'DEWP_ATTRIBUTES', 'SLP_ATTRIBUTES', 
                         'STP_ATTRIBUTES', 'VISIB_ATTRIBUTES', 'WDSP_ATTRIBUTES', 
                         'MAX_ATTRIBUTES', 'MIN_ATTRIBUTES', 'PRCP_ATTRIBUTES']:
            if attr_col not in input_df.columns:
                input_df[attr_col] = '0'
                logger.info(f"Added missing {attr_col} column")
        
        logger.info(f"Feature preparation complete. DataFrame shape: {input_df.shape}")
        return input_df 