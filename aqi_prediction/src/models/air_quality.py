"""
Air Quality Parameter and Scenario models
"""
import json
from typing import List, Optional, Dict, Any

from ..config.settings import UNHEALTHY_THRESHOLDS


class AQParam:
    """
    Air Quality Parameter class to define attributes for the main OpenAQ parameters.
    """
    def __init__(
        self, 
        id: int, 
        name: str, 
        unit: str, 
        unhealthy_threshold_default: float, 
        desc: str
    ):
        """
        Initialize an air quality parameter.
        
        Args:
            id: Parameter identifier
            name: Parameter name (e.g., 'pm25')
            unit: Measurement unit (e.g., 'µg/m³')
            unhealthy_threshold_default: Default threshold for unhealthy classification
            desc: Description of the parameter
        """
        self.id = id
        self.name = name
        self.unit = unit
        self.unhealthy_threshold_default = unhealthy_threshold_default
        self.desc = desc
    
    def is_valid(self) -> bool:
        """Check if the parameter is valid."""
        return (
            self is not None and 
            self.id > 0 and 
            self.unhealthy_threshold_default > 0.0 and 
            len(self.name) > 0 and 
            len(self.unit) > 0 and 
            len(self.desc) > 0
        )
            
    def to_json(self) -> str:
        """Convert to JSON representation."""
        return json.dumps(self.__dict__, sort_keys=True, indent=2)
    
    @classmethod
    def get_default_params(cls) -> Dict[str, 'AQParam']:
        """
        Get default parameters from predefined settings.
        
        Returns:
            Dictionary of AQParam objects keyed by name
        """
        params = {}
        
        # PM2.5 - Particulate Matter < 2.5 micrometers
        params["pm25"] = cls(
            id=1, 
            name="pm25", 
            unit="µg/m³", 
            unhealthy_threshold_default=UNHEALTHY_THRESHOLDS["pm25"],
            desc="Particulate Matter < 2.5 micrometers"
        )
        
        # PM10 - Particulate Matter < 10 micrometers
        params["pm10"] = cls(
            id=2, 
            name="pm10", 
            unit="µg/m³", 
            unhealthy_threshold_default=UNHEALTHY_THRESHOLDS["pm10"],
            desc="Particulate Matter < 10 micrometers"
        )
        
        # O3 - Ozone
        params["o3"] = cls(
            id=3, 
            name="o3", 
            unit="ppm", 
            unhealthy_threshold_default=UNHEALTHY_THRESHOLDS["o3"],
            desc="Ozone"
        )
        
        # NO2 - Nitrogen Dioxide
        params["no2"] = cls(
            id=4, 
            name="no2", 
            unit="ppm", 
            unhealthy_threshold_default=UNHEALTHY_THRESHOLDS["no2"],
            desc="Nitrogen Dioxide"
        )
        
        # SO2 - Sulfur Dioxide
        params["so2"] = cls(
            id=5, 
            name="so2", 
            unit="ppm", 
            unhealthy_threshold_default=UNHEALTHY_THRESHOLDS["so2"],
            desc="Sulfur Dioxide"
        )
        
        # CO - Carbon Monoxide
        params["co"] = cls(
            id=6, 
            name="co", 
            unit="ppm", 
            unhealthy_threshold_default=UNHEALTHY_THRESHOLDS["co"],
            desc="Carbon Monoxide"
        )
        
        return params


class AQScenario:
    """
    Air Quality Scenario class that defines an ML scenario including 
    location with NOAA Weather Station ID and the target OpenAQ Parameter.
    """
    def __init__(
        self, 
        location: str, 
        noaa_station_id: str, 
        aq_param_target: AQParam, 
        unhealthy_threshold: Optional[float] = None, 
        year_start: int = 2016, 
        year_end: int = 2024, 
        aq_radius_miles: int = 10,
        feature_columns_to_drop: Optional[List[str]] = None
    ):
        """
        Initialize an air quality scenario.
        
        Args:
            location: Location name
            noaa_station_id: NOAA station ID
            aq_param_target: Target air quality parameter
            unhealthy_threshold: Custom threshold (overrides default)
            year_start: Start year for data
            year_end: End year for data
            aq_radius_miles: Radius in miles to search for air quality sensors
            feature_columns_to_drop: Columns to exclude from model training
        """
        self.location = location
        self.name = f"{location}_{aq_param_target.name}"
        self.noaa_station_id = noaa_station_id
        self.noaa_station_lat = 0.0
        self.noaa_station_lng = 0.0
        self.open_aq_sensor_ids = []
        
        # Dictionary to store sensor distances for spatial interpolation
        self.sensor_distances = {}
        
        self.aq_param_target = aq_param_target
        
        if unhealthy_threshold and unhealthy_threshold > 0.0:
            self.unhealthy_threshold = unhealthy_threshold
        else:
            self.unhealthy_threshold = self.aq_param_target.unhealthy_threshold_default
        
        self.year_start = year_start
        self.year_end = year_end
        self.aq_radius_miles = aq_radius_miles
        self.aq_radius_meters = aq_radius_miles * 1610  # Rough integer approximation
        
        self.model_folder = "models"
            
    def get_summary(self) -> str:
        """Get a summary of the scenario."""
        return (
            f"Scenario: {self.name} => {self.aq_param_target.desc} "
            f"({self.aq_param_target.name}) with UnhealthyThreshold > "
            f"{self.unhealthy_threshold} {self.aq_param_target.unit}"
        )
    
    def get_model_path(self) -> str:
        """Get the path to store model files."""
        return f"{self.model_folder}/aq_{self.name}_{self.year_start}-{self.year_end}/"
    
    def update_noaa_station_coords(self, latitude: float, longitude: float) -> None:
        """
        Update the NOAA station coordinates.
        
        Args:
            latitude: Latitude of NOAA station
            longitude: Longitude of NOAA station
        """
        self.noaa_station_lat = latitude
        self.noaa_station_lng = longitude
        print(f"NOAA Station Lat,Lng Updated for Scenario: {self.name} => {self.noaa_station_lat},{self.noaa_station_lng}")
    
    def is_valid(self) -> bool:
        """Check if the scenario is valid."""
        return (
            self is not None and 
            self.aq_param_target is not None and
            self.year_start > 0 and 
            self.year_end > 0 and 
            self.year_end >= self.year_start and 
            self.aq_radius_miles > 0 and 
            self.aq_radius_meters > 0 and 
            self.unhealthy_threshold > 0.0 and 
            len(self.name) > 0 and 
            len(self.noaa_station_id) > 0
        )
            
    def to_json(self) -> str:
        """Convert to JSON representation."""
        return json.dumps(self.__dict__, default=lambda o: o.__dict__, sort_keys=True, indent=2)


# Create predefined scenarios
def get_default_scenarios() -> Dict[str, AQScenario]:
    """
    Get default scenarios for major US cities.
    
    Returns:
        Dictionary of AQScenario objects keyed by name
    """
    params = AQParam.get_default_params()
    scenarios = {}
    
    # Los Angeles - PM2.5
    la = AQScenario(
        location="los-angeles",
        noaa_station_id="72295023174",
        aq_param_target=params["pm25"]
    )
    la.update_noaa_station_coords(33.9381, -118.3889)  # LAX Airport coordinates
    scenarios[la.name] = la

    # Los Angeles - PM10
    la_pm10 = AQScenario(
        location="los-angeles",
        noaa_station_id="72295023174",
        aq_param_target=params["pm10"]
    )
    la_pm10.update_noaa_station_coords(33.9381, -118.3889)  # LAX Airport coordinates
    scenarios[la_pm10.name] = la_pm10

    # Los Angeles - O3
    la_o3 = AQScenario(
        location="los-angeles",
        noaa_station_id="72295023174",
        aq_param_target=params["o3"]
    )
    la_o3.update_noaa_station_coords(33.9381, -118.3889)  # LAX Airport coordinates
    scenarios[la_o3.name] = la_o3
    
    # San Francisco - PM2.5
    sf = AQScenario(
        location="san-francisco",
        noaa_station_id="72494023234",
        aq_param_target=params["pm25"]
    )
    sf.update_noaa_station_coords(37.6213, -122.3790)  # SFO Airport coordinates
    scenarios[sf.name] = sf
    
    # San Francisco - PM10
    sf_pm10 = AQScenario(
        location="san-francisco",
        noaa_station_id="72494023234",
        aq_param_target=params["pm10"]
    )
    sf_pm10.update_noaa_station_coords(37.6213, -122.3790)  # SFO Airport coordinates
    scenarios[sf_pm10.name] = sf_pm10
    
    # San Francisco - O3
    sf_o3 = AQScenario(
        location="san-francisco",
        noaa_station_id="72494023234",
        aq_param_target=params["o3"]
    )
    sf_o3.update_noaa_station_coords(37.6213, -122.3790)  # SFO Airport coordinates
    scenarios[sf_o3.name] = sf_o3
    
    # New York - PM2.5
    nyc = AQScenario(
        location="new-york",
        noaa_station_id="72503014732",
        aq_param_target=params["pm25"]
    )
    nyc.update_noaa_station_coords(40.7128, -74.0060)  # NYC coordinates
    scenarios[nyc.name] = nyc
    
    # New York - PM10
    nyc_pm10 = AQScenario(
        location="new-york",
        noaa_station_id="72503014732",
        aq_param_target=params["pm10"]
    )
    nyc_pm10.update_noaa_station_coords(40.7128, -74.0060)  # NYC coordinates
    scenarios[nyc_pm10.name] = nyc_pm10

    # New York - O3
    nyc_o3 = AQScenario(
        location="new-york",
        noaa_station_id="72503014732",
        aq_param_target=params["o3"]
    )
    nyc_o3.update_noaa_station_coords(40.7128, -74.0060)  # NYC coordinates
    scenarios[nyc_o3.name] = nyc_o3
    
    # Chicago - PM2.5
    chicago = AQScenario(
        location="chicago",
        noaa_station_id="72530094846",
        aq_param_target=params["pm25"]
    )
    chicago.update_noaa_station_coords(41.8781, -87.6298)  # Chicago coordinates
    scenarios[chicago.name] = chicago
    
    # Chicago - PM10
    chicago_pm10 = AQScenario(
        location="chicago",
        noaa_station_id="72530094846",
        aq_param_target=params["pm10"]
    )
    chicago_pm10.update_noaa_station_coords(41.8781, -87.6298)  # Chicago coordinates
    scenarios[chicago_pm10.name] = chicago_pm10
    
    # Chicago - O3
    chicago_o3 = AQScenario(
        location="chicago",
        noaa_station_id="72530094846",
        aq_param_target=params["o3"]
    )
    chicago_o3.update_noaa_station_coords(41.8781, -87.6298)  # Chicago coordinates
    scenarios[chicago_o3.name] = chicago_o3
    
    return scenarios 