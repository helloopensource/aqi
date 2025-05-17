export interface AQScenario {
  location: string;
  name: string;
  noaa_station_id: string;
  noaa_station_lat: number;
  noaa_station_lng: number;
  open_aq_sensor_ids: number[];
  unhealthy_threshold: number;
  year_start: number;
  year_end: number;
  aq_radius_miles: number;
  target_param: string;
  target_param_desc: string;
}

export interface AQICategory {
  name: string;
  range: string;
  color: string;
  health_implications: string;
  cautionary_statement: string;
}

export interface AQICategoryInfo {
  [key: string]: {
    color: string;
    class: string;
  };
}

export interface PredictionResponse {
  scenario: string;
  date: string;
  is_unhealthy: boolean;
  probability: number | null;
  aqi_value: number | null;
  category: string | null;
  health_implications: string | null;
  prediction_time: string;
}

export interface WeatherData {
  DEWP: number;
  WDSP: number;
  MAX: number;
  MIN: number;
  PRCP: number;
  MONTH: number;
  DAYOFWEEK: number;
  TEMP_RANGE: number;
  TEMP_AVG: number;
  TEMP_DEWP_DIFF: number;
  WDSP_TEMP: number;
}

export interface PredictionRequest {
  scenario: string;
  date: string;
  weather_data: WeatherData;
}

export interface CityImageData {
  city: string;
  image: string;
  altText: string;
}

export const cityImages: { [key: string]: CityImageData } = {
  "los-angeles": {
    city: "Los Angeles",
    image: "https://images.unsplash.com/photo-1580655653885-65763b2597d0?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=80",
    altText: "Los Angeles skyline with palm trees"
  },
  "san-francisco": {
    city: "San Francisco",
    image: "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=80",
    altText: "San Francisco Golden Gate Bridge in fog"
  },
  "new-york": {
    city: "New York",
    image: "https://images.unsplash.com/photo-1522083165195-3424ed129620?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=80",
    altText: "New York City skyline"
  },
  "chicago": {
    city: "Chicago",
    image: "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=80",
    altText: "Chicago skyline with river"
  }
};

export const aqiCategoryInfo: AQICategoryInfo = {
  "Good": {
    color: "#00E400",
    class: "good"
  },
  "Moderate": {
    color: "#FFFF00",
    class: "moderate"
  },
  "Unhealthy for Sensitive Groups": {
    color: "#FF7E00",
    class: "sensitive"
  },
  "Unhealthy": {
    color: "#FF0000",
    class: "unhealthy"
  },
  "Very Unhealthy": {
    color: "#8F3F97",
    class: "very-unhealthy"
  },
  "Hazardous": {
    color: "#7E0023",
    class: "hazardous"
  }
}; 