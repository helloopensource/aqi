"""
Utility functions for calculating AQI values and categories.
"""
from typing import Dict, Tuple, Optional, Any

from ..config.settings import AQI_BREAKPOINTS


# AQI category information
AQI_CATEGORIES = {
    (0, 50): {
        "name": "Good",
        "color": "#00E400",
        "health_implications": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "cautionary_statement": "None"
    },
    (51, 100): {
        "name": "Moderate",
        "color": "#FFFF00",
        "health_implications": "Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people who are unusually sensitive to air pollution.",
        "cautionary_statement": "Active children and adults, and people with respiratory disease, such as asthma, should limit prolonged outdoor exertion."
    },
    (101, 150): {
        "name": "Unhealthy for Sensitive Groups",
        "color": "#FF7E00",
        "health_implications": "Members of sensitive groups may experience health effects. The general public is not likely to be affected.",
        "cautionary_statement": "Active children and adults, and people with respiratory disease, such as asthma, should limit prolonged outdoor exertion."
    },
    (151, 200): {
        "name": "Unhealthy",
        "color": "#FF0000",
        "health_implications": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.",
        "cautionary_statement": "Active children and adults, and people with respiratory disease, such as asthma, should avoid prolonged outdoor exertion; everyone else, especially children, should limit prolonged outdoor exertion."
    },
    (201, 300): {
        "name": "Very Unhealthy",
        "color": "#8F3F97",
        "health_implications": "Health warnings of emergency conditions. The entire population is more likely to be affected.",
        "cautionary_statement": "Active children and adults, and people with respiratory disease, such as asthma, should avoid all outdoor exertion; everyone else, especially children, should limit outdoor exertion."
    },
    (301, 500): {
        "name": "Hazardous",
        "color": "#7E0023",
        "health_implications": "Health alert: everyone may experience more serious health effects.",
        "cautionary_statement": "Everyone should avoid all outdoor exertion."
    }
}


def calculate_aqi(concentration: float, pollutant: str) -> Tuple[int, Dict[str, Any]]:
    """
    Calculate AQI value from pollutant concentration.
    
    Args:
        concentration: Pollutant concentration
        pollutant: Pollutant name (e.g., 'pm25', 'pm10')
        
    Returns:
        Tuple of (AQI value, category information)
    """
    if pollutant not in AQI_BREAKPOINTS:
        raise ValueError(f"Unsupported pollutant: {pollutant}")
    
    breakpoints = AQI_BREAKPOINTS[pollutant]
    
    # Find which breakpoint range the concentration falls into
    for conc_range, aqi_range in breakpoints.items():
        c_low, c_high = conc_range
        i_low, i_high = aqi_range
        
        if c_low <= concentration <= c_high:
            # Calculate AQI using linear interpolation
            aqi = round(((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low)
            
            # Get category info
            category = None
            for aqi_cat_range, cat_info in AQI_CATEGORIES.items():
                cat_low, cat_high = aqi_cat_range
                if cat_low <= aqi <= cat_high:
                    category = {
                        **cat_info,
                        "range": f"{cat_low}-{cat_high}"
                    }
                    break
            
            return aqi, category
    
    # If concentration is above the highest breakpoint, return the highest AQI
    max_aqi = 500
    category = {
        **AQI_CATEGORIES[(301, 500)],
        "range": "301-500"
    }
    
    # For concentrations below the lowest breakpoint, return the lowest AQI
    if concentration < min(breakpoints.keys())[0]:
        min_aqi_range = breakpoints[min(breakpoints.keys())]
        min_aqi = min_aqi_range[0]
        category = {
            **AQI_CATEGORIES[(0, 50)],
            "range": "0-50"
        }
        return min_aqi, category
    
    return max_aqi, category


def get_category_from_aqi(aqi: int) -> Dict[str, Any]:
    """
    Get category information from AQI value.
    
    Args:
        aqi: AQI value
        
    Returns:
        Category information dictionary
    """
    for aqi_range, cat_info in AQI_CATEGORIES.items():
        low, high = aqi_range
        if low <= aqi <= high:
            return {
                **cat_info,
                "range": f"{low}-{high}"
            }
    
    # Default to hazardous for values above 500
    return {
        **AQI_CATEGORIES[(301, 500)],
        "range": "301-500"
    }


def is_unhealthy(aqi: int) -> bool:
    """
    Determine if AQI is in the unhealthy range.
    
    Args:
        aqi: AQI value
        
    Returns:
        True if AQI is greater than 100 (Unhealthy for Sensitive Groups or worse)
    """
    return aqi > 100 