"""
Tests for the AQI calculator.
"""
import sys
import os
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.aqi_calculator import calculate_aqi, get_category_from_aqi, is_unhealthy


class TestAqiCalculator(unittest.TestCase):
    """Tests for the AQI calculator."""
    
    def test_calculate_aqi_pm25(self):
        """Test PM2.5 AQI calculation."""
        # Test good air quality
        aqi, category = calculate_aqi(5.0, "pm25")
        self.assertEqual(aqi, 21)
        self.assertEqual(category["name"], "Good")
        
        # Test moderate air quality
        aqi, category = calculate_aqi(25.0, "pm25")
        self.assertEqual(aqi, 78)
        self.assertEqual(category["name"], "Moderate")
        
        # Test unhealthy for sensitive groups
        aqi, category = calculate_aqi(40.0, "pm25")
        self.assertEqual(aqi, 112)
        self.assertEqual(category["name"], "Unhealthy for Sensitive Groups")
        
        # Test unhealthy
        aqi, category = calculate_aqi(100.0, "pm25")
        self.assertEqual(aqi, 175)
        self.assertEqual(category["name"], "Unhealthy")
        
        # Test very unhealthy
        aqi, category = calculate_aqi(200.0, "pm25")
        self.assertEqual(aqi, 250)
        self.assertEqual(category["name"], "Very Unhealthy")
        
        # Test hazardous
        aqi, category = calculate_aqi(350.0, "pm25")
        self.assertEqual(aqi, 400)
        self.assertEqual(category["name"], "Hazardous")
        
        # Test values above the highest breakpoint
        aqi, category = calculate_aqi(600.0, "pm25")
        self.assertEqual(aqi, 500)
        self.assertEqual(category["name"], "Hazardous")
        
        # Test values below the lowest breakpoint
        aqi, category = calculate_aqi(0.1, "pm25")
        self.assertEqual(aqi, 0)
        self.assertEqual(category["name"], "Good")
    
    def test_calculate_aqi_pm10(self):
        """Test PM10 AQI calculation."""
        # Test good air quality
        aqi, category = calculate_aqi(25.0, "pm10")
        self.assertEqual(aqi, 23)
        self.assertEqual(category["name"], "Good")
        
        # Test moderate air quality
        aqi, category = calculate_aqi(100.0, "pm10")
        self.assertEqual(aqi, 74)
        self.assertEqual(category["name"], "Moderate")
        
        # Test unhealthy for sensitive groups
        aqi, category = calculate_aqi(200.0, "pm10")
        self.assertEqual(aqi, 131)
        self.assertEqual(category["name"], "Unhealthy for Sensitive Groups")
    
    def test_get_category_from_aqi(self):
        """Test getting category from AQI value."""
        # Test good
        category = get_category_from_aqi(25)
        self.assertEqual(category["name"], "Good")
        
        # Test moderate
        category = get_category_from_aqi(75)
        self.assertEqual(category["name"], "Moderate")
        
        # Test unhealthy for sensitive groups
        category = get_category_from_aqi(125)
        self.assertEqual(category["name"], "Unhealthy for Sensitive Groups")
        
        # Test unhealthy
        category = get_category_from_aqi(175)
        self.assertEqual(category["name"], "Unhealthy")
        
        # Test very unhealthy
        category = get_category_from_aqi(250)
        self.assertEqual(category["name"], "Very Unhealthy")
        
        # Test hazardous
        category = get_category_from_aqi(350)
        self.assertEqual(category["name"], "Hazardous")
        
        # Test value above highest breakpoint
        category = get_category_from_aqi(600)
        self.assertEqual(category["name"], "Hazardous")
    
    def test_is_unhealthy(self):
        """Test is_unhealthy function."""
        self.assertFalse(is_unhealthy(50))
        self.assertFalse(is_unhealthy(100))
        self.assertTrue(is_unhealthy(101))
        self.assertTrue(is_unhealthy(150))
        self.assertTrue(is_unhealthy(200))
        self.assertTrue(is_unhealthy(300))
        self.assertTrue(is_unhealthy(500))


if __name__ == "__main__":
    unittest.main() 