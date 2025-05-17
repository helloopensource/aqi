#!/usr/bin/env python
"""
Script to calculate AQI from concentration values.
Usage: python calculate_aqi.py --parameter pm25 --concentration 35.5
"""
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import date

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import application modules
from src.utils.aqi_calculator import calculate_aqi, get_category_from_aqi
from src.config.settings import AQI_BREAKPOINTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate AQI from concentration values")
    parser.add_argument(
        "--parameter", 
        type=str, 
        choices=list(AQI_BREAKPOINTS.keys()),
        default="pm25",
        help="Air quality parameter"
    )
    parser.add_argument(
        "--concentration", 
        type=float, 
        required=True,
        help="Concentration value"
    )
    parser.add_argument(
        "--json", 
        action="store_true",
        help="Output in JSON format"
    )
    return parser.parse_args()


def main():
    """Main function to calculate AQI."""
    args = parse_args()
    
    try:
        # Calculate AQI
        aqi_value, category = calculate_aqi(args.concentration, args.parameter)
        
        # Process results
        if args.json:
            result = {
                "parameter": args.parameter,
                "concentration": args.concentration,
                "aqi": aqi_value,
                "category": category["name"],
                "color": category["color"],
                "range": category["range"],
                "health_implications": category["health_implications"],
                "cautionary_statement": category["cautionary_statement"],
                "date": date.today().isoformat()
            }
            print(json.dumps(result, indent=2))
        else:
            print(f"\nAQI Calculation Results:")
            print(f"-------------------------")
            print(f"Parameter: {args.parameter}")
            print(f"Concentration: {args.concentration}")
            print(f"AQI Value: {aqi_value}")
            print(f"Category: {category['name']} ({category['range']})")
            print(f"Color Code: {category['color']}")
            print(f"Health Implications: {category['health_implications']}")
            print(f"Cautionary Statement: {category['cautionary_statement']}")
            print(f"-------------------------\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error calculating AQI: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 