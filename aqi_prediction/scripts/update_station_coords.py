#!/usr/bin/env python3
"""
Script to update NOAA station coordinates for scenarios.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.air_quality import AQParam, AQScenario, get_default_scenarios
from src.models.aqi_app import AQIApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Update NOAA station coordinates')
    parser.add_argument(
        '--scenario',
        type=str,
        help='Scenario name to update (omit to update all)'
    )
    parser.add_argument(
        '--lat',
        type=float,
        help='Latitude for the specific scenario'
    )
    parser.add_argument(
        '--lng',
        type=float,
        help='Longitude for the specific scenario'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    app = AQIApp()
    
    # Load default scenarios
    scenarios = get_default_scenarios()
    
    if args.scenario:
        # Update specific scenario
        if args.scenario not in scenarios:
            logger.error(f"Scenario '{args.scenario}' not found")
            return
            
        if args.lat is None or args.lng is None:
            logger.error("Both --lat and --lng must be provided when updating a specific scenario")
            return
            
        scenarios[args.scenario].update_noaa_station_coords(args.lat, args.lng)
        logger.info(f"Updated coordinates for scenario '{args.scenario}': {args.lat}, {args.lng}")
        
        # Add scenario to app
        app.add_aq_scenario(scenarios[args.scenario])
        app.select_scenario(args.scenario)
        
        # Force update by accessing NOAA data
        logger.info(f"Updating data for scenario '{args.scenario}'")
        noaa_data = app.get_noaa_data()
        logger.info(f"NOAA data retrieved: {len(noaa_data)} records")
        
    else:
        # Update all scenarios with default coordinates
        for name, scenario in scenarios.items():
            app.add_aq_scenario(scenario)
            app.select_scenario(name)
            
            logger.info(f"Updating data for scenario '{name}'")
            noaa_data = app.get_noaa_data()
            logger.info(f"NOAA data retrieved: {len(noaa_data)} records")
    
    logger.info("Coordinate update complete")

if __name__ == "__main__":
    main() 