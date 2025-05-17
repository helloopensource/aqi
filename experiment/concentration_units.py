import pandas as pd
import numpy as np

def ugm3_to_ppm(ugm3, molecular_weight, temperature=25, pressure=1013.25):
    """
    Convert µg/m³ to ppm (parts per million)
    
    Parameters:
    ugm3: concentration in µg/m³
    molecular_weight: molecular weight of the gas in g/mol
    temperature: temperature in Celsius (default 25°C)
    pressure: pressure in hPa (default 1013.25 hPa)
    
    Returns:
    concentration in ppm
    """
    # Convert temperature to Kelvin
    temp_k = temperature + 273.15
    
    # Universal gas constant (L·hPa/(mol·K))
    R = 83.144
    
    # Convert µg/m³ to ppm
    ppm = (ugm3 * R * temp_k) / (molecular_weight * pressure)
    
    return ppm

def ppm_to_ugm3(ppm, molecular_weight, temperature=25, pressure=1013.25):
    """
    Convert ppm to µg/m³
    
    Parameters:
    ppm: concentration in ppm
    molecular_weight: molecular weight of the gas in g/mol
    temperature: temperature in Celsius (default 25°C)
    pressure: pressure in hPa (default 1013.25 hPa)
    
    Returns:
    concentration in µg/m³
    """
    # Convert temperature to Kelvin
    temp_k = temperature + 273.15
    
    # Universal gas constant (L·hPa/(mol·K))
    R = 83.144
    
    # Convert ppm to µg/m³
    ugm3 = (ppm * molecular_weight * pressure) / (R * temp_k)
    
    return ugm3

# Example usage with common air pollutants
pollutants = {
    'NO2': 46.0055,    # Nitrogen dioxide
    'SO2': 64.066,     # Sulfur dioxide
    'O3': 47.9982,     # Ozone
    'CO': 28.0101,     # Carbon monoxide
    'PM2.5': 28.97     # Particulate matter (using average molecular weight of air)
}

# Create example DataFrame with concentrations
data = {
    'pollutant': list(pollutants.keys()),
    'ugm3': [40, 50, 60, 1000, 25],  # Example concentrations in µg/m³
}

df = pd.DataFrame(data)

# Add molecular weights
df['molecular_weight'] = df['pollutant'].map(pollutants)

# Convert µg/m³ to ppm
df['ppm'] = df.apply(lambda row: ugm3_to_ppm(row['ugm3'], row['molecular_weight']), axis=1)

# Convert back to µg/m³ to verify
df['ugm3_verify'] = df.apply(lambda row: ppm_to_ugm3(row['ppm'], row['molecular_weight']), axis=1)

print("\nAir Pollutant Concentration Conversions:")
print(df.round(4))

# Example of temperature and pressure effects
print("\nEffect of Temperature and Pressure on NO2 conversion (40 µg/m³):")
temperatures = [0, 25, 40]
pressures = [1013.25, 900, 1100]

for temp in temperatures:
    for press in pressures:
        ppm = ugm3_to_ppm(40, pollutants['NO2'], temp, press)
        print(f"Temperature: {temp}°C, Pressure: {press}hPa -> {ppm:.4f} ppm") 