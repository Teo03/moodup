import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np
from datetime import datetime
import os
import argparse

# Dictionary to provide human-readable descriptions of variables
VARIABLE_DESCRIPTIONS = {
    # Surface variables
    "hcc": "High Cloud Cover (fraction)",
    "tc_pb": "Temperature of Lead (K)",
    "tc_c3h6": "Temperature of Propene (K)",
    "tc_cl_c": "Temperature of Chlorinated Carbon (K)",
    "tp": "Total Precipitation (mm)",
    "vis": "Visibility (m)",
    "tsr": "Top Net Solar Radiation (J/m²)",
    "sund": "Sunshine Duration (s)",
    
    # Pressure level variables
    "t": "Temperature (K)",
    "w": "Vertical Velocity (Pa/s)",
    "co": "Carbon Monoxide (kg/kg)",
    "aermr04": "Sea Salt Aerosol (kg/kg)",
    "aermr06": "Dust Aerosol (kg/kg)",
    "c2h6": "Ethane (kg/kg)",
    "c5h8": "Isoprene (kg/kg)",
    "ch4_c": "Methane (kg/kg)",
    "go3": "Ozone (kg/kg)",
    "no3_a": "Nitrate Aerosol (kg/kg)",
    "clono2": "Chlorine Nitrate (kg/kg)",
    "cc": "Cloud Cover (fraction)",
    
    # Model level variables
    "t_ml": "Temperature at Model Level (K)"
}

def format_date(date_obj):
    """Format a datetime object nicely"""
    return date_obj.strftime("%B %d, %Y at %H:%M UTC")

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def analyze_dataset(ds, dataset_name):
    """
    Analyze the dataset structure and provide information about its content
    """
    print("\n" + "="*60)
    print(f"DETAILED DATASET ANALYSIS FOR {dataset_name}")
    print("="*60)
    
    # 1. Basic dimensions and coordinates
    print(f"Dimensions: {dict(ds.dims)}")
    
    # 2. Variables info
    print("\nVariables and their shapes:")
    for var_name, var in ds.data_vars.items():
        print(f"  • {var_name}: {var.shape} - {var.dtype}")
        
        # Check for all-zero or all-NaN variables
        if np.all(var.values == 0):
            print(f"    WARNING: '{var_name}' contains all zeros")
        elif np.all(np.isnan(var.values)):
            print(f"    WARNING: '{var_name}' contains all NaN values")
        
        # Show min, max, mean for non-zero variables
        if not np.all(var.values == 0) and not np.all(np.isnan(var.values)):
            try:
                min_val = float(np.nanmin(var.values))
                max_val = float(np.nanmax(var.values))
                mean_val = float(np.nanmean(var.values))
                print(f"    Range: {min_val} to {max_val}, Mean: {mean_val}")
            except Exception as e:
                print(f"    Error calculating statistics: {e}")
    
    # 3. Coordinates info
    print("\nCoordinates info:")
    for coord_name, coord in ds.coords.items():
        if coord.size < 10:
            print(f"  • {coord_name}: {coord.values}")
        else:
            print(f"  • {coord_name}: Range {coord.values.min()} to {coord.values.max()}, Size: {coord.size}")
    
    # 4. Check for attributes
    print("\nGlobal attributes:")
    for attr_name, attr_value in ds.attrs.items():
        print(f"  • {attr_name}: {attr_value}")

def get_value(ds, var_name, indices):
    """
    Get a value from the dataset
    
    Parameters:
    - ds: xarray Dataset
    - var_name: variable name to extract
    - indices: dictionary of indices for each dimension
    
    Returns:
    - value from dataset
    """
    try:
        value = float(ds[var_name].isel(**indices).values)
        return value
    except Exception as e:
        print(f"Error getting value for {var_name}: {e}")
        return np.nan

def search_by_coordinates(ds_dict, lat, lon, location_name=""):
    """
    Search for all weather variables at specific latitude and longitude coordinates
    across multiple datasets.
    
    Parameters:
    - ds_dict: Dictionary of xarray Datasets containing weather data
    - lat: target latitude (float between -90 and 90)
    - lon: target longitude (float between 0 and 360 or -180 and 180)
    - location_name: optional name of the location
    
    Returns:
    - Dictionary of variable values at the specified location
    """
    # Normalize longitude to 0-360 range if it's in -180 to 180 range
    if lon < 0:
        lon = lon + 360
    
    # Ensure coordinates are within valid range
    lat = max(min(lat, 90), -90)
    lon = lon % 360  # Wrap longitude around if needed
    
    # Print information about the search
    print("\n" + "="*60)
    location_str = f" - {location_name}" if location_name else ""
    print(f"WEATHER DATA FOR SPECIFIC LOCATION{location_str}")
    print("="*60)
    print(f"Requested coordinates: Lat {lat:.2f}, Lon {lon:.2f}")
    
    # Extract values for all variables at this location
    results = {}
    
    # Process each dataset
    for ds_name, ds in ds_dict.items():
        print(f"\n{ds_name.upper()} DATA:")
        
        # Find the closest grid points
        lat_idx = abs(ds.latitude.values - lat).argmin()
        lon_idx = abs(ds.longitude.values - lon).argmin()
        
        closest_lat = float(ds.latitude.values[lat_idx])
        closest_lon = float(ds.longitude.values[lon_idx])
        
        print(f"Closest grid point: Lat {closest_lat:.2f}, Lon {closest_lon:.2f}")
        print(f"Distance from requested point: {haversine_distance(lat, lon, closest_lat, closest_lon):.2f} km")
        
        # Get forecast time if available
        if 'forecast_reference_time' in ds.coords:
            forecast_time = pd.to_datetime(ds.forecast_reference_time.values[0])
            forecast_time_str = format_date(forecast_time)
            print(f"Forecast time: {forecast_time_str}")
        
        # Prepare indices for data extraction
        indices = {
            'latitude': lat_idx,
            'longitude': lon_idx
        }
        
        # Add forecast_period and forecast_reference_time if they exist
        if 'forecast_period' in ds.dims:
            indices['forecast_period'] = 0
        if 'forecast_reference_time' in ds.dims:
            indices['forecast_reference_time'] = 0
            
        # For pressure level and model level datasets, we need to select a specific level
        if ds_name == 'pressure_level' and 'pressure_level' in ds.dims:
            # For pressure level data, choose 850 hPa (usually around 1.5 km altitude)
            level_idx = 0  # Default to first level
            # Find closest level to 850 hPa
            if 850 in ds.pressure_level.values:
                level_idx = np.where(ds.pressure_level.values == 850)[0][0]
            else:
                level_idx = abs(ds.pressure_level.values - 850).argmin()
                
            print(f"Selected pressure level: {ds.pressure_level.values[level_idx]} hPa")
            indices['pressure_level'] = level_idx
        elif ds_name == 'model_level' and 'model_level' in ds.dims:
            # For model level, use the first level
            indices['model_level'] = 0
        
        print("\nVARIABLES:")
        
        # For each variable, get the value and print with appropriate formatting
        for var_name in ds.data_vars:
            # Skip variables that are constant across all dimensions
            if len(ds[var_name].shape) <= 1:
                continue
                
            # Get description
            description = VARIABLE_DESCRIPTIONS.get(var_name, "No description available")
            
            # For 4D variables (e.g., with time and level dimensions)
            try:
                # Extract the value
                value = get_value(ds, var_name, indices)
                results[var_name] = value
                
                # Format and print based on variable type
                if var_name == "hcc":
                    print(f"  • {description}: {value*100:.1f}%")
                elif var_name.startswith("tc_") or var_name == "t" or var_name == "t_ml":
                    print(f"  • {description}: {value:.1f}K ({value-273.15:.1f}°C)")
                elif var_name == "tp":
                    print(f"  • {description}: {value:.2f}mm")
                elif var_name == "vis":
                    print(f"  • {description}: {value/1000:.1f}km")
                elif var_name == "tsr":
                    print(f"  • {description}: {value:.1f} J/m²")
                elif var_name == "sund":
                    print(f"  • {description}: {value/3600:.1f} hours")
                elif var_name in ["u", "v", "u_ml", "v_ml"]:
                    print(f"  • {description}: {value:.1f} m/s")
                elif var_name in ["q", "q_ml"]:
                    print(f"  • {description}: {value*1000:.2f} g/kg")
                elif var_name == "r":
                    print(f"  • {description}: {value:.1f}%")
                else:
                    print(f"  • {description}: {value}")
            except Exception as e:
                print(f"  • {description}: Error - {e}")
    
    # Create a visualization for this location
    plot_location_data_multi(ds_dict, lat, lon, location_name)
    
    return results

def plot_location_data_multi(ds_dict, lat, lon, location_name=""):
    """Create visualizations of data at the specified location for each dataset"""
    # Create a directory for the plots
    os.makedirs('output', exist_ok=True)
    
    # Process each dataset
    for ds_name, ds in ds_dict.items():
        # Find the closest grid points
        lat_idx = abs(ds.latitude.values - lat).argmin()
        lon_idx = abs(ds.longitude.values - lon).argmin()
        
        closest_lat = float(ds.latitude.values[lat_idx])
        closest_lon = float(ds.longitude.values[lon_idx])
        
        # Only use variables that have at least 2 dimensions (lat, lon)
        variables = [var for var in ds.data_vars if len(ds[var].shape) >= 2]
        
        if not variables:
            print(f"No plottable variables found in {ds_name} dataset")
            continue
            
        # Create a polar plot for the variables at this location
        plt.figure(figsize=(12, 10))
        
        # Set up the polar axes
        ax = plt.subplot(111, polar=True)
        
        # Calculate angles for each variable
        angles = np.linspace(0, 2*np.pi, len(variables), endpoint=False).tolist()
        
        # Prepare indices for data extraction
        indices = {
            'latitude': lat_idx,
            'longitude': lon_idx
        }
        
        # Add forecast_period and forecast_reference_time if they exist
        if 'forecast_period' in ds.dims:
            indices['forecast_period'] = 0
        if 'forecast_reference_time' in ds.dims:
            indices['forecast_reference_time'] = 0
            
        # For pressure level and model level datasets, we need to select a specific level
        if ds_name == 'pressure_level' and 'pressure_level' in ds.dims:
            # For pressure level data, choose 850 hPa
            level_idx = 0  # Default to first level
            if 850 in ds.pressure_level.values:
                level_idx = np.where(ds.pressure_level.values == 850)[0][0]
            else:
                level_idx = abs(ds.pressure_level.values - 850).argmin()
            indices['pressure_level'] = level_idx
        elif ds_name == 'model_level' and 'model_level' in ds.dims:
            indices['model_level'] = 0
        
        # Make the plot circular by appending the first value to the end
        values = []
        var_values = {}  # Store actual values for the text annotations
        
        for var in variables:
            try:
                # Get the value
                val = get_value(ds, var, indices)
                var_values[var] = val
                
                # Normalize based on expected ranges rather than global min/max
                # This gives more meaningful radar chart when data has zeros
                if var == "hcc":
                    # Cloud cover ranges from 0 to 1
                    normalized = val
                elif var.startswith("tc_") or var == "t" or var == "t_ml":
                    # Temperature (assuming range from 240K to 320K)
                    normalized = (val - 240) / 80 if val > 240 else 0
                elif var == "tp":
                    # Precipitation (assuming max of 50mm)
                    normalized = val / 50
                elif var == "vis":
                    # Visibility (normalized to 0-100km range)
                    normalized = min(val / 100000, 1)
                elif var == "tsr":
                    # Solar radiation (normalized to 0-1000 J/m²)
                    normalized = val / 1000
                elif var == "sund":
                    # Sunshine duration (normalized to 0-12 hours)
                    normalized = val / (3600 * 12)
                elif var in ["u", "v", "u_ml", "v_ml"]:
                    # Wind components (normalized to -20 to 20 m/s range)
                    normalized = (val + 20) / 40
                elif var in ["q", "q_ml"]:
                    # Specific humidity (normalized to 0-0.02 kg/kg)
                    normalized = val / 0.02
                elif var == "r":
                    # Relative humidity (already 0-100%)
                    normalized = val / 100
                elif var == "z":
                    # Geopotential (normalized to 0-30000 range)
                    normalized = val / 30000
                elif var == "w":
                    # Vertical velocity (normalized to -2 to 2 Pa/s)
                    normalized = (val + 2) / 4
                else:
                    # Generic normalization for unknown variables
                    normalized = min(val / 100, 1)
                
                values.append(normalized)
            except Exception:
                # Skip this variable if there's an error
                values.append(0)
        
        # Skip if no valid values
        if not values:
            plt.close()
            continue
            
        # Close the loop for the polar plot
        values.append(values[0])
        angles.append(angles[0])
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label="Normalized values")
        ax.fill(angles, values, alpha=0.25)
        
        # Set the labels for each variable
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([v for v in variables], fontsize=8)
        
        # Add gridlines
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.set_rlabel_position(0)
        
        # Add annotations with actual values
        for i, var in enumerate(variables):
            if i >= len(angles) - 1:  # Skip if we're out of bounds
                continue
                
            angle = angles[i]
            if var not in var_values:  # Skip if variable has no value
                continue
                
            # Position the text at 1.25 times the maximum value radius
            x = 1.25 * np.cos(angle)
            y = 1.25 * np.sin(angle)
            
            # Format the value based on the variable type
            if var == "hcc":
                value_str = f"{var_values[var]*100:.1f}%"
            elif var.startswith("tc_") or var == "t" or var == "t_ml":
                value_str = f"{var_values[var]-273.15:.1f}°C"
            elif var == "tp":
                value_str = f"{var_values[var]:.1f}mm"
            elif var == "vis":
                value_str = f"{var_values[var]/1000:.1f}km"
            elif var == "sund":
                value_str = f"{var_values[var]/3600:.1f}h"
            elif var in ["u", "v", "u_ml", "v_ml"]:
                value_str = f"{var_values[var]:.1f}m/s"
            elif var in ["q", "q_ml"]:
                value_str = f"{var_values[var]*1000:.2f}g/kg"
            elif var == "r":
                value_str = f"{var_values[var]:.1f}%"
            else:
                value_str = f"{var_values[var]:.1f}"
                
            ax.text(angle, 1.1, value_str, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8)
        
        # Add title
        location_str = f" - {location_name}" if location_name else ""
        title = f"{ds_name.upper()} Data at Lat {closest_lat:.2f}, Lon {closest_lon:.2f}{location_str}"
        plt.title(title, y=1.15)
        
        # Save the figure
        location_part = location_name.lower().replace(" ", "_") if location_name else f"lat{lat:.1f}_lon{lon:.1f}"
        location_file = f"output/{ds_name}_{location_part}.png"
        plt.tight_layout()
        plt.savefig(location_file)
        print(f"\n{ds_name.upper()} data visualization saved to '{location_file}'")
        plt.close()

def create_pressure_profile(ds_plev, lat, lon, location_name=""):
    """Create a profile of atmospheric parameters at different pressure levels"""
    # Find the closest grid points
    lat_idx = abs(ds_plev.latitude.values - lat).argmin()
    lon_idx = abs(ds_plev.longitude.values - lon).argmin()
    
    closest_lat = float(ds_plev.latitude.values[lat_idx])
    closest_lon = float(ds_plev.longitude.values[lon_idx])
    
    # Extract vertical velocity at all levels
    w_values = []
    cc_values = []  # Cloud cover
    pressure_levels = ds_plev.pressure_level.values
    
    for level_idx, level in enumerate(pressure_levels):
        indices = {
            'latitude': lat_idx,
            'longitude': lon_idx,
            'pressure_level': level_idx,
            'forecast_period': 0,
            'forecast_reference_time': 0
        }
        
        w_val = get_value(ds_plev, 'w', indices)
        cc_val = get_value(ds_plev, 'cc', indices)
        
        w_values.append(w_val)
        cc_values.append(cc_val)
    
    # Convert to numpy arrays
    w_values = np.array(w_values)
    cc_values = np.array(cc_values)
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    
    # Plot vertical velocity profile
    ax1.barh(pressure_levels, w_values, height=30, color='skyblue')
    ax1.invert_yaxis()  # Invert y-axis to have lower pressure (higher altitude) at top
    ax1.set_xlabel('Vertical Velocity (Pa/s)')
    ax1.set_ylabel('Pressure Level (hPa)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot cloud cover profile
    ax2.barh(pressure_levels, cc_values * 100, height=30, color='lightgray')
    ax2.set_xlabel('Cloud Cover (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add title to the figure
    location_str = f" - {location_name}" if location_name else ""
    fig.suptitle(f'Atmospheric Profile at Lat {closest_lat:.2f}, Lon {closest_lon:.2f}{location_str}')
    
    # Save the figure
    location_part = location_name.lower().replace(" ", "_") if location_name else f"lat{lat:.1f}_lon{lon:.1f}"
    file_name = f"output/atmospheric_profile_{location_part}.png"
    plt.tight_layout()
    plt.savefig(file_name)
    print(f"\nAtmospheric profile visualization saved to '{file_name}'")
    plt.close()

def create_temperature_profile(ds_plev, lat, lon, location_name=""):
    """Create a temperature profile visualization for pressure level data"""
    # Find the closest grid points
    lat_idx = abs(ds_plev.latitude.values - lat).argmin()
    lon_idx = abs(ds_plev.longitude.values - lon).argmin()
    
    closest_lat = float(ds_plev.latitude.values[lat_idx])
    closest_lon = float(ds_plev.longitude.values[lon_idx])
    
    # Extract temperature at all levels
    t_values = []
    pressure_levels = ds_plev.pressure_level.values
    
    for level_idx, level in enumerate(pressure_levels):
        indices = {
            'latitude': lat_idx,
            'longitude': lon_idx,
            'pressure_level': level_idx,
            'forecast_period': 0,
            'forecast_reference_time': 0
        }
        
        t_val = get_value(ds_plev, 't', indices)
        t_values.append(t_val)
    
    # Convert to numpy arrays and to Celsius
    t_values = np.array(t_values) - 273.15
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot temperature profile
    plt.plot(t_values, pressure_levels, 'r-o', linewidth=2, markersize=5)
    plt.gca().invert_yaxis()  # Invert y-axis to have lower pressure (higher altitude) at top
    
    # Labels and title
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Pressure Level (hPa)')
    location_str = f" - {location_name}" if location_name else ""
    title = f'Temperature Profile at Lat {closest_lat:.2f}, Lon {closest_lon:.2f}{location_str}'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add 0°C line
    plt.axvline(x=0, color='blue', linestyle='--', alpha=0.7, label='0°C')
    
    # Add legend
    plt.legend()
    
    # Save the figure
    location_part = location_name.lower().replace(" ", "_") if location_name else f"lat{lat:.1f}_lon{lon:.1f}"
    file_name = f"output/temperature_profile_{location_part}.png"
    plt.tight_layout()
    plt.savefig(file_name)
    print(f"\nTemperature profile visualization saved to '{file_name}'")
    plt.close()

def calculate_mood_score(results):
    """
    Calculate a mood score from 1-100 based on weather conditions
    
    Parameters:
    - results: Dictionary of weather variables and their values
    
    Returns:
    - Mood score (1-100) with 100 being the best predicted mood
    - Explanation of the factors that influenced the score
    """
    # Initialize score at neutral 50
    score = 50
    factors = []
    
    # Use a default temperature if none available
    if 't' in results and results['t'] > 200:  # Check if it's a realistic temperature
        # We'll only use the pressure level temperature at 850hPa (approx 1.5km altitude)
        # which is more stable and representative than upper atmosphere temps
        temp_c = results['t'] - 273.15
        factors.append(f"Temperature at pressure level (850 hPa): {temp_c:.1f}°C")
        
        # Temperature score: 0 for extreme cold/hot, up to +25 for optimal (22.5°C)
        temp_score = 25 - min(25, abs(temp_c - 22.5) * 1.5)
        score += temp_score
        temp_impact = "positive" if temp_score > 0 else "negative"
        factors.append(f"Temperature had a {temp_impact} impact ({temp_score:.1f} points)")
    else:
        # If no realistic temperature is available
        factors.append("No realistic temperature data available")
    
    # Cloud cover - lower is better for mood
    if 'hcc' in results:
        cloud_cover = results['hcc']
        # Cloud cover score: -20 for overcast to +15 for clear skies
        cloud_score = 15 - (cloud_cover * 35)
        score += cloud_score
        cloud_impact = "positive" if cloud_score > 0 else "negative"
        factors.append(f"Cloud cover ({cloud_cover*100:.1f}%) had a {cloud_impact} impact ({cloud_score:.1f} points)")
    
    # Visibility - higher is better
    if 'vis' in results:
        vis_km = results['vis'] / 1000  # Convert to km
        # Visibility score: up to +10 for excellent visibility
        vis_score = min(10, vis_km / 2)
        score += vis_score
        factors.append(f"Visibility of {vis_km:.1f}km added {vis_score:.1f} points")
    
    # Precipitation - less is better
    if 'tp' in results and results['tp'] > 0:
        precip = results['tp']
        # Precipitation score: -15 for heavy rain to 0 for no rain
        precip_score = -min(15, precip * 10)
        score += precip_score
        factors.append(f"Precipitation of {precip:.1f}mm reduced score by {abs(precip_score):.1f} points")
    
    # Vertical air motion - stable conditions are better
    if 'w' in results:
        vertical_velocity = results['w']
        # Score for vertical velocity: +5 for calm to -10 for strong updrafts/downdrafts
        w_score = 5 - min(15, abs(vertical_velocity) * 5)
        score += w_score
        w_impact = "positive" if w_score > 0 else "negative"
        factors.append(f"Vertical air motion had a {w_impact} impact ({w_score:.1f} points)")
    
    # Sunshine duration - more is better
    if 'sund' in results and results['sund'] > 0:
        sunshine_hours = results['sund'] / 3600  # Convert to hours
        # Sunshine score: up to +15 for lots of sunshine
        sun_score = min(15, sunshine_hours * 1.5)
        score += sun_score
        factors.append(f"Sunshine duration of {sunshine_hours:.1f} hours added {sun_score:.1f} points")
    
    # Ensure the score is between 1 and 100
    score = max(1, min(100, score))
    
    return int(score), factors

def create_mood_visualization(score, factors, location_name, output_dir):
    """Create a visualization of the mood score"""
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Create a gauge-like visualization
    ax = plt.subplot(111)
    
    # Define colors for different score ranges
    colors = ['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9']
    color_idx = min(4, score // 20)
    color = colors[color_idx]
    
    # Draw the gauge
    plt.pie([score, 100-score], colors=[color, '#F5F5F5'], startangle=90, counterclock=False, 
            wedgeprops={'width': 0.3, 'edgecolor': 'w'})
    
    # Add the score in the center
    plt.text(0, 0, f"{score}", fontsize=36, ha='center', va='center')
    plt.text(0, -0.2, f"Mood Score", fontsize=14, ha='center', va='center')
    
    # Add the interpretation
    if score >= 80:
        interpretation = "Excellent mood conditions"
    elif score >= 60:
        interpretation = "Good mood conditions"
    elif score >= 40:
        interpretation = "Moderate mood conditions"
    elif score >= 20:
        interpretation = "Poor mood conditions"
    else:
        interpretation = "Very poor mood conditions"
    
    plt.text(0, -0.4, interpretation, fontsize=12, ha='center', va='center')
    
    # Make the plot circular
    plt.axis('equal')
    
    # Hide axis
    plt.axis('off')
    
    # Add title
    location_str = location_name if location_name else f"Lat {lat:.2f}, Lon {lon:.2f}"
    plt.title(f"Weather-Based Mood Score for {location_str}", y=1.1)
    
    # Save the figure
    file_name = location_name.lower().replace(" ", "_") if location_name else f"lat{lat:.1f}_lon{lon:.1f}"
    mood_file = f"{output_dir}/mood_score_{file_name}.png"
    plt.tight_layout()
    plt.savefig(mood_file)
    print(f"\nMood visualization saved to '{mood_file}'")
    plt.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze weather data from NetCDF files")
    
    # Add arguments for coordinates
    parser.add_argument("--lat", type=float, help="Latitude (between -90 and 90)", required=True)
    parser.add_argument("--lon", type=float, help="Longitude (between -180 and 180)", required=True)
    parser.add_argument("--name", type=str, help="Location name")
    
    # Add argument to specify output directory
    parser.add_argument("--output", type=str, default="output", help="Output directory (default: 'output')")
    
    return parser.parse_args()

def search_location(datasets, lat, lon, location_name, output_dir="output"):
    """Search for weather data at a location"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Search for weather data
    results = search_by_coordinates(datasets, lat, lon, location_name)
    
    # Calculate mood score
    mood_score, mood_factors = calculate_mood_score(results)
    
    # Print mood score
    print("\n" + "="*60)
    print(f"MOOD SCORE FOR {location_name if location_name else f'Lat {lat:.2f}, Lon {lon:.2f}'}")
    print("="*60)
    print(f"Overall Mood Score: {mood_score}/100")
    print("\nFactors affecting the score:")
    for factor in mood_factors:
        print(f"  • {factor}")
    
    # Create additional visualizations
    create_pressure_profile(datasets['pressure_level'], lat, lon, location_name)
    create_temperature_profile(datasets['pressure_level'], lat, lon, location_name)
    
    # Create a mood visualization
    create_mood_visualization(mood_score, mood_factors, location_name, output_dir)
    
    print(f"\nData for {location_name if location_name else f'Lat {lat:.2f}, Lon {lon:.2f}'} saved to {output_dir}/")
    
    return mood_score, mood_factors

def load_datasets():
    """Load all NetCDF datasets"""
    print("Loading surface dataset...")
    ds_sfc = xr.open_dataset('data/data_sfc.nc', decode_timedelta=True)
    
    print("Loading pressure level dataset...")
    ds_plev = xr.open_dataset('data/data_plev.nc', decode_timedelta=True)
    
    print("Loading model level dataset...")
    ds_mlev = xr.open_dataset('data/data_mlev.nc', decode_timedelta=True)
    
    # Create a dictionary of datasets
    datasets = {
        'surface': ds_sfc,
        'pressure_level': ds_plev,
        'model_level': ds_mlev
    }
    
    # Analyze each dataset
    for ds_name, ds in datasets.items():
        analyze_dataset(ds, ds_name)
    
    return datasets

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load datasets
    datasets = load_datasets()
    
    # Process the location
    location_name = args.name if args.name else f"Lat{args.lat:.1f}_Lon{args.lon:.1f}"
    search_location(datasets, args.lat, args.lon, location_name, args.output)
    
    # Close all datasets
    for ds in datasets.values():
        ds.close()
    
    print("\nAnalysis complete. Mood scores are based on weather conditions and how they typically affect human mood.")
    print("Higher scores (closer to 100) indicate weather conditions associated with better moods.")