from datetime import datetime, timedelta
import random
from .models import Mood
import os
import xarray as xr
import numpy as np
import sys
import os.path
import openai
import json

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path to the directory containing test.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test import calculate_mood_score, load_datasets

# Global variable to store loaded datasets
_datasets = None

def _get_datasets():
    """
    Load datasets if not already loaded
    """
    global _datasets
    if _datasets is None:
        try:
            _datasets = load_datasets()
        except Exception as e:
            print(f"Error loading datasets: {e}")
            _datasets = {}
    return _datasets

def generate_mood_description(mood_score, factors, location_name):
    """
    Generate an AI description of the user's current mood based on weather factors
    
    Parameters:
    - mood_score: numerical mood score (0-100)
    - factors: list of factors affecting the mood
    - location_name: name of the user's location
    
    Returns:
    - AI-generated description of the user's mood
    """
    # Fallback description in case API call fails
    fallback_description = {
        "mood_description": f"Based on the weather conditions in {location_name}, your mood score is {mood_score}/100, suggesting a {'positive' if mood_score > 50 else 'moderate' if mood_score > 30 else 'challenging'} outlook for today.",
        "emotional_state": "reflective",
        "recommended_activities": "Taking some time outdoors might be beneficial today."
    }
    
    # Fallback recommendations in case API call fails
    fallback_recommendations = [
        {
            "title": "Morning meditation",
            "description": "Start your day with a 10-minute meditation to improve focus"
        },
        {
            "title": "Afternoon walk",
            "description": "A brisk walk in the afternoon can boost your mood"
        },
        {
            "title": "Social connection",
            "description": "Schedule a quick call with someone you care about"
        }
    ]
    
    try:
        if not openai.api_key:
            print("OpenAI API key not found")
            return {**fallback_description, "recommendations": fallback_recommendations}
            
        # Format factors into a readable string
        factors_text = "\n".join([f"- {factor}" for factor in factors]) if factors else "No specific factors available."
        
        # Create a prompt for the AI
        prompt = f"""
        You are an expert in weather psychology and mood analysis. Based on the following weather data for {location_name}, please provide a brief, personalized mood description for the user.
        
        Mood Score: {mood_score}/100
        Weather Factors:
        {factors_text}
        
        Respond with a JSON object that includes:
        - mood_description: A 2-3 sentence personalized description of how the weather might be affecting the user's mood
        - emotional_state: A single word describing the likely emotional state (e.g., "energetic", "calm", "reflective")
        - weekly_insight: A brief sentence about how the weather trend might affect their mood over the coming week
        - recommendations: An array of 3 personalized activity recommendations based on the weather and mood data. Each recommendation should have a "title" (short name) and "description" (1 sentence explanation)
        
        Keep the tone supportive and insightful, not clinical.
        """
        
        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI psychology assistant specializing in weather's impact on mood."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        
        # Parse the response
        content = response.choices[0].message.content.strip()
        try:
            # Try to extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                mood_data = json.loads(json_str)
                # If recommendations not present, use fallback
                if "recommendations" not in mood_data:
                    mood_data["recommendations"] = fallback_recommendations
                return mood_data
            else:
                print("Could not find JSON in OpenAI response")
                return {**fallback_description, "recommendations": fallback_recommendations}
        except Exception as e:
            print(f"Error parsing OpenAI response: {e}")
            return {**fallback_description, "recommendations": fallback_recommendations}
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {**fallback_description, "recommendations": fallback_recommendations}

def calculate_mood_stats(lat=None, lon=None, location_name=None):
    """
    Calculate statistics from mood entries using real weather data
    
    Parameters:
    - lat: latitude provided by frontend
    - lon: longitude provided by frontend
    - location_name: name of the location (optional)
    """
    # Get datasets
    datasets = _get_datasets()
    if not datasets or lat is None or lon is None:
        # Return empty data if datasets can't be loaded or coordinates aren't provided
        return {
            'average_mood': 0,
            'entries_count': 0,
            'highest_mood': {
                'value': 0,
                'date': datetime.now().strftime('%Y-%m-%d')
            },
            'lowest_mood': {
                'value': 0,
                'date': datetime.now().strftime('%Y-%m-%d')
            },
            'most_frequent_mood': 0
        }
    
    # Use the provided coordinates
    location = {"lat": float(lat), "lon": float(lon), "name": location_name or "Current Location"}
    
    # Extract data at the location
    results = {}
    for ds_name, ds in datasets.items():
        # Find closest grid points
        lat_idx = abs(ds.latitude.values - location["lat"]).argmin()
        lon_idx = abs(ds.longitude.values - location["lon"]).argmin()
        
        # Prepare indices for data extraction
        indices = {'latitude': lat_idx, 'longitude': lon_idx}
        
        # Add forecast_period and forecast_reference_time if they exist
        if 'forecast_period' in ds.dims:
            indices['forecast_period'] = 0
        if 'forecast_reference_time' in ds.dims:
            indices['forecast_reference_time'] = 0
        
        # For pressure level data, use 850 hPa level
        if ds_name == 'pressure_level' and 'pressure_level' in ds.dims:
            level_idx = 0
            if 850 in ds.pressure_level.values:
                level_idx = np.where(ds.pressure_level.values == 850)[0][0]
            else:
                level_idx = abs(ds.pressure_level.values - 850).argmin()
            indices['pressure_level'] = level_idx
        elif ds_name == 'model_level' and 'model_level' in ds.dims:
            indices['model_level'] = 0
        
        # Extract values for each variable
        for var_name in ds.data_vars:
            if len(ds[var_name].shape) <= 1:
                continue
            try:
                value = float(ds[var_name].isel(**indices).values)
                results[var_name] = value
            except Exception:
                pass
    
    # Calculate mood score
    score, factors = calculate_mood_score(results)
    
    # Scale the score from 0-100 to 0-10 range for better frontend integration
    scaled_score = score / 10
    
    # Generate AI mood description with recommendations
    mood_analysis = generate_mood_description(score, factors, location["name"])
    
    # Create consolidated response with all required data
    return {
        'average_mood': scaled_score,
        'entries_count': 1,
        'highest_mood': {
            'value': scaled_score,
            'location': location["name"],
            'date': datetime.now().strftime('%Y-%m-%d'),
            'factors': factors
        },
        'lowest_mood': {
            'value': scaled_score,
            'location': location["name"],
            'date': datetime.now().strftime('%Y-%m-%d'),
            'factors': factors
        },
        'most_frequent_mood': scaled_score,
        'ai_mood_analysis': mood_analysis
    } 