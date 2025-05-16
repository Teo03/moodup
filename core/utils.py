from datetime import datetime, timedelta
import random
from .models import Mood

def calculate_mood_stats():
    """
    Calculate statistics from mood entries
    Returns placeholder data for now
    """
    # Placeholder data - in production this would query the database
    return {
        'average_mood': 7.5,
        'entries_count': 28,
        'highest_mood': {
            'value': 9,
            'date': '2023-07-15'
        },
        'lowest_mood': {
            'value': 4,
            'date': '2023-07-10'
        },
        'most_frequent_mood': 7
    }

def calculate_weekly_trend():
    """
    Calculate weekly trend data for mood entries
    Returns placeholder data for now
    """
    # Generate dates for the last 7 days
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
    
    # Placeholder data - in production this would query the database
    return {
        'dates': dates,
        'values': [6, 7, 5, 8, 7, 9, 8],
        'trend': 'increasing'
    }

def get_mood_recommendations():
    """
    Get personalized recommendations based on mood data
    Returns placeholder recommendations for now
    """
    # Placeholder data - in production this would use mood data to generate personalized recs
    recommendations = [
        {
            'type': 'activity',
            'title': 'Morning meditation',
            'description': 'Start your day with a 10-minute meditation to improve focus',
            'benefit': 'Reduces stress and anxiety'
        },
        {
            'type': 'exercise',
            'title': '30-minute walk',
            'description': 'A brisk walk in the afternoon can boost your mood',
            'benefit': 'Releases endorphins and improves sleep'
        },
        {
            'type': 'social',
            'title': 'Connect with a friend',
            'description': 'Schedule a quick call with someone you care about',
            'benefit': 'Social connections improve emotional wellbeing'
        }
    ]
    
    return {
        'recommendations': recommendations,
        'mood_insight': 'Your mood has been improving over the past week. Keep up the good work!'
    } 