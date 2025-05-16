from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Mood
from .serializers import MoodSerializer
from .utils import calculate_mood_stats, calculate_weekly_trend, get_mood_recommendations

# Create your views here.

class MoodViewSet(viewsets.ModelViewSet):
    queryset = Mood.objects.all()
    serializer_class = MoodSerializer

@api_view(['GET'])
def api_root(request):
    return Response({
        'message': 'Welcome to MoodUp API!',
        'status': 'API is functioning correctly',
    })

@api_view(['GET'])
def mood_statistics(request):
    """Get statistics about user's mood entries"""
    stats = calculate_mood_stats()
    return Response(stats)

@api_view(['GET'])
def weekly_mood_trend(request):
    """Get weekly mood trend data"""
    trend_data = calculate_weekly_trend()
    return Response(trend_data)

@api_view(['GET'])
def mood_recommendations(request):
    """Get personalized recommendations based on mood data"""
    recommendations = get_mood_recommendations()
    return Response(recommendations)
