from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .models import Mood
from .serializers import MoodSerializer
from .utils import calculate_mood_stats

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
    """Get statistics and recommendations about user's mood based on weather"""
    # Extract latitude, longitude and location name from query parameters if provided
    lat = request.query_params.get('lat')
    lon = request.query_params.get('lon')
    location_name = request.query_params.get('location_name')
    
    # Calculate statistics using provided coordinates if available
    stats = calculate_mood_stats(lat=lat, lon=lon, location_name=location_name)
    return Response(stats)
