#!/bin/bash

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

# Start Gunicorn server
echo "Starting Gunicorn server..."
exec gunicorn --bind 0.0.0.0:8000 moodup.wsgi 