FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set static root directory for Django
ENV STATIC_ROOT=/app/staticfiles

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create static files directory
RUN mkdir -p /app/staticfiles

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Collect static files
RUN python manage.py collectstatic --noinput

# Run migrations and create superuser (if needed)
# These are commented out and should be run after container starts
# CMD ["python", "manage.py", "migrate"]

# Expose the port the app runs on
EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"] 