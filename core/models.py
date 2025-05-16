from django.db import models

# Create your models here.

class Mood(models.Model):
    name = models.CharField(max_length=100)
    emoji = models.CharField(max_length=10)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.emoji} {self.name}"
