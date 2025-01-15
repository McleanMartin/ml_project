from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError

class Disease(models.Model):
    dn = models.IntegerField()
    name = models.CharField(max_length=225, verbose_name="Disease Name")
    description = models.TextField(verbose_name="Disease Description")
    symptoms = models.TextField()

    def __str__(self):
        return self.name


class Image(models.Model):
    image = models.ImageField()
    uploaded_at = models.DateTimeField(auto_now=True, verbose_name="Upload Time")

    def __str__(self):
        return f"Image uploaded at {self.uploaded_at}"

class Diagnosis(models.Model):
    disease = models.ForeignKey("Disease", on_delete=models.CASCADE, verbose_name="Diagnosed Disease")
    image = models.ForeignKey("Image", on_delete=models.CASCADE)
    confidence_level = models.DecimalField(max_digits=3, decimal_places=2, verbose_name="Confidence Level")

    created_at = models.DateTimeField(auto_now=True, verbose_name="Diagnosis Time")

    def clean(self):
        if not (0 <= self.confidence_level <= 1):
            raise ValidationError('Confidence level must be between 0 and 1.')

    def __str__(self):
        return f"Diagnosis of {self.disease} with confidence {self.confidence_level:.2f}"