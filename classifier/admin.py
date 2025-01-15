from django.contrib import admin
from .models import Diagnosis, Disease, Image

# Register your models here.

@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    list_display = ('disease', 'confidence_level', 'created_at')
    list_filter = ('disease', 'created_at')
    search_fields = ('disease__name',)
    ordering = ('-created_at',)

@admin.register(Disease)
class DiseaseAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)
    ordering = ('name',)

@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ('image', 'uploaded_at')
    list_filter = ('uploaded_at',)
    ordering = ('-uploaded_at',)
