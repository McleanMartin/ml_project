from django.contrib import admin
from .models import Diagnosis,Disease,Image

# Register your models here.

@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    pass

@admin.register(Disease)
class DiseaseAdmin(admin.ModelAdmin):
    pass

@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    pass


