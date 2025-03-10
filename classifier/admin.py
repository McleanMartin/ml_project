from django.contrib import admin
from .models import Diagnosis, Disease, Image

# Register your models here.

@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    list_display = ('disease', 'confidence_level', 'created_at')
    list_filter = ('disease', 'created_at')
    search_fields = ('disease__name',)
    ordering = ('-created_at',)
    
    def has_add_permission(self,request,obj=None):
        return False

@admin.register(Disease)
class DiseaseAdmin(admin.ModelAdmin):
    list_display = ('dn','name','description','symptoms','notes')
    list_display_links = ['name']
    search_fields = ('name',)
    ordering = ('name',)

    def has_add_permission(self,request,obj=None):
        return False

@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ('image', 'uploaded_at')
    list_filter = ('uploaded_at',)
    ordering = ('-uploaded_at',)

    def has_add_permission(self,request,obj=None):
        return False

admin.site.site_header = 'Maize Diseases Detection Admin'
admin.site.index_title = 'Maize Diseases Detection Management'
admin.site.site_title = 'Maize Diseases Detection Admin Panel'