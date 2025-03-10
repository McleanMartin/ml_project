"""ml_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from classifier import views as v
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', v.upload_image, name="ml_analyze"),
    path('accounts/login/', v.Login_View.as_view(), name="login"),
    path('accounts/register/', v.register_view, name="register"),
    path('accounts/logout/', v.logout_view, name="logout"),
    path('stats/', v.diagnostic_stats, name="stats"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
