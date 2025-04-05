import os
import numpy as np
from PIL import Image as PILImage
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.views.generic import View
from django import forms
from .models import Disease, Image, Diagnosis
from keras.models import load_model
from ml_project.settings import BASE_DIR

# Load the ML model
model = load_model(os.path.join(BASE_DIR, 'maize_disease_model.h5'))


class Login_View(View):
    def get(self, request):
        return render(request, 'login.html')

    def post(self, request):
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {username}!")
            return redirect('ml_analyze')
        
        messages.error(request, "Invalid username or password.")
        return render(request, 'login.html', {
            'form_data': {'username': username or ''}
        })

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
        elif User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
        else:
            user = User.objects.create_user(username=username, email=email, password=password)
            login(request, user)
            messages.success(request, "Registration successful!")
            return redirect('ml_analyze')
        
        return render(request, 'register.html', {
            'form_data': {
                'username': username or '',
                'email': email or ''
            }
        })
    
    return render(request, 'register.html')

@login_required
def logout_view(request):
    logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect('login')

# ML Processing Functions
def process_image(filepath):
    """Process the uploaded image for prediction."""
    img = PILImage.open(filepath).resize((128, 128))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 128, 128, 3)

@login_required
def upload_image(request):
    result = None  
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        filepath = os.path.join(fs.location, filename)

        try:
            # Process and predict
            img_array = process_image(filepath)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence_level = float(np.max(prediction))

            # Get disease info
            disease = get_object_or_404(Disease, pk=(predicted_class + 1))

            # Save diagnosis
            diagnosis_image = Image.objects.create(image=image_file)
            diagnosis = Diagnosis.objects.create(
                disease=disease,
                image=diagnosis_image,
                confidence_level=confidence_level
            )

            
            previous_scan = Diagnosis.objects.all().order_by('-created_at')
            paginator = Paginator(previous_scan, 6) 
            page = request.GET.get('page') 
            previous_scan = paginator.get_page(page)

            result = {
                'previous_scan': previous_scan,
                'diagnosis': diagnosis,
                'confidence_level': confidence_level,
                'predicted_class': diagnosis.disease.name,
                'disease_description': diagnosis.disease.description,
                'symptoms': diagnosis.disease.symptoms,
                'notes': diagnosis.disease.notes,
            }
            return render(request, 'results.html', {'result': result})

        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            return render(request, 'upload.html', {'result': result})
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return render(request, 'upload.html', {'result': result})

@login_required
def diagnostic_stats(request):
    """Generate diagnostic statistics for the current user."""
    diseases = Disease.objects.all()
    if not diseases:
        return render(request, 'diagnostic_stats.html', {'error': 'No diseases found.'})

    disease_counts = {
        disease.name: Diagnosis.objects.filter(
            disease=disease, 
        ).count() for disease in diseases
    }
    
    return render(request, 'diagnostic_stats.html', {
        'labels': list(disease_counts.keys()),
        'data': list(disease_counts.values())
    })
