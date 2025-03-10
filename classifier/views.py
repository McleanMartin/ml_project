import os
import io
import base64
from django.shortcuts import render, get_object_or_404,redirect
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.contrib import messages  
from .models import Disease, Image, Diagnosis
from keras.models import load_model  
import numpy as np
from PIL import Image as PILImage
from ml_project.settings import BASE_DIR
import matplotlib.pyplot as plt
from .forms import LoginForm,RegisterForm
from django.contrib.auth import login,logout,aauthenticate
from django.contrib.auth.models import User
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required

# Load the trained dataset
model = load_model(os.path.join(BASE_DIR, 'maize_disease_model.h5'))

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
            # Process the image
            img_array = process_image(filepath)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence_level = float(np.max(prediction))

            # Get the disease
            disease = get_object_or_404(Disease, pk=(predicted_class + 1))

            # Save to Diagnosis
            diagnosis_image = Image.objects.create(image=image_file)
            diagnosis = Diagnosis.objects.create(
                disease=disease,
                image=diagnosis_image,
                confidence_level=confidence_level
            )

            # Fetch previous scans with pagination
            previous_scan = Diagnosis.objects.all().order_by('-created_at')
            paginator = Paginator(previous_scan, 10) 
            page = request.GET.get('page') 

            try:
                previous_scan = paginator.page(page)
            except PageNotAnInteger:
                previous_scan = paginator.page(1)
            except EmptyPage:
                previous_scan = paginator.page(paginator.num_pages)

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
        
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    return render(request, 'upload.html', {'result': result})

@login_required
def diagnostic_stats(request):
    """Generate and render diagnostic statistics."""
    diseases = Disease.objects.all()
    disease_counts = {disease.name: Diagnosis.objects.filter(disease=disease).count() for disease in diseases}

    # Create a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(disease_counts.keys(), disease_counts.values(), color='skyblue')
    plt.xlabel('Diseases')
    plt.ylabel('Number of Diagnosis')
    plt.title('Number of Diagnosis per Disease')
    plt.xticks(rotation=45)
    
    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the image to display in HTML
    graph = base64.b64encode(image_png).decode('utf-8')
    
    return render(request, 'diagnostic_stats.html', {'graph': graph})


class Login_View(LoginView):
    template_name = 'login.html'


def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = User.objects.create_user(
                username=request.POST['username'],
                email=request.POST['email'],
                password=request.POST['password']
            )
            
            messages.success(request, 'Client account created successfully!')
            return redirect('login') 
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('ml_analyze')