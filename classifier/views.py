import os
import io
import cv2
import numpy as np
from PIL import Image as PILImage
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, get_object_or_404
from django.contrib import messages
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from ml_project.settings import BASE_DIR
from .models import Disease, Image, Diagnosis
from django.contrib.auth import login, logout,authenticate
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
import matplotlib.pyplot as plt

# Disable GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load disease model
model_path = os.path.join(BASE_DIR, 'classifier', 'maize_disease_model.h5')


def process_image(filepath, target_size=(128, 128)):
    """Efficient image processing"""
    img = load_img(filepath, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@login_required
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Basic file validation
        if not image_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            messages.error(request, "Only JPEG/PNG images allowed")
            return render(request, 'upload.html')
            
        if image_file.size > 10 * 1024 * 1024:  # 10MB max
            messages.error(request, "Image too large (max 10MB)")
            return render(request, 'upload.html')
        
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        filepath = os.path.join(fs.location, filename)

        try:
            img_array = process_image(filepath)
            prediction = disease_model.predict(img_array, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = float(np.max(prediction))
            
            # Save results
            disease = get_object_or_404(Disease, pk=class_idx+1)
            img_record = Image.objects.create(image=image_file)
            diagnosis = Diagnosis.objects.create(
                disease=disease,
                image=img_record,
                confidence_level=confidence
            )
            
            # Prepare results
            result = {
                'diagnosis': diagnosis,
                'confidence_level': confidence,
                'predicted_class': disease.name,
                'disease_description': disease.description,
                'symptoms': disease.symptoms,
                'notes': disease.notes,
                'image_url': img_record.image.url
            }
            
            return render(request, 'results.html', {'result': result})
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            messages.error(request, f"Processing error: {str(e)}")
    
    return render(request, 'upload.html')


@login_required
def diagnostic_stats(request):
    """Generate diagnostic statistics"""
    diseases = Disease.objects.all()
    disease_counts = {disease.name: Diagnosis.objects.filter(disease=disease).count() 
                     for disease in diseases}

    plt.figure(figsize=(10, 5))
    plt.bar(disease_counts.keys(), disease_counts.values(), color='skyblue')
    plt.xlabel('Diseases')
    plt.ylabel('Number of Diagnosis')
    plt.title('Diagnosis Statistics')
    plt.xticks(rotation=45)
    
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close


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
            messages.success(request, 'Account created successfully!')
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('ml_analyze')