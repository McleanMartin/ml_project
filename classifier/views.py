import os
import io
import base64
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from django.contrib import messages
from .models import Disease, Image, Diagnosis
from tensorflow.keras.models import load_model
from PIL import Image as PILImage
from ml_project.settings import BASE_DIR
import matplotlib.pyplot as plt
from .forms import LoginForm, RegisterForm
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.models import User
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load disease classification model
model_path = os.path.join(BASE_DIR, 'classifier', 'maize_disease_model.h5')
disease_model = load_model(model_path)
disease_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class MaizeLeafValidator:
    def __init__(self):
        self.reference_dir = os.path.join(BASE_DIR, 'classifier', 'reference_maize')
        self.min_similarity = 0.6  # Similarity threshold
        
        if not os.path.exists(self.reference_dir):
            os.makedirs(self.reference_dir)
            raise FileNotFoundError(f"Please add reference images to {self.reference_dir}")
    
    def _calculate_similarity(self, img1, img2):
        """Calculate similarity between two images using histogram comparison"""
        # Convert to HSV color space
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
        
        # Calculate histograms
        hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 256])
        
        # Normalize and compare
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def is_maize_leaf(self, img_path):
        """Validate if image contains maize leaf features"""
        try:
            # Load uploaded image
            uploaded_img = np.array(PILImage.open(img_path).convert('RGB'))
            
            # Compare with reference images
            max_similarity = 0
            for ref_img in os.listdir(self.reference_dir):
                if ref_img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ref_path = os.path.join(self.reference_dir, ref_img)
                    ref_img = np.array(PILImage.open(ref_path).convert('RGB'))
                    
                    # Resize to common dimensions
                    uploaded_resized = cv2.resize(uploaded_img, (256, 256))
                    ref_resized = cv2.resize(ref_img, (256, 256))
                    
                    similarity = self._calculate_similarity(uploaded_resized, ref_resized)
                    max_similarity = max(max_similarity, similarity)
            
            return max_similarity >= self.min_similarity, max_similarity
        
        except Exception as e:
            return False, 0.0

# Initialize validator
try:
    import cv2
    maize_validator = MaizeLeafValidator()
except ImportError:
    cv2 = None
    maize_validator = None
    print("Warning: OpenCV not available. Maize validation disabled.")

def process_image(filepath, target_size=(128, 128)):
    """Memory-efficient image processing"""
    img = load_img(filepath, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@login_required
def upload_image(request):
    result = None
    
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        allowed_types = ['image/jpeg', 'image/png']
        if image_file.content_type not in allowed_types:
            messages.error(request, "Only JPEG/PNG images allowed")
            return render(request, 'upload.html', {'result': result})
        
        if image_file.size > 5 * 1024 * 1024:  # 5MB
            messages.error(request, "Image too large (max 5MB)")
            return render(request, 'upload.html', {'result': result})
        
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        filepath = os.path.join(fs.location, filename)

        try:
            if maize_validator:
                is_maize, confidence = maize_validator.is_maize_leaf(filepath)
                if not is_maize:
                    os.remove(filepath)
                    messages.error(request, f"Not a maize leaf (similarity: {confidence:.2%})")
                    return render(request, 'upload.html', {'result': result})
            
            img_array = process_image(filepath)
            prediction = disease_model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction))
    
            disease = get_object_or_404(Disease, pk=predicted_class+1)
            diagnosis_image = Image.objects.create(image=image_file)
            diagnosis = Diagnosis.objects.create(
                disease=disease,
                image=diagnosis_image,
                confidence_level=confidence
            )
            
            previous_scan = Diagnosis.objects.all()
            paginator = Paginator(previous_scan, 10)
            page = request.GET.get('page', 1)
            
            try:
                previous_scan = paginator.page(page)
            except (PageNotAnInteger, EmptyPage):
                previous_scan = paginator.page(1)
            
            result = {
                'previous_scan': previous_scan,
                'diagnosis': diagnosis,
                'confidence_level': confidence,
                'predicted_class': disease.name,
                'disease_description': disease.description,
                'symptoms': disease.symptoms,
                'notes': disease.notes,
                'image_url': diagnosis_image.image.url
            }
            
            return render(request, 'results.html', {'result': result})
            
        except Exception as e:
            messages.error(request, f"Processing error: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
        
    return render(request, 'upload.html', {'result': result})

@login_required
def diagnostic_stats(request):
    """Generate and render diagnostic statistics."""
    diseases = Disease.objects.all()
    disease_counts = {disease.name: Diagnosis.objects.filter(disease=disease).count() for disease in diseases}

    plt.figure(figsize=(10, 5))
    plt.bar(disease_counts.keys(), disease_counts.values(), color='skyblue')
    plt.xlabel('Diseases')
    plt.ylabel('Number of Diagnosis')
    plt.title('Number of Diagnosis per Disease')
    plt.xticks(rotation=45)
    
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
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