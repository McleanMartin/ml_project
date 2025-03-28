import os
import cv2
import numpy as np
import io
import base64
import tempfile
from PIL import Image as PILImage
from django.shortcuts import render, get_object_or_404, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.contrib.auth import login, logout, authenticate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from ml_project.settings import BASE_DIR
from .models import Disease, Image, Diagnosis
from .forms import LoginForm, RegisterForm
from django.contrib.auth.models import User
import matplotlib.pyplot as plt

# Disable GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load disease model
model_path = os.path.join(BASE_DIR, 'classifier', 'maize_disease_model.h5')
disease_model = load_model(model_path)
disease_model.compile(optimizer='adam', 
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

class MaizeImageValidator:
    def __init__(self):
        self.reference_dir = os.path.join(BASE_DIR, 'classifier', 'reference_maize')
        self.min_green_percentage = 0.80 
        self.min_similarity = 0.90  
        self.reference_features = self._load_reference_features()
        
        if not self.reference_features:
            raise ValueError(f"No valid reference images in {self.reference_dir}")

    def _load_reference_features(self):
        """Load and process reference maize leaf images with robust loading"""
        features = []
        if not os.path.exists(self.reference_dir):
            os.makedirs(self.reference_dir)
            return features
            
        for img_file in sorted(os.listdir(self.reference_dir)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(self.reference_dir, img_file)
                    # Use PIL for more robust loading
                    pil_img = PILImage.open(img_path).convert('RGB')
                    img = np.array(pil_img)
                    
                    # Handle different bit depths
                    if img.dtype == 'uint16':
                        img = (img / 256).astype('uint8')
                    elif img.dtype in ['float32', 'float64']:
                        img = (img * 255).astype('uint8')
                    
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, (300, 300))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    features.append({
                        'hist': self._get_color_histogram(img),
                        'shape': self._get_shape_features(img),
                        'texture': self._get_texture_features(img)
                    })
                except Exception as e:
                    print(f"Error processing reference {img_file}: {str(e)}")
        return features

    def _get_color_histogram(self, img):
        """Get HSV color histogram"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _get_shape_features(self, img):
        """Get shape features using Hu Moments"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            moments = cv2.moments(contours[0])
            return cv2.HuMoments(moments).flatten()
        return None

    def _get_texture_features(self, img):
        """Get texture features using LBP"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp = self._local_binary_pattern(gray)
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def _local_binary_pattern(self, img, radius=3, neighbors=24):
        """Calculate Local Binary Pattern"""
        lbp = np.zeros_like(img)
        for i in range(radius, img.shape[0]-radius):
            for j in range(radius, img.shape[1]-radius):
                center = img[i,j]
                values = []
                for k in range(neighbors):
                    angle = 2*np.pi*k/neighbors
                    x = int(i + radius*np.cos(angle))
                    y = int(j - radius*np.sin(angle))
                    values.append(1 if img[x,y] > center else 0)
                lbp[i,j] = sum([v*(2**n) for n,v in enumerate(values)])
        return lbp

    def _is_green_plant(self, img):
        """Check if image contains sufficient green plant material"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = cv2.countNonZero(mask)
        return (green_pixels / (img.shape[0] * img.shape[1])) >= self.min_green_percentage

    def _has_leaf_shape(self, shape_features):
        """Check if shape matches maize leaf characteristics"""
        if shape_features is None:
            return False
        # Typical maize leaf Hu Moments should be within certain ranges
        return (0.1 < shape_features[0] < 0.5 and 
                0.01 < shape_features[1] < 0.2)

    def validate_maize_image(self, img_path):
        """Comprehensive maize leaf validation with robust image loading"""
        try:
            # Load with PIL first (more robust for different formats)
            pil_img = PILImage.open(img_path)
            img = np.array(pil_img.convert('RGB'))
            
            # Handle different data types
            if img.dtype == 'uint16':
                img = (img / 256).astype('uint8')
            elif img.dtype in ['float32', 'float64']:
                img = (img * 255).astype('uint8')
                
            # Continue with OpenCV processing
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (300, 300))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Initial checks
            if not self._is_green_plant(img):
                return False, 0.0, "Not enough green plant material"
                
            shape_features = self._get_shape_features(img)
            if not self._has_leaf_shape(shape_features):
                return False, 0.0, "Shape doesn't match maize leaves"
            
            # Compare with reference images
            test_hist = self._get_color_histogram(img)
            test_texture = self._get_texture_features(img)
            max_similarity = 0
            
            for ref in self.reference_features:
                # Color similarity
                hist_sim = cv2.compareHist(test_hist, ref['hist'], cv2.HISTCMP_CORREL)
                
                # Texture similarity
                texture_sim = cv2.compareHist(test_texture, ref['texture'], cv2.HISTCMP_CORREL)
                
                # Shape similarity
                if shape_features is not None and ref['shape'] is not None:
                    shape_sim = 1.0 / (1.0 + np.linalg.norm(shape_features - ref['shape']))
                else:
                    shape_sim = 0
                
                # Combined score
                combined = 0.5*hist_sim + 0.3*texture_sim + 0.2*shape_sim
                
                if combined > max_similarity:
                    max_similarity = combined
            
            is_valid = max_similarity >= self.min_similarity
            return is_valid, max_similarity, None
            
        except Exception as e:
            return False, 0.0, f"Validation error: {str(e)}"

# Initialize validator
try:
    maize_validator = MaizeImageValidator()
except Exception as e:
    print(f"Validator initialization failed: {str(e)}")
    maize_validator = None

def process_image(filepath, target_size=(128, 128)):
    """Robust image processing that handles different bit depths"""
    try:
        # First try with PIL (handles most formats)
        img = PILImage.open(filepath)
        img = img.convert('RGB')  # Ensure RGB format
        img = img.resize(target_size)
        
        # Convert to numpy array and check range
        img_array = np.array(img)
        
        # Handle different data types
        if img_array.dtype == 'uint16':
            img_array = (img_array / 256).astype('uint8')
        elif img_array.dtype == 'float32' or img_array.dtype == 'float64':
            if img_array.max() <= 1.0:  # If normalized 0-1
                img_array = (img_array * 255).astype('uint8')
            else:  # If values exceed 1.0
                img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Final normalization for model
        img_array = img_array.astype('float32') / 255.0
        return np.expand_dims(img_array, axis=0)
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        raise

@login_required
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            
            # Validate file type
            valid_extensions = ['.png', '.jpg', '.jpeg']
            if not any(image_file.name.lower().endswith(ext) for ext in valid_extensions):
                messages.error(request, "Only JPEG/PNG images allowed")
                return render(request, 'upload.html')
                
            if image_file.size > 10 * 1024 * 1024:  # 10MB max
                messages.error(request, "Image too large (max 10MB)")
                return render(request, 'upload.html')
            
            # Save to temp file first for validation
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                for chunk in image_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            try:
                # Strict maize validation
                if maize_validator:
                    is_maize, confidence, err = maize_validator.validate_maize_image(tmp_path)
                    if not is_maize:
                        msg = err if err else (
                            f"Image rejected (confidence: {confidence:.2%}). "
                            "Please upload a clear photo of a maize leaf."
                        )
                        messages.error(request, msg)
                        return render(request, 'upload.html')
                
                # Only process if validation passed
                img_array = process_image(tmp_path)
                prediction = disease_model.predict(img_array, verbose=0)
                class_idx = np.argmax(prediction)
                confidence = float(np.max(prediction))
                
                # Save to permanent storage
                fs = FileSystemStorage()
                filename = fs.save(image_file.name, image_file)
                
                # Save results
                disease = get_object_or_404(Disease, pk=class_idx+1)
                img_record = Image.objects.create(image=filename)
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
                
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            messages.error(request, f"Processing error: {str(e)}")
            return render(request, 'upload.html')
    
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