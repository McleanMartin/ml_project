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
disease_model = load_model(model_path)
disease_model.compile(optimizer='adam', 
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

class StrictMaizeValidator:
    def __init__(self):
        self.reference_dir = os.path.join(BASE_DIR, 'classifier', 'maize_references')
        self.min_green_percentage = 0.70  # 70% of image must be green
        self.min_similarity = 0.85        # 85% similarity to references
        self.reference_features = self._load_reference_features()
        
        if not self.reference_features:
            raise ValueError(f"No valid reference images in {self.reference_dir}")

    def _safe_load_image(self, img_path):
        """Safely load and normalize image to 8-bit format"""
        try:
            # Use PIL to load image (handles more formats than OpenCV)
            pil_img = PILImage.open(img_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Convert to numpy array and ensure proper range
            img = np.array(pil_img)
            
            # Normalize to 0-255 if needed
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            return img
        except Exception as e:
            print(f"Error loading image {os.path.basename(img_path)}: {str(e)}")
            return None

    def _load_reference_features(self):
        """Load and process reference images with robust error handling"""
        features = []
        if not os.path.exists(self.reference_dir):
            os.makedirs(self.reference_dir)
            return features
            
        for img_file in sorted(os.listdir(self.reference_dir)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.reference_dir, img_file)
                img = self._safe_load_image(img_path)
                
                if img is not None:
                    try:
                        # Convert to BGR for OpenCV
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        img = cv2.resize(img, (300, 300))
                        
                        features.append({
                            'hist': self._get_color_histogram(img),
                            'shape': self._get_shape_features(img),
                            'texture': self._get_texture_features(img),
                            'file': img_file  # Store filename for debugging
                        })
                    except Exception as e:
                        print(f"Error processing reference {img_file}: {str(e)}")
        return features

    def _get_color_histogram(self, img):
        """Get HSV color histogram with normalization"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _get_shape_features(self, img):
        """Get shape features using Hu Moments with error handling"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            moments = cv2.moments(contours[0])
            hu_moments = cv2.HuMoments(moments)
            return hu_moments.flatten()
        return None

    def _get_texture_features(self, img):
        """Get texture features using LBP with normalization"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Ensure proper 8-bit format
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Calculate LBP
        radius = 3
        neighbors = 24
        lbp = np.zeros_like(gray)
        for i in range(radius, gray.shape[0]-radius):
            for j in range(radius, gray.shape[1]-radius):
                center = gray[i,j]
                values = []
                for k in range(neighbors):
                    angle = 2*np.pi*k/neighbors
                    x = int(i + radius*np.cos(angle))
                    y = int(j - radius*np.sin(angle))
                    values.append(1 if gray[x,y] > center else 0)
                lbp[i,j] = sum([v*(2**n) for n,v in enumerate(values)])
        
        # Normalize histogram
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def _is_green_plant(self, img):
        """Check if image contains sufficient green plant material"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40]) 
        upper_green = np.array([85, 255, 255]) 
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = cv2.countNonZero(mask)
        return (green_pixels / (img.shape[0] * img.shape[1])) >= self.min_green_percentage

    def _has_leaf_shape(self, shape_features):
        """Check if shape matches maize leaf characteristics"""
        if shape_features is None:
            return False
        # Typical maize leaf Hu Moments ranges
        return (0.1 < shape_features[0] < 0.5 and 
                0.01 < shape_features[1] < 0.2)

    def validate_maize_image(self, img_path):
        """Comprehensive maize leaf validation with robust error handling"""
        try:
            # Load and preprocess image
            img = self._safe_load_image(img_path)
            if img is None:
                return False, 0.0, "Invalid image file"
                
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (300, 300))
            
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
                try:
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
                except Exception as e:
                    print(f"Error comparing with reference {ref.get('file', 'unknown')}: {str(e)}")
                    continue
            
            is_valid = max_similarity >= self.min_similarity
            return is_valid, max_similarity, None
            
        except Exception as e:
            return False, 0.0, f"Validation error: {str(e)}"

# Initialize validator
try:
    maize_validator = StrictMaizeValidator()
except Exception as e:
    print(f"Validator initialization failed: {str(e)}")
    maize_validator = None

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
            # Strict maize validation
            if maize_validator:
                is_maize, confidence, err = maize_validator.validate_maize_image(filepath)
                if not is_maize:
                    os.remove(filepath)
                    msg = err if err else (
                        f"Image rejected (confidence: {confidence:.2%}). "
                        "This doesn't appear to be a maize leaf."
                    )
                    messages.error(request, msg)
                    return render(request, 'upload.html')
            
            # Only process if validation passed
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