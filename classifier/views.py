import os
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render, get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.contrib import messages  
from .models import Disease, Image, Diagnosis
from keras.models import load_model  # type: ignore
import numpy as np
from PIL import Image as PILImage
from ml_project.settings import BASE_DIR

# Load the model
model = load_model(os.path.join(BASE_DIR, 'maize_disease_model.h5'))

def process_image(filepath):
    """Process the uploaded image for prediction."""
    img = PILImage.open(filepath)
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 128, 128, 3)

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
            diagnosis = Diagnosis.objects.create(
                disease=disease,
                image=Image.objects.create(image=image_file),
                confidence_level=confidence_level
            )

            previous_scan = Diagnosis.objects.all().order_by('-created_at')

            result = {
                'previous_scan': previous_scan,
                'diagnosis': diagnosis,
                'confidence_level': confidence_level,
                'predicted_class': diagnosis.disease.name,
                'disease_description': diagnosis.disease.description,
                'symptoms': diagnosis.disease.symptoms,
            }
            return render(request, 'results.html', {'result': result})

        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
        
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    return render(request, 'upload.html', {'result': result})


def diagnostic_stats(request):
    # Get all diseases and their corresponding diagnosis counts
    diseases = Disease.objects.all()
    disease_counts = {}

    for disease in diseases:
        count = Diagnosis.objects.filter(disease=disease).count()
        disease_counts[disease.name] = count

    # Create a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(disease_counts.keys(), disease_counts.values(), color='skyblue')
    plt.xlabel('Diseases')
    plt.ylabel('Number of Diagnoses')
    plt.title('Number of Diagnoses per Disease')
    plt.xticks(rotation=45)
    
    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode the image to display in HTML
    graph = base64.b64encode(image_png).decode('utf-8')
    
    return render(request, 'diagnostic_stats.html', {'graph': graph})