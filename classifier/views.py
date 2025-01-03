import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.contrib import messages  
from .models import Disease, Image, Diagnosis
from keras.models import load_model  # type: ignore
import numpy as np
from PIL import Image as PILImage
from ml_project.settings import BASE_DIR

# Load the model
model = load_model(os.path.join(BASE_DIR, 'maize_disease_model.h5'))

def upload_image(request):
    result = None  # Initialize result variable
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        filepath = os.path.join(fs.location, filename)

        try:
            # Process the image
            img = PILImage.open(filepath)
            img = img.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 128, 128, 3)

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence_level = np.max(prediction)

            # Save to Diagnosis
            # diagnosis = Diagnosis.objects.create(
            #     disease=Disease.objects.get(pk=predicted_class),
            #     image=Image(image=image_file),
            #     confidence_level=confidence_level
            # )

            result = {
                # 'diagnosis': diagnosis,
                'confidence_level': confidence_level,
                # 'predicted_class': diagnosis.disease.name
            }

            return render(request,'results.html',{'confidence_level':confidence_level})

        except Exception as e:
            messages.warning(request, f'An error occurred: {str(e)}')
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    return render(request, 'upload.html', {'result': result})