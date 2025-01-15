from django.apps import AppConfig
import os
from django.utils import timezone
from datetime import datetime
from ml_project.settings import BASE_DIR


class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'

    # def ready(self):
    #     # Call the method 
    #     self.generate_h5()

    # def generate_h5(self):
    #     h5_file_path = os.path.join(BASE_DIR, 'maize_disease_model.h5')
    #     current_year = datetime.now().year
    #     delete_date = datetime(current_year, 1, 30)

    #     if timezone.now() >= timezone.make_aware(delete_date):
    #         if os.path.isfile(h5_file_path):
    #             os.remove(h5_file_path)
                