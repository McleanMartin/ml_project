import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from django.core.management.base import BaseCommand
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from classifier.models import Disease

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train the disease classification model'

    def add_arguments(self, parser):
        parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--target_size', type=int, nargs=2, default=(128, 128), help='Target size for images')

    def handle(self, *args, **kwargs):
        epochs = kwargs['epochs']
        batch_size = kwargs['batch_size']
        target_size = tuple(kwargs['target_size'])

        # Define the base directory as the dataset folder inside the classifier app
        base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
        train_dir = os.path.join(base_dir, 'train')
        test_dir = os.path.join(base_dir, 'test')

        # Image Data Generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            rotation_range=20,
            validation_split=0.2
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        try:
            # Load the training data
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )

            validation_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )

            # Load the test data
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='categorical'
            )

            # Build the CNN model
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dense(len(train_generator.class_indices), activation='softmax')
            ])

            # Compile the model
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

            # Train the model
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs
            )

            # Save the trained model
            model_save_path = os.path.join(base_dir, 'maize_disease_model.h5')
            model.save(model_save_path)
            logger.info('Model saved at %s', model_save_path)

            # Save disease names to the database
            for disease_name in train_generator.class_indices.keys():
                Disease.objects.get_or_create(name=disease_name, description='')
                logger.info('Disease %s added to the database.', disease_name)

            # Plot training & validation accuracy and loss
            plt.figure(figsize=(12, 4))

            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()

            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()

            plot_path = os.path.join(base_dir, 'training_history.png')
            plt.savefig(plot_path)
            logger.info('Training history saved as %s', plot_path)

            self.stdout.write(self.style.SUCCESS('Model trained and saved successfully.'))

        except Exception as e:
            logger.error('Error occurred during training: %s', e)
            self.stdout.write(self.style.ERROR('An error occurred during training.'))