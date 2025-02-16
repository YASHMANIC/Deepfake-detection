import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import os

class DeepfakeDetector:
    def __init__(self, model_path=None):
        """
        Initialize the DeepfakeDetector.
        Args:
            model_path: Optional path to a saved model. If not provided, creates a new model.
        """
        self.image_size = (224, 224)
        if model_path and os.path.exists(model_path):
            try:
                # Create a fresh model
                self.model = self._build_model()
                
                # Load weights directly instead of the full model
                print(f"Loading weights from {model_path}")
                self.model.load_weights(model_path)
                print("Weights loaded successfully")
                
            except Exception as e:
                print(f"Error loading model weights: {str(e)}")
                print("Creating new model instead.")
                self.model = self._build_model()
        else:
            self.model = self._build_model()
            print("Created new model")

    def _build_model(self):
        """Build and compile the model"""
        # Load EfficientNet-B0 as base model
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        
        # Unfreeze some layers for fine-tuning
        for layer in base_model.layers[-30:]:
            layer.trainable = True
            
        # Build the model
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with a lower learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model

    def save_model(self, filepath):
        """Save the model to a file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def preprocess_image(self, image_path):
        """Preprocess an image for prediction"""
        try:
            # Load and preprocess the image
            image = tf.keras.preprocessing.image.load_img(
                image_path, 
                target_size=self.image_size
            )
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
            return np.expand_dims(image_array, axis=0)
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def detect(self, image_path):
        """
        Detect if an image is likely a deepfake.
        Returns a probability between 0 and 1, where higher values indicate higher likelihood of being a deepfake.
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Preprocess the image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return 0.5
            
            # Get prediction
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Return probability (between 0 and 1)
            return float(prediction[0][0])
            
        except Exception as e:
            print(f"Error in deepfake detection: {str(e)}")
            # Return a moderate probability in case of error
            return 0.5

    def analyze(self, media_path):
        try:
            # Process the image
            input_tensor = self.preprocess_image(media_path)
            
            # Model inference
            prediction = self.model.predict(input_tensor)[0][0]
            
            # Convert NumPy types to Python native types
            prediction_float = float(prediction)
            
            return {
                'probability': prediction_float,
                'is_deepfake': bool(prediction_float > 0.5),
                'confidence': float(abs(0.5 - prediction_float) * 2)
            }
        except Exception as e:
            return {
                'error': str(e),
                'probability': None,
                'is_deepfake': None
            }

    def evaluate(self, test_generator):
        """
        Evaluate the model on a test dataset
        Args:
            test_generator: A data generator providing test data
        Returns:
            Dictionary containing evaluation metrics
        """
        results = self.model.evaluate(test_generator, verbose=1)
        return {
            'loss': results[0],
            'accuracy': results[1]
        }
