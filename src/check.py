import os
import tensorflow as tf
from detector import DeepfakeDetector

def main():
    # Enable memory growth to avoid GPU memory issues
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")

    # Find the latest model weights file
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found.")
        return

    weight_files = [f for f in os.listdir(models_dir) if f.endswith('.weights.h5')]
    if not weight_files:
        print(f"Error: No model weight files found in '{models_dir}'.")
        print("Please train the model first using train_model.py")
        return

    # Sort by creation time to get the latest weights
    latest_weights = max(weight_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    weights_path = os.path.join(models_dir, latest_weights)
    
    try:
        # Initialize detector with the weights
        print(f"Loading model weights from: {weights_path}")
        detector = DeepfakeDetector(weights_path)
        
        # Specify the image path
        image_path = input("Enter the path to the image to analyze (or press Enter to use a sample image): ").strip()
        if not image_path:
            image_path = 'dataset/val/real/sample.jpg'  # Use a sample from validation set
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return
            
        # Preprocess and predict
        print(f"Processing image: {image_path}")
        image = detector.preprocess_image(image_path)
        if image is None:
            print("Error: Failed to preprocess image.")
            return
            
        # Make prediction with error handling
        try:
            prediction = detector.model.predict(image, verbose=0)
            print(f"Probability of {image_path} being a deepfake: {prediction[0][0]:.3f}")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()