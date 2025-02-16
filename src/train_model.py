import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from detector import DeepfakeDetector
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.detector = DeepfakeDetector()
        
        # Expected directory structure:
        # data_dir/
        #   train/
        #     real/
        #     fake/
        #   val/
        #     real/
        #     fake/
        
    def setup_data_generators(self):
        """Set up data generators for training and validation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Set up generators
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake']
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake']
        )
    
    def train(self, epochs=100, initial_epoch=0):
        """Train the model"""
        # Create callbacks
        checkpoint = ModelCheckpoint(
            'models/deepfake_model_{epoch:02d}_{val_accuracy:.4f}.weights.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        
        tensorboard = TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            write_graph=True
        )
        
        # Train the model
        history = self.detector.model.fit(
            self.train_generator,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=self.val_generator,
            callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
        )
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'])
        
        # Plot loss
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'])
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

def prepare_dataset_structure(base_dir):
    """Create the required directory structure"""
    dirs = [
        os.path.join(base_dir, 'train', 'real'),
        os.path.join(base_dir, 'train', 'fake'),
        os.path.join(base_dir, 'val', 'real'),
        os.path.join(base_dir, 'val', 'fake')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Created directory structure:")
    print("Place your training images in:")
    print(f"  - Real images: {dirs[0]}")
    print(f"  - Fake images: {dirs[1]}")
    print("Place your validation images in:")
    print(f"  - Real images: {dirs[2]}")
    print(f"  - Fake images: {dirs[3]}")

if __name__ == "__main__":
    # Set up the dataset directory structure
    data_dir = "dataset"
    prepare_dataset_structure(data_dir)
    
    # Create and train the model
    trainer = ModelTrainer(data_dir)
    trainer.setup_data_generators()
    
    # Train the model
    history = trainer.train(epochs=5)
    
    # Plot training history
    trainer.plot_training_history(history)
