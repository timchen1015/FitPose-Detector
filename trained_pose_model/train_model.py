import os
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras    
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout, BatchNormalization, Attention, Concatenate, Input, GlobalAveragePooling1D
from keras.api.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from albumentations import Compose, GaussianBlur, HorizontalFlip, HueSaturationValue, RandomBrightnessContrast, Rotate, ShiftScaleRotate
class PoseModelTrainer:
    def __init__(self, dataset_path, export_dir="trained_pose_model"):
        """
        Initialize pose recognition model trainer
        
        Args:
            dataset_path: Path to dataset root directory
            export_dir: Directory for model export
        """
        self.dataset_path = os.path.abspath(dataset_path)
        self.export_dir = os.path.abspath(export_dir)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.label_encoder = LabelEncoder()
        
        os.makedirs(self.export_dir, exist_ok=True)
    
    def extract_pose_features(self, image_path):
        """
        Extract pose features from image
        
        Args:
            image_path: Path to image file
        Returns:
            Pose features array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
            
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            
            # Process pose detection
            results = pose.process(image)
            
            if results.pose_landmarks:
                # Extract landmark coordinates
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                image = augment_image(image)
                return np.array(landmarks)
            
        return None
    
    def prepare_dataset(self):
        """
        Prepare training dataset by processing each video as a sequence of frames.
        
        Returns:
            Features array and labels array
        """
        X = []  # Features
        y = []  # Labels
        
        # Iterate through all exercise types (e.g., sit_up, push_up)
        for action in os.listdir(self.dataset_path):
            action_path = os.path.join(self.dataset_path, action)
            if not os.path.isdir(action_path):
                continue
                    
            print(f"Processing {action} class data...")
            
            # Iterate through all video folders inside each exercise type
            for video_folder in os.listdir(action_path):
                video_path = os.path.join(action_path, video_folder)
                if not os.path.isdir(video_path):
                    continue
                
                print(f"  Processing video: {video_folder}")
                
                # Store frames from the current video
                frames = []
                for image_file in sorted(os.listdir(video_path)):  # Sort to ensure frames are processed in order
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(video_path, image_file)
                        try:
                            # Extract pose features
                            features = self.extract_pose_features(image_path)
                            if features is not None:
                                frames.append(features)
                        except Exception as e:
                            print(f"    Error processing image {image_file}: {str(e)}")
                

                # pad and truncate
                MAX_SEQUENCE_LENGTH  = 50
                if len(frames) > 10:                            # Only process videos with sufficient frames
                    frames = frames[:MAX_SEQUENCE_LENGTH]       # Truncate longer sequences
                    while len(frames) < MAX_SEQUENCE_LENGTH:
                        frames.append(np.zeros_like(frames[0])) # Padding with zeros if shorter
                    
                    X.append(np.array(frames))                  # Store the entire video as a sequence of frames
                    y.append(action)                            # Label exercise type
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid pose data found in the dataset.")
        
        return np.array(X), np.array(y)

    # def create_lstm_model(self, input_shape, num_classes):
    #     """
    #     Create LSTM model
        
    #     Args:
    #         input_shape: Shape of input features
    #         num_classes: Number of classification classes
    #     """
    #     model = Sequential([
    #         LSTM(128, input_shape=input_shape, return_sequences=True),
    #         BatchNormalization(),
    #         Dropout(0.3),

    #         LSTM(64, input_shape=input_shape, return_sequences=True),
    #         BatchNormalization(),
    #         Dropout(0.3),

    #         Dense(64, activation='relu'),
    #         BatchNormalization(),
    #         Dropout(0.3),
    #         Dense(num_classes, activation='softmax')
    #     ])
        
    #     model.compile(
    #         optimizer='adam',
    #         loss='sparse_categorical_crossentropy',
    #         metrics=['accuracy']
    #     )
        
    #     return model

    def create_lstm_model(self, input_shape, num_classes):
        """
        Create LSTM model with attention mechanism.
        
        Args:
            input_shape: Shape of input features (timesteps, features)
            num_classes: Number of classification classes
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        x = LSTM(128, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Second LSTM layer
        x = LSTM(64, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Attention mechanism
        attention = Attention()([x, x])  # Self-attention layer
        x = Concatenate()([x, attention])  # Combine LSTM output with attention
        
        # Fully connected layers
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Apply global average pooling to reduce time steps dimension
        x = GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create and compile model
        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_model(self, epochs=60, batch_size=8):
        """
        Train pose recognition model
        
        Args:
            epochs: Number of training epochs
            batch_size: Size of training batches
        """
        try:
            print("Preparing dataset...")
            X, y = self.prepare_dataset()
            
            if len(X) == 0:
                raise ValueError("No valid training data found")
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Reshape data for LSTM input (samples, timesteps, features)
            n_features = 33 * 3  # MediaPipe provides 33 landmarks, each with x, y, z coordinates
            X_reshaped = X.reshape(len(X), X.shape[1], n_features)
            
            # Split training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_reshaped, y_encoded, test_size=0.2, random_state=42
            )
            
            print(f"Training set size: {len(X_train)} samples")
            print(f"Test set size: {len(X_test)} samples")
            print(f"Feature dimensions: {X_train.shape}")
            print(f"Action classes: {self.label_encoder.classes_}")
            
            # Create model
            model = self.create_lstm_model(
                input_shape=(X_reshaped.shape[1], n_features),  # n_timesteps is X_reshaped.shape[1]
                num_classes=len(self.label_encoder.classes_)
            )
            
            # Add ModelCheckpoint to save the best model
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.export_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )

            # Add early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train model
            print("Starting model training...")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[checkpoint, early_stopping],
                verbose=1
            )
            
            # Evaluate model
            print("\nEvaluating model performance...")
            test_loss, test_acc = model.evaluate(X_test, y_test)
            print(f"Test accuracy: {test_acc:.4f}")
            
            # Save model
            print("\nSaving model...")
            model_path = os.path.join(self.export_dir, 'lstm_model.keras')
            model.save(model_path)
            
            # Save label encoder
            np.save(
                os.path.join(self.export_dir, 'label_encoder.npy'),
                self.label_encoder.classes_
            )
            
            # Plot training history
            self.plot_training_history(history)
            
        except Exception as e:
            raise Exception(f"Error during model training: {str(e)}")

    def plot_training_history(self, history):
        """
        Plot training history graphs
        """
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'training_history.png'))
        plt.show()


def main():
    try:
        DATASET_PATH = "exercise_dataset/image_dataset"
        trainer = PoseModelTrainer(DATASET_PATH)
        trainer.train_model(epochs=60, batch_size=8)
        print("Training completed!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def augment_image(image):
    """
    Apply image augmentations and return the augmented image.
    """
    augmentations = Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=15, p=0.5),
        RandomBrightnessContrast(p=0.3),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        GaussianBlur(blur_limit=(3, 7), p=0.3),
    ])

    if image is None:
        raise ValueError(f"Cannot read image")
    augmented = augmentations(image=image)
    return augmented['image']

if __name__ == "__main__":
    main() 