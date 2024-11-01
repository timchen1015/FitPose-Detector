import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# Suppress MediaPipe logging
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import logging

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

def init_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    return mp_drawing, mp_drawing_styles, mp_holistic

def load_model_and_labels(model_dir="trained_pose_model"):
    """
    Load trained model and labels
    """
    try:
        # Load model
        model_path = os.path.join(model_dir, 'pose_model.h5')
        model = tf.keras.models.load_model(model_path)
        
        # Load label encoder
        labels = np.load(os.path.join(model_dir, 'label_encoder.npy'))
        return model, labels
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

def extract_pose_features(results):
    """
    Extract pose features from MediaPipe results
    """
    if results.pose_landmarks:
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    return None

def process_frame(frame, holistic, mp_drawing, mp_drawing_styles, mp_holistic, model, labels):
    if frame is None:
        return None
    
    frame = cv2.resize(frame, (520, 300))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    # Draw skeleton
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # Perform pose recognition
    if results.pose_landmarks and model is not None:
        # Extract features
        features = extract_pose_features(results)
        if features is not None:
            # Reshape features to match model input format
            features = features.reshape(1, 1, -1)
            
            # Make prediction
            prediction = model.predict(features, verbose=0)
            predicted_class = labels[np.argmax(prediction[0])]
            confidence = np.max(prediction[0])
            
            # Display prediction results
            text = f"{predicted_class}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def main():
    # Initialize MediaPipe
    mp_drawing, mp_drawing_styles, mp_holistic = init_mediapipe()
    
    # Load trained model
    model, labels = load_model_and_labels()
    if model is None:
        print("Could not load model, will only show skeleton tracking")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Use MediaPipe for pose detection
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                break

            # Process frame
            processed_frame = process_frame(
                frame, 
                holistic, 
                mp_drawing, 
                mp_drawing_styles, 
                mp_holistic,
                model,
                labels
            )

            # Show results
            cv2.imshow('Exercise Recognition', processed_frame)

            # Press ESC to exit
            if cv2.waitKey(5) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()