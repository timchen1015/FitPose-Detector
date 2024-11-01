import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# Suppress MediaPipe logging
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import cv2
import mediapipe as mp
import os
import time
import sys
import logging

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

class PoseDataCollector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def collect_poses(self, exercise_type, save_dir, num_images=100):
        """
        Collect exercise pose images
        """
        # Create save directory
        save_path = os.path.join(save_dir, exercise_type)
        os.makedirs(save_path, exist_ok=True)
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            sys.exit(1)  # Exit with error code
            
        try:
            # Set camera resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            image_count = 0
            last_save_time = time.time()
            save_interval = 0.5
            
            print(f"Starting to collect {exercise_type} poses...")
            print(f"Target: {num_images} images")
            print("Press 'ESC' to quit, or Ctrl+C to force exit")
            
            while True:  # Changed to infinite loop with manual break
                ret, frame = cap.read()
                if not ret:
                    print("Cannot read camera frame")
                    break
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                    
                # Convert to RGB and detect pose
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                # Copy frame for display
                display_frame = frame.copy()
                
                # If pose detected
                if results.pose_landmarks:
                    # Draw skeleton
                    self.mp_draw.draw_landmarks(
                        display_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    
                    # Save image at intervals
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval and image_count < num_images:
                        image_name = f"{exercise_type}_{image_count}.jpg"
                        save_file = os.path.join(save_path, image_name)
                        cv2.imwrite(save_file, frame)
                        image_count += 1
                        last_save_time = current_time
                        print(f"Saved: {image_count}/{num_images}", end='\r')
                
                # Show progress
                cv2.putText(display_frame, f"Collected: {image_count}/{num_images}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Pose Collection', display_frame)
                
                # Check for exit keys
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\nCollection stopped by user")
                    break
                
                # Check if we've collected enough images
                if image_count >= num_images:
                    print(f"\nCompleted collecting {num_images} images")
                    break
                
        except KeyboardInterrupt:
            print("\nCollection stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            print(f"{exercise_type} data collection ended")

def main():
    try:
        # Set save directory
        dataset_dir = "exercise_dataset"
        
        # Create collector
        collector = PoseDataCollector()
        
        # Collect data for each exercise
        exercises = ['push_up', 'squat', 'sit_up']
        for exercise in exercises:
            try:
                # TODO: The input is not working
                # input(f"\nReady to collect {exercise} data. Press Enter to start (Ctrl+C to skip)...")
                collector.collect_poses(exercise, dataset_dir)
            except KeyboardInterrupt:
                print(f"\nSkipping {exercise}")
                continue
            
            if exercise != exercises[-1]:
                try:
                    input("Prepare for next exercise, press Enter to continue (Ctrl+C to exit)...")
                except KeyboardInterrupt:
                    print("\nData collection ended by user")
                    break
    
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        print("Program ended")

if __name__ == "__main__":
    main()