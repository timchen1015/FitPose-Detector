import os
import cv2
import mediapipe as mp
import time
import sys
import logging
import math

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# Suppress MediaPipe logging
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

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

    def calculate_angle(self, a, b, c):
        """計算三個點形成的角度"""
        ab = [b[0] - a[0], b[1] - a[1]]
        bc = [c[0] - b[0], c[1] - b[1]]
        dot_product = ab[0] * bc[0] + ab[1] * bc[1]
        ab_magnitude = math.sqrt(ab[0]**2 + ab[1]**2)
        bc_magnitude = math.sqrt(bc[0]**2 + bc[1]**2)
        angle = math.acos(dot_product / (ab_magnitude * bc_magnitude))
        return math.degrees(angle)

    def is_push_up_valid(self, landmarks):
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # 計算肩膀到手肘的角度
        left_arm_angle = self.calculate_angle(left_shoulder, left_elbow, (left_elbow[0], left_elbow[1] + 1))
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, (right_elbow[0], right_elbow[1] + 1))

        # 計算手肘到髖部的角度
        left_torso_angle = self.calculate_angle(left_elbow, left_hip, (left_hip[0], left_hip[1] + 1))
        right_torso_angle = self.calculate_angle(right_elbow, right_hip, (right_hip[0], right_hip[1] + 1))

        left_wrist_vertical_angle = self.calculate_angle(left_elbow, left_wrist, (left_wrist[0], left_wrist[1] + 1))
        right_wrist_vertical_angle = self.calculate_angle(right_elbow, right_wrist, (right_wrist[0], right_wrist[1] + 1))
        if  130 < left_arm_angle < 200:
            print("left_arm_angle : " + str(left_arm_angle ))
        if 130 < left_torso_angle < 200 : 
            print("left_torso_angle :" + str(left_torso_angle))
        # 判斷條件，允許一定角度誤差，例如±10度
        return (110 < left_arm_angle < 210 and
        110 < right_arm_angle < 210 and
        110 < left_torso_angle < 210 and
        110 < right_torso_angle < 210 )



    def is_sit_up_valid(self, landmarks):
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]        
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]

        if left_knee[1] < left_ankle[1] and \
           (abs(left_wrist[0] - left_ear[0]) < 100 and abs(right_wrist[0] - right_ear[0]) < 100):
            torso_angle = self.calculate_angle(left_shoulder, left_knee, left_hip)
            return torso_angle < 90
        return False

    def is_squat_valid(self, landmarks):
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    
        # 計算膝蓋和髖關節之間的角度
        knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
    
        # 計算上身與地面的角度
        torso_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
    
        # 深蹲判斷: 膝蓋角度小於90度且上身保持接近直立
        if knee_angle < 90 and torso_angle > 70:  # 70 度可以根據需求調整
            return True
        return False
    
    def collect_poses(self, exercise_type, save_dir, num_images=100):
        save_path = os.path.join(save_dir, exercise_type)
        os.makedirs(save_path, exist_ok=True)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("無法打開攝影機")
            sys.exit(1)

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            image_count = 0
            last_save_time = time.time()
            save_interval = 2

            print(f"開始收集 {exercise_type} 姿勢...")
            print(f"目標：{num_images} 張圖片")
            print("按 'ESC' 鍵退出")

            while image_count < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("無法讀取相機畫面")
                    break
                
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                display_frame = frame.copy()
                
                if results.pose_landmarks:
                    landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                                 for lm in results.pose_landmarks.landmark]
                    current_time = time.time()

                    # 僅在符合動作標準的情況下，儲存圖片
                    if current_time - last_save_time >= save_interval and image_count < num_images:
    # 檢查動作類型並判斷條件是否符合
                        if exercise_type == 'push_up' and self.is_push_up_valid(landmarks):
                            image_name = f"{exercise_type}_{image_count}.jpg"
                            save_file = os.path.join(save_path, image_name)
                            cv2.imwrite(save_file, frame)
                            image_count += 1
                            last_save_time = current_time
                            print(f"儲存: {image_count}/{num_images}", end='\r')

                        elif exercise_type == 'sit_up' and self.is_sit_up_valid(landmarks):
                            image_name = f"{exercise_type}_{image_count}.jpg"
                            save_file = os.path.join(save_path, image_name)
                            cv2.imwrite(save_file, frame)
                            image_count += 1
                            last_save_time = current_time
                            print(f"儲存: {image_count}/{num_images}", end='\r')

                        elif exercise_type == 'squat' and self.is_squat_valid(landmarks):
                            image_name = f"{exercise_type}_{image_count}.jpg"
                            save_file = os.path.join(save_path, image_name)
                            cv2.imwrite(save_file, frame)
                            image_count += 1
                            last_save_time = current_time
                            print(f"儲存: {image_count}/{num_images}", end='\r')

                    # 顯示骨架
                    self.mp_draw.draw_landmarks(display_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                cv2.putText(display_frame, f"Collected: {image_count}/{num_images}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Pose Collection', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("\n收集已被使用者停止")
                    break

            print(f"\n完成收集 {num_images} 張圖片")

        except KeyboardInterrupt:
            print("\n收集已被使用者中斷 (Ctrl+C)")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("程式結束")

def main():
    try:
        dataset_dir = "exercise_dataset"
        
        collector = PoseDataCollector()
        
        exercises = ['push_up', 'squat', 'sit_up']
        for exercise in exercises:
            try:
                input(f"\n準備收集 {exercise} 數據。按 Enter 開始（Ctrl+C 跳過）...")
                collector.collect_poses(exercise, dataset_dir)
            except KeyboardInterrupt:
                print(f"\n跳過 {exercise}")
                continue
            
            if exercise != exercises[-1]:
                try:
                    input("準備好下一項運動，按 Enter 繼續（Ctrl+C 退出）...")
                except KeyboardInterrupt:
                    print("\n用戶終止數據收集")
                    break
    
    except KeyboardInterrupt:
        print("\n程序由用戶終止")
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        print("程序結束")

if __name__ == "__main__":
    main()