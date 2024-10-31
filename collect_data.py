import cv2
import mediapipe as mp
import os
import time

class PoseDataCollector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        
    def collect_poses(self, exercise_type, save_dir, num_images=100):
        """
        收集運動姿勢圖片
        
        Args:
            exercise_type: 運動類型 (push_up, squat, sit_up)
            save_dir: 保存目錄
            num_images: 要收集的圖片數量
        """
        # 創建保存目錄
        save_path = os.path.join(save_dir, exercise_type)
        os.makedirs(save_path, exist_ok=True)
        
        # 開啟攝像頭
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("無法開啟攝像頭")
            return
            
        image_count = 0
        last_save_time = time.time()
        save_interval = 0.5  # 每0.5秒保存一張圖片
        
        print(f"開始收集 {exercise_type} 的姿勢數據...")
        print(f"目標收集 {num_images} 張圖片")
        print("按 'q' 提前結束收集")
        
        while image_count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 轉換為 RGB 並進行姿勢檢測
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            # 如果檢測到姿勢
            if results.pose_landmarks:
                # 繪製骨架
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # 每隔一段時間保存一張圖片
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    image_name = f"{exercise_type}_{image_count}.jpg"
                    save_file = os.path.join(save_path, image_name)
                    cv2.imwrite(save_file, frame)
                    image_count += 1
                    last_save_time = current_time
                    print(f"已保存 {image_count}/{num_images} 張圖片", end='\r')
            
            # 顯示當前幀和進度
            cv2.putText(frame, f"Collected: {image_count}/{num_images}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Pose Collection', frame)
            
            # 按 q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n{exercise_type} 數據收集完成！")

def main():
    # 設置保存目錄
    dataset_dir = "exercise_dataset"
    
    # 創建數據收集器
    collector = PoseDataCollector()
    
    # 收集每種運動的數據
    exercises = ['push_up', 'squat', 'sit_up']
    for exercise in exercises:
        input(f"\n準備收集 {exercise} 的數據。請按 Enter 開始...")
        collector.collect_poses(exercise, dataset_dir)
        
        # 等待用戶準備下一個動作
        if exercise != exercises[-1]:
            input("請準備下一個動作，按 Enter 繼續...")

if __name__ == "__main__":
    main() 