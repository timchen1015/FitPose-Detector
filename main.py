import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

def init_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    return mp_drawing, mp_drawing_styles, mp_holistic

def load_model_and_labels(model_dir="trained_pose_model"):
    """
    載入訓練好的模型和標籤
    """
    try:
        # 載入模型
        model = tf.keras.models.load_model(os.path.join(model_dir, 'lstm_model'))
        # 載入標籤編碼器
        labels = np.load(os.path.join(model_dir, 'label_encoder.npy'))
        return model, labels
    except Exception as e:
        print(f"載入模型時發生錯誤: {str(e)}")
        return None, None

def extract_pose_features(results):
    """
    從 MediaPipe 結果中提取姿勢特徵
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

    # 繪製身體骨架
    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # 進行姿勢識別
    if results.pose_landmarks and model is not None:
        # 提取特徵
        features = extract_pose_features(results)
        if features is not None:
            # 重塑特徵以符合模型輸入格式
            features = features.reshape(1, 1, -1)
            
            # 進行預測
            prediction = model.predict(features, verbose=0)
            predicted_class = labels[np.argmax(prediction[0])]
            confidence = np.max(prediction[0])
            
            # 在畫面上顯示預測結果
            text = f"{predicted_class}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def main():
    # 初始化 MediaPipe
    mp_drawing, mp_drawing_styles, mp_holistic = init_mediapipe()
    
    # 載入訓練好的模型
    model, labels = load_model_and_labels()
    if model is None:
        print("無法載入模型，將只顯示骨架追蹤")
    
    # 開啟攝像頭
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟攝像頭")
        return

    # 使用 MediaPipe 進行全身偵測
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法接收影像")
                break

            # 處理影像幀
            processed_frame = process_frame(
                frame, 
                holistic, 
                mp_drawing, 
                mp_drawing_styles, 
                mp_holistic,
                model,
                labels
            )

            # 顯示結果
            cv2.imshow('Exercise Recognition', processed_frame)

            # 按下 ESC 鍵退出
            if cv2.waitKey(5) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 