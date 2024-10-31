import os
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class PoseModelTrainer:
    def __init__(self, dataset_path, export_dir="trained_pose_model"):
        """
        初始化姿勢識別模型訓練器
        """
        self.dataset_path = os.path.abspath(dataset_path)
        self.export_dir = os.path.abspath(export_dir)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.label_encoder = LabelEncoder()
        
        os.makedirs(self.export_dir, exist_ok=True)

    def extract_pose_features(self, image_path):
        """
        從圖片中提取姿勢特徵
        
        Args:
            image_path: 圖片文件路徑
        Returns:
            姿勢特徵
        """
        # 讀取圖片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"無法讀取圖片: {image_path}")
            
        # 轉換為 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            
            # 進行姿勢檢測
            results = pose.process(image)
            
            if results.pose_landmarks:
                # 提取關鍵點座標
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                return np.array(landmarks)
            
        return None

    def prepare_dataset(self):
        """
        準備訓練數據集
        """
        X = []  # 特徵
        y = []  # 標籤
        
        # 遍歷所有運動類別
        for action in os.listdir(self.dataset_path):
            action_path = os.path.join(self.dataset_path, action)
            if not os.path.isdir(action_path):
                continue
                
            print(f"處理 {action} 類別的數據...")
            
            # 遍歷該類別下的所有圖片
            for image_file in os.listdir(action_path):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                image_path = os.path.join(action_path, image_file)
                try:
                    # 提取姿勢特徵
                    features = self.extract_pose_features(image_path)
                    if features is not None:
                        X.append(features)
                        y.append(action)
                except Exception as e:
                    print(f"處理圖片 {image_file} 時發生錯誤: {str(e)}")
        
        return np.array(X), np.array(y)

    def create_lstm_model(self, input_shape, num_classes):
        """
        創建 LSTM 模型
        
        Args:
            input_shape: 輸入特徵的形狀
            num_classes: 分類類別數量
        """
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_model(self, epochs=50, batch_size=32):
        """
        訓練姿勢識別模型
        """
        try:
            print("開始準備數據集...")
            X, y = self.prepare_dataset()
            
            if len(X) == 0:
                raise ValueError("沒有找到有效的訓練數據")
            
            # 編碼標籤
            y_encoded = self.label_encoder.fit_transform(y)
            
            # 重塑數據以適應 LSTM 輸入格式 (samples, timesteps, features)
            n_features = 33 * 3  # MediaPipe 提供 33 個關鍵點，每個點有 x, y, z 座標
            X_reshaped = X.reshape(len(X), 1, n_features)
            
            # 分割訓練集和測試集
            X_train, X_test, y_train, y_test = train_test_split(
                X_reshaped, y_encoded, test_size=0.2, random_state=42
            )
            
            print(f"訓練集大小: {len(X_train)} 樣本")
            print(f"測試集大小: {len(X_test)} 樣本")
            print(f"特徵維度: {X_train.shape}")
            print(f"動作類別: {self.label_encoder.classes_}")
            
            # 創建模型
            model = self.create_lstm_model(
                input_shape=(1, n_features),
                num_classes=len(self.label_encoder.classes_)
            )
            
            # 添加早停機制
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # 訓練模型
            print("開始訓練模型...")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # 評估模型
            print("\n評估模型性能...")
            test_loss, test_acc = model.evaluate(X_test, y_test)
            print(f"測試集準確率: {test_acc:.4f}")
            
            # 保存模型為 .h5 格式
            print("\n保存模型...")
            model_path = os.path.join(self.export_dir, 'pose_model.h5')
            model.save(model_path)
            
            # 保存標籤編碼器
            np.save(
                os.path.join(self.export_dir, 'label_encoder.npy'),
                self.label_encoder.classes_
            )
            
            # 繪製訓練歷史
            self.plot_training_history(history)
            
        except Exception as e:
            raise Exception(f"訓練模型時發生錯誤: {str(e)}")

    def plot_training_history(self, history):
        """
        繪製訓練歷史圖表
        """
        plt.figure(figsize=(12, 4))
        
        # 繪製準確率
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='訓練準確率')
        plt.plot(history.history['val_accuracy'], label='驗證準確率')
        plt.title('模型準確率')
        plt.xlabel('Epoch')
        plt.ylabel('準確率')
        plt.legend()
        
        # 繪製損失
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='訓練損失')
        plt.plot(history.history['val_loss'], label='驗證損失')
        plt.title('模型損失')
        plt.xlabel('Epoch')
        plt.ylabel('損失')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.export_dir, 'training_history.png'))
        plt.show()

    def visualize_pose(self, image_path):
        """
        視覺化圖片中的姿勢檢測結果
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"無法讀取圖片: {image_path}")
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with self.mp_pose.Pose() as pose:
            results = pose.process(image)

            # 繪製姿勢標記
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')
            plt.show()

def main():
    try:
        DATASET_PATH = "exercise_dataset"
        trainer = PoseModelTrainer(DATASET_PATH)
        trainer.train_model(epochs=50, batch_size=32)
        print("訓練完成！")
        
    except Exception as e:
        print(f"發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 