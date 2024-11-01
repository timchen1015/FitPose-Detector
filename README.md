# Exercise Pose Detection System

A real-time exercise pose detection system based on MediaPipe and LSTM, capable of recognizing different exercise movements including push-ups, squats, and sit-ups.

## Features

- Real-time pose detection
- Support for multiple exercise movements
- LSTM deep learning model
- Real-time skeleton tracking
- Display of recognition results and confidence scores

## System Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```bash
project_root/
│
├── main.py
├── collect_data.py
├── trained_pose_model/
│   ├── train_model.py
│   ├── lstm_model.h5
│   ├── label_encoder.npy
│   └── training_history.png
│
└── exercise_dataset/
├── push_up/
│ ├── push_up_0.jpg
│ ├── push_up_1.jpg
│ └── ...
├── squat/
│ ├── squat_0.jpg
│ ├── squat_1.jpg
│ └── ...
└── sit_up/
├── sit_up_0.jpg
├── sit_up_1.jpg
└── ...
```
