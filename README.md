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
│   ├── lstm_model.keras
│   ├── label_encoder.npy
│   └── training_history.png
│
└── exercise_dataset/  # Not included in the repository due to size constraints
    ├── image_dataset/
    ├── img_dataset/
    └── video_dataset/
```

## Dataset

The dataset used for training the model is not included in this repository due to its large size. It consists of:

1. Image datasets of push-ups, sit-ups, and squats
2. Video datasets used for training and testing

If you need access to the dataset, please [contact the repository owner](https://github.com/timchen1015) or use the provided `collect_data.py` script to create your own dataset.
