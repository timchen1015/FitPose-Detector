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
project_root/
│
├── main.py                    # Main program for real-time detection
├── collect_data.py           # Data collection program
├── trained_pose_model/       # Model training and storage directory
│   ├── train_model.py       # Model training program
│   ├── lstm_model.h5        # Trained model file
│   ├── label_encoder.npy    # Label encoder file
│   └── training_history.png # Training history visualization
│
└── exercise_dataset/ # Training dataset directory
├── push_up/ # Push-up images
│ ├── push_up_0.jpg
│ ├── push_up_1.jpg
│ └── ...
├── squat/ # Squat images
│ ├── squat_0.jpg
│ ├── squat_1.jpg
│ └── ...
└── sit_up/ # Sit-up images
├── sit_up_0.jpg
├── sit_up_1.jpg
└── ...