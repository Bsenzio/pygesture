# PyGesture Studio

**PyGesture Studio** is an open-source machine learning toolkit for real-time hand gesture recognition using a standard webcam.

Inspired by the Gesture Recognition Toolkit (GRT), this project provides a modern Python-based workflow for creating gesture recognition applications using **OpenCV**, **MediaPipe**, and **scikit-learn**.

The project allows users to:

- Capture gesture datasets from a webcam.
- Automatically extract hand landmarks.
- Train and compare multiple machine learning classifiers.
- Perform real-time gesture recognition.
- Build custom Human-Computer Interaction (HCI) applications.

---

# Features

- Real-time webcam acquisition
- Hand tracking using MediaPipe
- Automatic extraction of 21 hand landmarks (63 features)
- Multi-class gesture dataset recording
- Automatic classifier training
- Multiple machine learning algorithms
- Real-time gesture prediction
- Model saving and loading using Joblib

---

# Machine Learning Algorithms

PyGesture Studio automatically compares several classifiers, including:

- Random Forest
- Extra Trees
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- Multi-Layer Perceptron (MLP)

The best-performing model is automatically selected and saved.

---

# Current Workflow

```
Webcam
    │
    ▼
MediaPipe Hand Tracking
    │
    ▼
Feature Extraction (63 Features)
    │
    ▼
Dataset Recording
    │
    ▼
Model Training
    │
    ▼
Real-Time Prediction
```

---

# Project Structure

```
PyGestureStudio/
│
├── camera.py
├── features.py
├── capture.py
├── train.py
├── predict.py
│
├── dataset/
│   └── gestures.csv
│
├── models/
│   └── gesture_model.joblib
│
├── screenshots/
├── docs/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PyGestureStudio.git

cd PyGestureStudio
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## 1. Test the webcam

```bash
python camera.py
```

---

## 2. Extract hand landmarks

```bash
python features.py
```

---

## 3. Record gesture samples

```bash
python capture.py
```

---

## 4. Train the classifier

```bash
python train.py
```

---

## 5. Run real-time prediction

```bash
python predict.py
```

---

# Current Status

| Module | Status |
|---------|--------|
| Webcam Acquisition | ✅ |
| Hand Detection | ✅ |
| Feature Extraction | ✅ |
| Dataset Recording | ✅ |
| Model Training | ✅ |
| Real-Time Prediction | ✅ |

---

# Planned Features

- Graphical User Interface (GUI)
- Live dataset visualization
- Automatic recording sessions
- Confidence visualization
- Cross-validation reports
- Confusion matrix visualization
- Custom gesture labels
- Multiple hand support
- Pose recognition
- Face gesture recognition
- ONNX model export
- TensorFlow/PyTorch support

---

# Dependencies

- Python 3.11+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- scikit-learn
- Joblib

---

# Inspiration

This project is inspired by the **Gesture Recognition Toolkit (GRT)** developed by Professor Nick Gillian, while providing a modern implementation based on current computer vision and machine learning libraries.

---

# License

MIT License