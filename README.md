# PyGesture Studio

A modern machine learning toolkit for real-time gesture recognition using a webcam.

PyGesture Studio is inspired by the Gesture Recognition Toolkit (GRT), but built with modern Python libraries such as OpenCV, MediaPipe, and PyCaret.

The goal is to simplify the collection of gesture datasets, automatic model training, and real-time gesture prediction.

---

## Features

- Webcam acquisition
- Hand tracking with MediaPipe
- Automatic feature extraction
- Dataset recording
- Machine learning with PyCaret
- Real-time prediction
- Model saving and loading

---

## Planned Features

- Multiple gesture classes
- Confidence estimation
- Live training
- Model comparison
- Performance metrics
- GUI interface
- Export trained models

---

## Project Structure

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
├── models/
├── screenshots/
└── docs/
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Current Status

### Step 1

- [x] Webcam acquisition
- [x] Hand detection using MediaPipe

### Next Steps

- [ ] Landmark extraction
- [ ] Dataset recording
- [ ] Model training
- [ ] Live prediction

---

## Dependencies

- Python 3.11+
- OpenCV
- MediaPipe
- NumPy
- Pandas
- PyCaret

---

## License

MIT License