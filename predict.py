"""
predict.py

PyGesture Studio

Real-time gesture recognition
"""

import cv2
import mediapipe as mp
import joblib
import os

# ==========================================
# Load trained model
# ==========================================

MODEL_FILE = "models/gesture_model.joblib"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        "Model not found. Run train.py first."
    )

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")

# ==========================================
# MediaPipe
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================================
# Webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Cannot open webcam.")

# ==========================================
# Main Loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    prediction = "-"
    confidence = "-"

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        features = []

        for lm in hand.landmark:
            features.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        X = [features]

        prediction = model.predict(X)[0]

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(X)[0]

            confidence = max(probs) * 100

            confidence = f"{confidence:.1f}%"

    # ======================================
    # HUD
    # ======================================

    cv2.putText(
        frame,
        f"Gesture : {prediction}",
        (10,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {confidence}",
        (10,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "ESC = Quit",
        (10,115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.imshow(
        "PyGesture Studio - Prediction",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()