"""
features.py

Step 2
-------
Extract hand landmarks using MediaPipe.

This script displays:
    - Webcam image
    - Hand landmarks
    - Number of extracted features
    - Landmark coordinates in the console

Press ESC to exit.
"""

import cv2
import mediapipe as mp
import time

# -----------------------------
# Initialize MediaPipe
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# Feature extraction function
# -----------------------------

def extract_features(hand_landmarks):

    features = []

    for landmark in hand_landmarks.landmark:
        features.append(landmark.x)
        features.append(landmark.y)
        features.append(landmark.z)

    return features


# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Cannot open webcam.")

prev_time = time.time()

# Used only to avoid printing every frame
last_print = 0

# -----------------------------
# Main loop
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        features = extract_features(hand)

        # Print once per second
        if time.time() - last_print > 1:

            last_print = time.time()

            print("=" * 60)
            print("Hand detected")
            print("Feature vector length:", len(features))
            print()

            for i in range(21):

                x = features[i * 3]
                y = features[i * 3 + 1]
                z = features[i * 3 + 2]

                print(
                    f"Landmark {i:02d}: "
                    f"x={x:.4f}  "
                    f"y={y:.4f}  "
                    f"z={z:.4f}"
                )

    # -----------------------------
    # FPS
    # -----------------------------

    current = time.time()

    fps = 1.0 / (current - prev_time)

    prev_time = current

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        "ESC = Quit",
        (10,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("PyGesture Studio - Feature Extraction", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()