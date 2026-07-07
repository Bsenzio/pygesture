"""
camera.py

Step 1
------
Open webcam and detect a hand using MediaPipe.

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
# Open webcam
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Cannot open webcam.")

prev_time = time.time()

# -----------------------------
# Main Loop
# -----------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # -----------------------------
    # FPS
    # -----------------------------

    current_time = time.time()

    fps = 1.0 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "ESC = Quit",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("PyGesture Studio - Camera", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

# -----------------------------
# Cleanup
# -----------------------------

cap.release()

cv2.destroyAllWindows()