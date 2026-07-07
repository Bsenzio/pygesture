import cv2
import mediapipe as mp
import pandas as pd
import os

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
# Dataset
# ==========================================

os.makedirs("dataset", exist_ok=True)

csv_file = "dataset/gestures.csv"

columns = []

for i in range(21):
    columns.extend([f"x{i}", f"y{i}", f"z{i}"])

columns.append("class")

buffer = []

if os.path.exists(csv_file):
    existing = pd.read_csv(csv_file)
    total_samples = len(existing)
else:
    pd.DataFrame(columns=columns).to_csv(csv_file, index=False)
    total_samples = 0

# ==========================================
# Webcam
# ==========================================

cap = cv2.VideoCapture(0)

current_class = 1
recording = False
session_samples = 0

print("Click once on the camera window to give it focus.")

# ==========================================
# Main loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
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

        if recording:

            features = []

            for lm in hand.landmark:
                features.extend([lm.x, lm.y, lm.z])

            buffer.append(features + [current_class])

            session_samples += 1
            total_samples += 1

    # ======================================
    # HUD
    # ======================================

    color = (0, 255, 0)

    if recording:
        color = (0, 0, 255)

    cv2.putText(
        frame,
        f"Class : {current_class}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Recording : {'YES' if recording else 'NO'}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Session Samples : {session_samples}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Dataset Samples : {total_samples}",
        (10,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        "1-9 Class",
        (10,170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        "R Start   S Stop",
        (10,200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        "ESC Quit",
        (10,230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    if recording:

        cv2.circle(frame, (600,30), 10, (0,0,255), -1)

        cv2.putText(
            frame,
            "REC",
            (620,38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

    cv2.imshow("PyGesture Studio - Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Select class
    if ord('1') <= key <= ord('9'):
        current_class = int(chr(key))
        session_samples = 0

    # Start recording
    elif key == ord('r'):
        recording = True
        session_samples = 0

    # Stop recording
    elif key == ord('s'):
        recording = False

    # Quit
    elif key == 27:
        break

# ==========================================
# Save dataset
# ==========================================

if len(buffer) > 0:

    pd.DataFrame(buffer, columns=columns).to_csv(
        csv_file,
        mode="a",
        header=False,
        index=False
    )

print(f"\nSaved {len(buffer)} new samples.")

cap.release()
cv2.destroyAllWindows()