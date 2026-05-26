"""
SteerSafe — Real-time local webcam detection.
Run: python src/detection.py
Requirements: webcam, steersafe.onnx, shape_predictor_68_face_landmarks.dat
"""

import cv2
import dlib
import numpy as np
import pygame
import csv
import os
from datetime import datetime
import onnxruntime as ort

# Add src to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ear_utils import calculate_EAR, calculate_MAR, landmarks_to_numpy

# ── Config ────────────────────────────────────────────────
EAR_THRESHOLD   = 0.25
MAR_THRESHOLD   = 0.70
FRAME_THRESHOLD = 20

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, 'models', 'steersafe.onnx')
PREDICTOR_PATH = os.path.join(BASE_DIR, 'models', 'shape_predictor_68_face_landmarks.dat')
ALARM_PATH     = os.path.join(BASE_DIR, 'assets', 'alarm.wav')
LOG_PATH       = os.path.join(BASE_DIR, 'logs', 'session_log.csv')

# dlib landmark indices
LEFT_EYE  = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
MOUTH     = list(range(48, 68))


def load_onnx_model():
    """Load ONNX model with GPU fallback to CPU."""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=providers
    )

    print("ONNX model loaded ✓")
    print("Using provider:", session.get_providers()[0])

    return session


def predict_onnx(session, frame):
    """Run ONNX inference on single frame."""
    img = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    inp = img.transpose(2, 0, 1)[np.newaxis]

    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: inp})[0]

    probs = np.exp(out[0]) / np.sum(np.exp(out[0]))
    pred = int(np.argmax(probs))

    return ('Awake' if pred == 0 else 'Sleepy'), float(probs[pred])


# ── Alarm setup ───────────────────────────────────────────
pygame.mixer.init()
alarm_on = False


def start_alarm():
    global alarm_on
    if not alarm_on and os.path.exists(ALARM_PATH):
        pygame.mixer.music.load(ALARM_PATH)
        pygame.mixer.music.play(-1)
        alarm_on = True


def stop_alarm():
    global alarm_on
    if alarm_on:
        pygame.mixer.music.stop()
        alarm_on = False


def init_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['timestamp', 'ear', 'mar', 'model_pred', 'event'])


def log_event(ear, mar, pred, event):
    with open(LOG_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([
            datetime.now().strftime('%H:%M:%S'),
            round(ear, 4), round(mar, 4), pred, event
        ])


# ── Main detection loop ───────────────────────────────────
def run():
    print("Loading models...")
    model = load_onnx_model()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    cap = cv2.VideoCapture(0)

    # Fix 3 — Reduce frame resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    frame_count = 0
    yawn_count = 0
    alert_count = 0
    frame_num = 0

    pred_label = "Awake"
    conf = 1.0

    init_log()

    print("SteerSafe running! Press Q to quit.")
    print(f"EAR threshold: {EAR_THRESHOLD} | Frame threshold: {FRAME_THRESHOLD}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        ear_val = 0.30
        mar_val = 0.00

        for face in faces:
            shape = predictor(gray, face)
            coords = landmarks_to_numpy(shape)

            left_eye = coords[LEFT_EYE]
            right_eye = coords[RIGHT_EYE]
            mouth_pts = coords[MOUTH]

            ear_val = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0
            mar_val = calculate_MAR(mouth_pts)

            # Fix 2 — Predict every 3rd frame
            frame_num += 1
            if frame_num % 3 == 0:
                pred_label, conf = predict_onnx(model, frame)

            # Draw eye contours
            eye_color = (0, 255, 0) if ear_val >= EAR_THRESHOLD else (0, 0, 255)
            cv2.polylines(frame, [left_eye], True, eye_color, 1)
            cv2.polylines(frame, [right_eye], True, eye_color, 1)

            # Yawn detection
            if mar_val > MAR_THRESHOLD:
                yawn_count += 1
                cv2.putText(frame, "YAWNING DETECTED", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                log_event(ear_val, mar_val, pred_label, 'YAWN')

            # Drowsiness logic
            if ear_val < EAR_THRESHOLD:
                frame_count += 1
                if frame_count >= FRAME_THRESHOLD:
                    start_alarm()
                    alert_count += 1
                    log_event(ear_val, mar_val, pred_label, 'DROWSY_ALERT')

                    cv2.rectangle(
                        frame,
                        (0, 0),
                        (frame.shape[1], frame.shape[0]),
                        (0, 0, 255),
                        8
                    )

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    cv2.putText(frame,
                                f"Eyes closing... {frame_count}/{FRAME_THRESHOLD}",
                                (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 165, 255),
                                2)
            else:
                frame_count = 0
                stop_alarm()

            # Stats overlay
            status_color = (0, 255, 0) if ear_val >= EAR_THRESHOLD else (0, 0, 255)

            cv2.putText(
                frame,
                f"EAR: {ear_val:.3f} | {pred_label} {conf*100:.0f}%",
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2
            )

            cv2.putText(
                frame,
                f"Alerts: {alert_count} | Yawns: {yawn_count}",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

        cv2.imshow("SteerSafe — Driver Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()

    print("\nSession ended.")
    print(f"Total Alerts: {alert_count} | Total Yawns: {yawn_count}")
    print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    run()