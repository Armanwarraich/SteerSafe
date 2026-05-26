"""
SteerSafe — Gradio Live Webcam App
For HuggingFace Spaces deployment.
Copy this file + steersafe.onnx + shape_predictor_68.dat to HuggingFace repo.
"""
import gradio as gr
import onnxruntime as ort
import numpy as np
import cv2
import dlib
from scipy.spatial import distance

session   = ort.InferenceSession("steersafe.onnx",
                                  providers=['CPUExecutionProvider'])
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

EAR_THRESHOLD   = 0.25
FRAME_THRESHOLD = 20

frame_counter = 0
alert_count   = 0
yawn_count    = 0

LEFT_EYE  = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
MOUTH     = list(range(48, 68))


def euclidean(a, b): return distance.euclidean(a, b)

def ear(pts):
    return (euclidean(pts[1], pts[5]) + euclidean(pts[2], pts[4])) \
           / (2.0 * euclidean(pts[0], pts[3]))

def mar(pts):
    return (euclidean(pts[2], pts[10]) + euclidean(pts[4], pts[8])) \
           / (2.0 * euclidean(pts[0], pts[6]))

def preprocess(frame):
    img  = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = (img - mean) / std
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def detect(frame):
    global frame_counter, alert_count, yawn_count
    if frame is None:
        return None, "No frame", 0.0, "None", 0, 0

    gray  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray, 0)
    ear_val = 0.30
    status  = "No face detected"
    alert   = "None"

    for face in faces:
        shape  = predictor(gray, face)
        coords = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        l_ear   = ear(coords[LEFT_EYE])
        r_ear   = ear(coords[RIGHT_EYE])
        ear_val = (l_ear + r_ear) / 2.0
        mar_val = mar(coords[MOUTH])

        output   = session.run(None, {'input': preprocess(frame)})[0]
        probs    = softmax(output[0])
        pred     = int(np.argmax(probs))
        conf_pct = float(probs[pred] * 100)
        label    = "Awake" if pred == 0 else "Sleepy"

        eye_col = (0, 255, 0) if ear_val >= EAR_THRESHOLD else (255, 0, 0)
        for idx in [LEFT_EYE, RIGHT_EYE]:
            pts = coords[idx]
            cv2.polylines(frame, [pts.reshape((-1, 1, 2))], True, eye_col, 1)

        if mar_val > 0.70:
            yawn_count += 1

        if ear_val < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= FRAME_THRESHOLD:
                alert_count += 1
                alert = "DROWSINESS DETECTED!"
                cv2.rectangle(frame, (0, 0),
                              (frame.shape[1], frame.shape[0]), (255, 0, 0), 6)
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        else:
            frame_counter = 0

        status = f"{label} | EAR: {ear_val:.3f} | {conf_pct:.0f}% confident"
        cv2.putText(frame, status, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, status, round(ear_val, 4), alert, alert_count, yawn_count


with gr.Blocks(title="SteerSafe") as demo:
    gr.Markdown("""
    # 🚗 SteerSafe — Real-Time Driver Drowsiness Detection
    Allow camera access. System monitors eyes and alerts if drowsiness detected.
    **Model:** ResNet-152 | **Accuracy:** 99.10% | **F1:** 99.09%
    """)

    with gr.Row():
        with gr.Column(scale=2):
            cam_in  = gr.Image(source="webcam", streaming=True, label="Camera")
            cam_out = gr.Image(label="Detection Output")
        with gr.Column(scale=1):
            status    = gr.Textbox(label="Status")
            ear_score = gr.Number(label="EAR Score (< 0.25 = drowsy)")
            alert_box = gr.Textbox(label="Alert")
            alerts    = gr.Number(label="Total Alerts", value=0)
            yawns     = gr.Number(label="Yawns", value=0)

    cam_in.stream(fn=detect, inputs=[cam_in],
                  outputs=[cam_out, status, ear_score, alert_box, alerts, yawns])

demo.launch()