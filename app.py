"""
app.py — Fire & Smoke detection server
Place this file next to index.html and run:
    python app.py

Model path is set to:
E:\YOLOv8-Fire-and-Smoke-Detection-main\models\best.pt
"""

import io
import os
import time
import uuid
import base64
import tempfile
from pathlib import Path
from threading import Lock

from flask import Flask, request, jsonify, send_file, render_template_string

import cv2
import numpy as np
from ultralytics import YOLO
import torch

# ---------- CONFIG ----------
MODEL_PATH = r"models\best.pt"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "fire_detect_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_CLASSES = {"fire", "smoke"}  # only show these

# ---------- FLASK ----------
app = Flask(__name__, static_folder='.', static_url_path='/static')

# ---------- LOAD MODEL (preload) ----------
models = {}
models_lock = Lock()

def load_model():
    with models_lock:
        if 'best' not in models:
            print(f"[INFO] Loading model from {MODEL_PATH} ...")
            model = YOLO(MODEL_PATH)
            models['best'] = model
            # normalize names (dict or list)
            names = model.names
            # convert to lowercase dict mapping id->name
            if isinstance(names, dict):
                models['names'] = {int(k): v.lower() for k, v in names.items()}
            else:
                models['names'] = {i: str(n).lower() for i, n in enumerate(names)}
            print("[INFO] Model loaded. Classes:", models['names'])
            # determine if GPU is available
            print("[INFO] torch.cuda.is_available():", torch.cuda.is_available())

load_model()

# ---------- UTILITIES ----------
def read_image_from_bytes(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def encode_jpeg_to_dataurl(img_bgr, quality=90):
    ret, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f"data:image/jpeg;base64,{b64}"

def annotate_and_filter(model, names_map, frame, conf=0.25, imgsz=640):
    """
    Run model on frame and draw boxes only for allowed classes.
    Returns annotated frame (BGR), list of detections (dicts), fps_estimate (None here).
    """
    results = model(frame, conf=float(conf), imgsz=imgsz)
    res = results[0]

    # if no boxes -> return original
    if res.boxes is None or len(res.boxes) == 0:
        return frame, [], False

    # extract tensors
    try:
        xyxy = res.boxes.xyxy.cpu().numpy()         # (N,4)
        scores = res.boxes.conf.cpu().numpy().flatten()   # (N,)
        cls_ids = res.boxes.cls.cpu().numpy().astype(int).flatten()  # (N,)
    except Exception:
        # fallback to res.boxes data via list
        detections = []
        return frame, detections, False

    h, w = frame.shape[:2]
    annotated = frame.copy()
    detections = []
    alarm = False

    # drawing params
    thickness = max(2, int(round((w + h) / 600)))

    for (box, score, cid) in zip(xyxy, scores, cls_ids):
        name = names_map.get(int(cid), "unknown").lower()
        if name not in ALLOWED_CLASSES:
            continue  # ignore other classes

        x1, y1, x2, y2 = map(int, box)
        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        label = f"{name} {score:.2f}"
        # color: fire->red, smoke->orange/gray
        if name == "fire":
            color = (0, 0, 255)  # BGR red
        else:
            color = (0, 165, 255)  # orange-ish
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({"class": name, "conf": float(score), "bbox": [int(x1), int(y1), int(x2), int(y2)]})
        alarm = True

    return annotated, detections, alarm

# ---------- ROUTES ----------
@app.route('/')
def index():
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return send_file(str(index_path))
    # fallback
    return "<p>index.html not found. Place index.html in the same folder as app.py</p>"

@app.route('/status')
def status():
    names = models.get('names', {})
    return jsonify({
        "model_loaded": 'best' in models,
        "classes": names,
        "allowed_classes": list(ALLOWED_CLASSES),
        "cuda_available": torch.cuda.is_available()
    })

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify(success=False, error="no file uploaded"), 400
    f = request.files['file']
    conf = float(request.form.get('conf', 0.25))
    img_bytes = f.read()
    frame = read_image_from_bytes(img_bytes)
    if frame is None:
        return jsonify(success=False, error="invalid image"), 400

    model = models['best']
    annotated, detections, alarm = annotate_and_filter(model, models['names'], frame, conf=conf)
    data_url = encode_jpeg_to_dataurl(annotated, quality=90)
    return jsonify(success=True, image=data_url, detections=detections, alarm=alarm)

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    # expects JSON { image: dataURL, conf: float }
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify(success=False, error="no image"), 400
    img_data = data['image']
    if img_data.startswith('data:'):
        header, b64 = img_data.split(',', 1)
        img_bytes = base64.b64decode(b64)
    else:
        img_bytes = base64.b64decode(img_data)
    frame = read_image_from_bytes(img_bytes)
    if frame is None:
        return jsonify(success=False, error="invalid image"), 400

    conf = float(data.get('conf', 0.25))
    model = models['best']
    annotated, detections, alarm = annotate_and_filter(model, models['names'], frame, conf=conf)
    data_url = encode_jpeg_to_dataurl(annotated, quality=80)
    return jsonify(success=True, image=data_url, detections=detections, alarm=alarm)

@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify(success=False, error="no file uploaded"), 400
    f = request.files['file']
    conf = float(request.form.get('conf', 0.25))
    in_fname = OUTPUT_DIR / f"input_{uuid.uuid4().hex}.mp4"
    out_fname = OUTPUT_DIR / f"out_{uuid.uuid4().hex}.mp4"
    f.save(str(in_fname))

    model = models['best']
    cap = cv2.VideoCapture(str(in_fname))
    if not cap.isOpened():
        return jsonify(success=False, error="cannot open video"), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_fname), fourcc, fps, (w, h))

    frame_count = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, _, _ = annotate_and_filter(model, models['names'], frame, conf=conf)
        out.write(annotated)
        frame_count += 1
    cap.release()
    out.release()
    elapsed = time.time() - start
    print(f"[INFO] Processed {frame_count} frames in {elapsed:.1f}s -> {out_fname.name}")
    url = f"/download/{out_fname.name}"
    return jsonify(success=True, url=url, frames=frame_count)

@app.route('/download/<filename>')
def download_file(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return "not found", 404
    return send_file(str(path), as_attachment=True, download_name=filename)

# ---------- RUN ----------
if __name__ == "__main__":
    print("[STARTING] Fire & Smoke detection server")
    print("[INFO] Model path:", MODEL_PATH)
    print("[INFO] Serving at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)