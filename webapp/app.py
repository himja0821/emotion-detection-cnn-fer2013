"""Minimal Flask web demo for the FER-2013 emotion CNN.

    python webapp/app.py
    # then open http://127.0.0.1:5000

Endpoints
---------
GET  /              demo page
POST /api/predict   {"image": "data:image/png;base64,..."} -> emotion + probs
GET  /health        liveness / model-info probe
"""

import base64
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from flask import Flask, jsonify, render_template, request

from src import config
from src.face_detector import FaceDetector
from src.model import load_trained_model
from src.predict import predict_array

app = Flask(__name__)

print("Loading model ...")
MODEL = load_trained_model()
DETECTOR = FaceDetector()


def _decode_image(data_url: str):
    import cv2

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_image(bgr) -> str:
    import cv2

    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


@app.route("/")
def index():
    return render_template("index.html", emotions=config.EMOTIONS)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": os.path.basename(config.MODEL_PATH),
        "classes": config.EMOTIONS,
        "input_size": [config.IMG_SIZE, config.IMG_SIZE],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    import cv2

    image_b64 = None
    if request.is_json:
        image_b64 = (request.get_json(silent=True) or {}).get("image")
    elif "file" in request.files:
        file = request.files["file"]
        image_b64 = "data:image/png;base64," + base64.b64encode(file.read()).decode("ascii")

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    img = _decode_image(image_b64)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = predict_array(MODEL, img)

    annotated = img.copy()
    if result["face_box"]:
        x, y, w, h = result["face_box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (46, 204, 113), 2)
    cv2.putText(annotated, f"{result['emotion']} {result['confidence'] * 100:.1f}%",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (46, 204, 113), 2)

    return jsonify({
        "emotion": result["emotion"],
        "confidence": round(result["confidence"] * 100, 2),
        "probabilities": {k: round(v * 100, 2) for k, v in result["probabilities"].items()},
        "face_detected": result["face_box"] is not None,
        "annotated_image": _encode_image(annotated),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
