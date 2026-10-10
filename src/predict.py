"""Run emotion prediction on an image, a folder, or a webcam stream.

Examples
--------
    python -m src.predict --image path\\to\\face.jpg
    python -m src.predict --image path\\to\\face.jpg --save out.png
    python -m src.predict --webcam
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config
from src.face_detector import FaceDetector
from src.model import load_trained_model

_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = FaceDetector()
    return _DETECTOR


def preprocess_face(face_img) -> np.ndarray:
    """BGR/grayscale image -> (1, 48, 48, 1) float32 tensor in [0, 1]."""
    import cv2

    if face_img is None or getattr(face_img, "size", 0) == 0:
        raise ValueError("empty image")
    if face_img.ndim == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    resized = cv2.resize(gray, (config.IMG_SIZE, config.IMG_SIZE))
    return (resized.astype("float32") / 255.0).reshape(1, config.IMG_SIZE, config.IMG_SIZE, 1)


def predict_array(model, image_bgr):
    """Predict a single BGR image, cropping the largest face when detected."""
    face = image_bgr
    box = None
    detected = _detector().largest(image_bgr)
    if detected is not None:
        x, y, w, h, _conf = detected
        box = (x, y, w, h)
        face = image_bgr[y:y + h, x:x + w]

    tensor = preprocess_face(face)
    probs = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "emotion": config.EMOTIONS[idx],
        "confidence": float(probs[idx]),
        "probabilities": {e: float(p) for e, p in zip(config.EMOTIONS, probs)},
        "face_box": box,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Emotion prediction on an image / webcam")
    parser.add_argument("--image", help="path to an image file")
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--webcam", action="store_true", help="run live webcam demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--save", help="save annotated image to this path")
    args = parser.parse_args(argv)

    model = load_trained_model(args.model)

    if args.webcam:
        return _run_webcam(model, args.camera)

    if not args.image:
        parser.error("provide --image PATH or --webcam")

    import cv2

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    result = predict_array(model, img)
    print(f"\nPredicted emotion : {result['emotion']}")
    print(f"Confidence        : {result['confidence'] * 100:.2f}%")
    print("\nAll probabilities:")
    for e, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(p * 40))
        print(f"  {e:>9}: {p * 100:6.2f}%  {bar}")

    if args.save:
        if result["face_box"]:
            x, y, w, h = result["face_box"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{result['emotion']} {result['confidence'] * 100:.1f}%"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite(args.save, img)
        print("\nSaved annotated image ->", args.save)
    return result


def _run_webcam(model, camera_index=0):
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = predict_array(model, frame)
        if result["face_box"]:
            x, y, w, h = result["face_box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{result['emotion']} {result['confidence'] * 100:.1f}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection (press q)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
