"""Face detection using OpenCV's DNN SSD (res10) detector.

OpenCV 5.x removed the bundled Haar cascades, so this project ships the small
Caffe SSD face detector (``deploy.prototxt`` + ``res10_300x300_ssd_iter_140000``)
in ``models/``.  If the files are missing, detection degrades gracefully to
"no face box" and the whole image is classified.
"""

import os

import numpy as np

from . import config


class FaceDetector:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.net = self._load_net()

    @staticmethod
    def _load_net():
        prototxt = os.path.join(config.MODELS_DIR, "deploy.prototxt")
        caffemodel = os.path.join(config.MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            try:
                import cv2

                return cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            except Exception:
                return None
        return None

    def detect(self, image_bgr):
        """Return a list of ``(x, y, w, h, confidence)`` face boxes."""
        if image_bgr is None or self.net is None:
            return []
        import cv2

        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image_bgr, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), confidence))
        return boxes

    def largest(self, image_bgr):
        boxes = self.detect(image_bgr)
        if not boxes:
            return None
        return max(boxes, key=lambda b: b[2] * b[3])
