"""Unit tests for data, model and inference pipeline."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

from src import config
from src.dataset import class_distribution, compute_class_weights, load_fer2013_csv
from src.model import build_cnn
from src.predict import preprocess_face


@pytest.fixture(scope="session")
def data():
    return load_fer2013_csv()


@pytest.fixture(scope="session")
def model():
    return build_cnn()


def test_dataset_shapes(data):
    (X_train, y_train), (X_test, y_test) = data
    assert X_train.shape[1:] == (config.IMG_SIZE, config.IMG_SIZE, 1)
    assert X_test.shape[1:] == (config.IMG_SIZE, config.IMG_SIZE, 1)
    assert X_train.max() <= 1.0 and X_train.min() >= 0.0


def test_dataset_size(data):
    (X_train, _), (X_test, _) = data
    assert len(X_train) == 28709
    assert len(X_test) == 7178


def test_class_distribution(data):
    (_, y_train), (_, y_test) = data
    dist = class_distribution(y_train)
    assert set(dist.keys()) == set(config.EMOTIONS)
    assert sum(dist.values()) == len(y_train)


def test_class_weights(data):
    (_, y_train), _ = data
    weights = compute_class_weights(y_train)
    assert len(weights) == config.NUM_CLASSES
    # rare classes must receive a larger weight than frequent ones
    assert weights[1] > weights[3]  # disgust > happy


def test_model_output_shape(model):
    out = model.predict(np.zeros((3, config.IMG_SIZE, config.IMG_SIZE, 1), dtype="float32"),
                        verbose=0)
    assert out.shape == (3, config.NUM_CLASSES)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(3), atol=1e-4)


def test_model_param_count(model):
    assert model.count_params() > 100_000


def test_preprocess_face_shape():
    img = np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8)
    tensor = preprocess_face(img)
    assert tensor.shape == (1, config.IMG_SIZE, config.IMG_SIZE, 1)
    assert tensor.max() <= 1.0


def test_preprocess_grayscale():
    img = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
    tensor = preprocess_face(img)
    assert tensor.shape == (1, config.IMG_SIZE, config.IMG_SIZE, 1)


def test_face_detector_smoke():
    from src.face_detector import FaceDetector

    det = FaceDetector()
    boxes = det.detect(np.zeros((200, 200, 3), dtype=np.uint8))
    assert isinstance(boxes, list)


def test_predict_format(model):
    from src.predict import predict_array

    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    result = predict_array(model, img)
    assert result["emotion"] in config.EMOTIONS
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["probabilities"]) == config.NUM_CLASSES
