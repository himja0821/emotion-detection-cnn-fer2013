"""Central configuration for the Emotion Detection project.

Dataset : FER-2013 (Facial Expression Recognition 2013)
Source  : https://www.kaggle.com/datasets/msambare/fer2013
Task    : 7-class facial expression classification with a CNN.
"""

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "fer2013")
CSV_PATH = os.path.join(DATA_DIR, "icml_face_data.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train")          # msambare-style folder
TEST_DIR = os.path.join(DATA_DIR, "test")            # msambare-style folder
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(MODELS_DIR, "emotion_cnn_fer2013.keras")
MODEL_H5_PATH = os.path.join(MODELS_DIR, "emotion_cnn_fer2013.h5")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.json")
LABELS_PATH = os.path.join(MODELS_DIR, "emotion_labels.json")

# --------------------------------------------------------------------------- #
# Dataset / model hyper-parameters
# --------------------------------------------------------------------------- #
IMG_SIZE = 48            # FER-2013 images are 48x48 grayscale
NUM_CLASSES = 7
CHANNELS = 1

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Original FER-2013 integer label -> emotion name
EMOTION_MAP = {i: name for i, name in enumerate(EMOTIONS)}

BATCH_SIZE = 128
EPOCHS = 100             # upper bound; EarlyStopping usually stops sooner
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.10  # carved out of the training set

# Deadline guard: never train for longer than this (minutes).
MAX_TRAINING_MINUTES = 70

# Optional hard cap so the pipeline stays inside a tight time budget.
MAX_EPOCHS = 100

RANDOM_SEED = 42

for _d in (MODELS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)
