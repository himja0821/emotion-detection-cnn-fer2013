"""FER-2013 data loading, preprocessing and dataset export.

The project ships the full FER-2013 data as a single labelled CSV
(``emotion, Usage, pixels``).  The Kaggle "msambare/fer2013" release is the
same data re-packed as ``train/<class>/*.png`` and ``test/<class>/*.png``.
This module can load either representation so the pipeline matches the
dataset described in the abstract.
"""

import csv
import os

import numpy as np

from . import config

try:  # TensorFlow is optional for pure data exploration
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
def load_fer2013_csv(csv_path: str = None, img_size: int = None, normalize: bool = True):
    """Load FER-2013 from the canonical CSV.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
        X arrays shaped ``(N, img_size, img_size, 1)``, ``y`` integer labels 0-6.
        ``test`` combines the PublicTest + PrivateTest splits, matching the
        7,178-image test set of the msambare Kaggle release.
    """
    csv_path = csv_path or config.CSV_PATH
    img_size = img_size or config.IMG_SIZE

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"FER-2013 CSV not found at {csv_path}.\n"
            "Run  python scripts/prepare_data.py  or download the dataset first."
        )

    X_train, y_train, X_test, y_test = [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header: emotion, Usage, pixels
        for row in reader:
            if len(row) < 3:
                continue
            emotion = int(row[0])
            usage = row[1].strip()
            pixels = np.asarray(row[2].split(), dtype="float32").reshape(img_size, img_size, 1)
            if usage == "Training":
                X_train.append(pixels)
                y_train.append(emotion)
            else:
                X_test.append(pixels)
                y_test.append(emotion)

    X_train = np.stack(X_train)
    X_test = np.stack(X_test)
    y_train = np.asarray(y_train, dtype="int64")
    y_test = np.asarray(y_test, dtype="int64")

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    return (X_train, y_train), (X_test, y_test)


# --------------------------------------------------------------------------- #
# msambare-style folder export
# --------------------------------------------------------------------------- #
def export_to_folders(csv_path: str = None, out_root: str = None, img_size: int = None) -> dict:
    """Convert the CSV into ``train/<class>`` and ``test/<class>`` PNG folders.

    Mirrors the layout of https://www.kaggle.com/datasets/msambare/fer2013 .
    """
    import cv2

    csv_path = csv_path or config.CSV_PATH
    out_root = out_root or config.DATA_DIR
    img_size = img_size or config.IMG_SIZE

    train_dir = os.path.join(out_root, "train")
    test_dir = os.path.join(out_root, "test")
    for split_dir in (train_dir, test_dir):
        for name in config.EMOTIONS:
            os.makedirs(os.path.join(split_dir, name), exist_ok=True)

    counters = {"train": {e: 0 for e in config.EMOTIONS}, "test": {e: 0 for e in config.EMOTIONS}}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            emotion = int(row[0])
            usage = row[1].strip()
            img = np.asarray(row[2].split(), dtype="uint8").reshape(img_size, img_size)
            split = "train" if usage == "Training" else "test"
            name = config.EMOTIONS[emotion]
            idx = counters[split][name]
            out_path = os.path.join(out_root, split, name, f"{name}_{idx:05d}.png")
            cv2.imwrite(out_path, img)
            counters[split][name] += 1

    return counters


def load_folder_dataset(root: str = None, img_size: int = None, batch_size: int = None,
                        subset: str = "training", validation_split: float = None, seed: int = 42):
    """Load FER-2013 from the msambare-style folder tree via tf.data."""
    if tf is None:
        raise RuntimeError("TensorFlow is required for folder loading.")

    root = root or config.TRAIN_DIR
    img_size = img_size or config.IMG_SIZE
    batch_size = batch_size or config.BATCH_SIZE
    validation_split = config.VALIDATION_SPLIT if validation_split is None else validation_split

    return tf.keras.utils.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
    )


# --------------------------------------------------------------------------- #
# Convenience: class weights for the imbalanced FER-2013 classes
# --------------------------------------------------------------------------- #
def compute_class_weights(y_train) -> dict:
    y_train = np.asarray(y_train)
    counts = np.bincount(y_train, minlength=config.NUM_CLASSES).astype("float64")
    counts[counts == 0] = 1.0
    total = counts.sum()
    return {i: float(total / (config.NUM_CLASSES * counts[i])) for i in range(config.NUM_CLASSES)}


def class_distribution(y) -> dict:
    y = np.asarray(y)
    counts = np.bincount(y, minlength=config.NUM_CLASSES)
    return {config.EMOTIONS[i]: int(counts[i]) for i in range(config.NUM_CLASSES)}


if __name__ == "__main__":  # quick smoke test
    (Xtr, ytr), (Xte, yte) = load_fer2013_csv()
    print("train:", Xtr.shape, "test:", Xte.shape)
    print("train distribution:", class_distribution(ytr))
    print("test  distribution:", class_distribution(yte))
