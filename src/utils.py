"""Shared helper utilities (seeding, plotting, label I/O)."""

import json
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Make runs as reproducible as TensorFlow allows on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def save_labels(path: str, emotions) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"emotions": list(emotions)}, f, indent=2)


def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["emotions"]


def plot_history(history: dict, out_path: str) -> None:
    """Plot training/validation accuracy and loss curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
