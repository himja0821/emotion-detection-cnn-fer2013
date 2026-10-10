"""Evaluate the trained CNN on the FER-2013 test set.

Produces
--------
    results/confusion_matrix.png            raw counts
    results/confusion_matrix_normalized.png row-normalised
    results/classification_report.txt       precision / recall / f1 per class
    results/metrics.json                    headline numbers
    results/sample_predictions.png          example predictions
    results/per_class_accuracy.png          bar chart
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config
from src.dataset import load_fer2013_csv
from src.model import load_trained_model


def _plot_confusion(cm, emotions, out_path, normalized=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if normalized:
        with np.errstate(all="ignore"):
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(8.5, 7))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=emotions, yticklabels=emotions, cbar=True,
                square=True, linewidths=0.5)
    plt.xlabel("Predicted emotion")
    plt.ylabel("True emotion")
    title = "Normalized confusion matrix" if normalized else "Confusion matrix"
    plt.title(title + " - FER-2013 test set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_samples(X, y_true, y_pred, emotions, out_path, n=16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.random.default_rng(0).choice(len(X), size=min(n, len(X)), replace=False)
    cols, rows = 4, int(np.ceil(len(idx) / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.8))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X[i].reshape(config.IMG_SIZE, config.IMG_SIZE), cmap="gray")
        color = "green" if y_true[i] == y_pred[i] else "red"
        ax.set_title(f"T:{emotions[y_true[i]]}\nP:{emotions[y_pred[i]]}",
                     color=color, fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    fig.suptitle("Sample predictions (green = correct, red = wrong)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_per_class(acc_per_class, emotions, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(emotions, acc_per_class, color="#4C72B0")
    plt.ylim(0, 1)
    plt.ylabel("accuracy")
    plt.title("Per-class accuracy on FER-2013 test set")
    for b, v in zip(bars, acc_per_class):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                 ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the emotion CNN")
    parser.add_argument("--model", default=config.MODEL_PATH)
    parser.add_argument("--csv", default=config.CSV_PATH)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args(argv)

    from sklearn.metrics import classification_report, confusion_matrix

    emotions = config.EMOTIONS
    print("Loading model:", args.model)
    model = load_trained_model(args.model)

    print("Loading test data ...")
    (_, _), (X_test, y_test) = load_fer2013_csv(args.csv)
    print("test set:", X_test.shape)

    probs = model.predict(X_test, batch_size=args.batch, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    acc = float(np.mean(y_pred == y_test))
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(y_test, y_pred, target_names=emotions, digits=4)
    per_class_acc = [float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0
                     for i in range(len(emotions))]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("FER-2013 - Emotion CNN classification report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Overall accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    _plot_confusion(cm, emotions, os.path.join(config.RESULTS_DIR, "confusion_matrix.png"))
    _plot_confusion(cm, emotions, os.path.join(config.RESULTS_DIR, "confusion_matrix_normalized.png"), normalized=True)
    _plot_samples(X_test, y_test, y_pred, emotions, os.path.join(config.RESULTS_DIR, "sample_predictions.png"))
    _plot_per_class(per_class_acc, emotions, os.path.join(config.RESULTS_DIR, "per_class_accuracy.png"))

    metrics = {
        "overall_accuracy": round(acc, 4),
        "overall_accuracy_percent": round(acc * 100, 2),
        "per_class_accuracy": {e: round(a, 4) for e, a in zip(emotions, per_class_acc)},
        "confusion_matrix": cm.tolist(),
        "test_samples": int(len(y_test)),
    }
    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 55)
    print(f"Overall test accuracy: {acc * 100:.2f}%")
    print("=" * 55)
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nArtifacts written to", config.RESULTS_DIR)
    return metrics


if __name__ == "__main__":
    main()
