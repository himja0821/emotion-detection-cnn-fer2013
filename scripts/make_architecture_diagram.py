"""Generate the CNN architecture diagram used in the project report.

This is a documentation helper only: it draws the layer stack with Matplotlib
(no TensorFlow / Graphviz required) and saves ``docs/architecture.png``.

    python scripts/make_architecture_diagram.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# (title, detail, face-color, edge-color)
BLOCKS = [
    ("Input", "48 x 48 x 1 grayscale face", "#ECEFF1", "#546E7A"),
    ("Data augmentation (training only)",
     "RandomRotation +-5%  .  RandomTranslation +-5%  .  RandomZoom +-5%",
     "#F3E5F5", "#8E24AA"),
    ("Block 1", "Conv2D(32) -> BN -> ReLU   x2\nMaxPool 2x2 -> Dropout 0.20   ->   24 x 24 x 32",
     "#E3F2FD", "#1565C0"),
    ("Block 2", "Conv2D(64) -> BN -> ReLU   x2\nMaxPool 2x2 -> Dropout 0.20   ->   12 x 12 x 64",
     "#E3F2FD", "#1565C0"),
    ("Block 3", "Conv2D(128) -> BN -> ReLU   x2\nMaxPool 2x2 -> Dropout 0.30   ->   6 x 6 x 128",
     "#E3F2FD", "#1565C0"),
    ("Classifier head",
     "Flatten (4608) -> Dense(256) -> BN -> ReLU\n-> Dropout 0.40 -> Dense(7, softmax)",
     "#FFF3E0", "#EF6C00"),
    ("Output",
     "angry  .  disgust  .  fear  .  happy  .  sad  .  surprise  .  neutral",
     "#E8F5E9", "#2E7D32"),
]


def main(out_path=None):
    out_path = out_path or os.path.join(config.BASE_DIR, "docs", "architecture.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(BLOCKS) * 1.5 + 0.5)
    ax.axis("off")

    box_w, box_h = 8.2, 1.0
    x0 = (10 - box_w) / 2.0
    y = len(BLOCKS) * 1.5

    centers = []
    for title, detail, face, edge in BLOCKS:
        y -= 1.5
        box = FancyBboxPatch(
            (x0, y), box_w, box_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x0 + 0.25, y + box_h - 0.28, title, fontsize=12.5,
                fontweight="bold", color=edge, va="center", zorder=3)
        ax.text(x0 + 0.25, y + 0.34, detail, fontsize=9.6, color="#37474F",
                va="center", zorder=3)
        centers.append((y, y + box_h))

    for i in range(len(centers) - 1):
        y_top = centers[i][0]
        y_bottom_next = centers[i + 1][1]
        arrow = FancyArrowPatch(
            (5, y_top), (5, y_bottom_next),
            arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color="#90A4AE",
            zorder=1,
        )
        ax.add_patch(arrow)

    ax.set_title("Emotion CNN architecture - FER-2013 (1,470,247 parameters)",
                 fontsize=14, fontweight="bold", pad=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved architecture diagram ->", out_path)
    return out_path


if __name__ == "__main__":
    main()
