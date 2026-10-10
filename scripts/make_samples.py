"""Export one sample face per emotion from the test set into data/samples/."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from src import config
from src.dataset import load_fer2013_csv


def main():
    os.makedirs(config.SAMPLES_DIR, exist_ok=True)
    (_, _), (X_test, y_test) = load_fer2013_csv()
    for i, name in enumerate(config.EMOTIONS):
        idx = int(np.where(y_test == i)[0][0])
        img = (X_test[idx].reshape(config.IMG_SIZE, config.IMG_SIZE) * 255).astype("uint8")
        out = os.path.join(config.SAMPLES_DIR, f"sample_{name}.png")
        cv2.imwrite(out, img)
        print("wrote", out)
    print("done")


if __name__ == "__main__":
    main()
