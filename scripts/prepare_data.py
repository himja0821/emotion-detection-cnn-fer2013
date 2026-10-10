"""Prepare the FER-2013 dataset.

Verifies the CSV (``data/fer2013/icml_face_data.csv``) and optionally exports
the msambare-style ``train/<class>`` and ``test/<class>`` folder tree.

    python scripts/prepare_data.py                 # verify + build folders
    python scripts/prepare_data.py --verify-only   # just check the CSV
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.dataset import class_distribution, export_to_folders, load_fer2013_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    print("CSV:", config.CSV_PATH)
    (Xtr, ytr), (Xte, yte) = load_fer2013_csv()
    print(f"train: {Xtr.shape}   test: {Xte.shape}")
    print("train distribution:", class_distribution(ytr))
    print("test  distribution:", class_distribution(yte))
    print(f"total images: {len(ytr) + len(yte)}")

    if args.verify_only:
        return

    print("\nExporting msambare-style folders ...")
    counters = export_to_folders()
    for split in ("train", "test"):
        total = sum(counters[split].values())
        print(f"  {split}: {total} images -> {os.path.join(config.DATA_DIR, split)}")
    print("Done.")


if __name__ == "__main__":
    main()
