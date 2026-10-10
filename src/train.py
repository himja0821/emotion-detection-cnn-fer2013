"""Train the FER-2013 emotion CNN.

Usage
-----
    python -m src.train                 # full pipeline with default budget
    python -m src.train --epochs 40 --minutes 30 --batch 128

Artifacts written
-----------------
    models/emotion_cnn_fer2013.keras   best model
    models/emotion_cnn_fer2013.h5      HDF5 copy
    models/emotion_labels.json         class-index -> name map
    results/history.json               training curves data
    results/training_curves.png        accuracy / loss plot
    results/train_log.csv              per-epoch log
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config
from src.dataset import compute_class_weights, class_distribution, load_fer2013_csv
from src.model import build_cnn
from src.utils import plot_history, save_labels, set_seed


class TimeStopping:
    """Keras callback that stops training once a wall-clock budget is exceeded."""

    def __init__(self, minutes: float):
        self.minutes = minutes
        self.start = None

    def _as_keras(self):
        from tensorflow import keras

        outer = self

        class _CB(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                outer.start = time.time()

            def on_epoch_end(self, epoch, logs=None):
                elapsed = (time.time() - outer.start) / 60.0
                if logs is not None:
                    logs["elapsed_min"] = round(elapsed, 2)
                if elapsed >= outer.minutes:
                    print(f"\n[TimeStopping] budget of {outer.minutes} min reached "
                          f"({elapsed:.1f} min) - stopping.")
                    self.model.stop_training = True

        return _CB()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the FER-2013 emotion CNN")
    parser.add_argument("--csv", default=config.CSV_PATH, help="path to icml_face_data.csv")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--minutes", type=float, default=config.MAX_TRAINING_MINUTES,
                        help="wall-clock training budget in minutes")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    from tensorflow import keras

    print("=" * 70)
    print("Emotion Detection from Facial Expressions - CNN training (FER-2013)")
    print("=" * 70)

    print("\n[1/5] Loading FER-2013 ...")
    (X_train, y_train), (X_test, y_test) = load_fer2013_csv(args.csv)
    print(f"      train={X_train.shape}  test={X_test.shape}")
    print(f"      train dist: {class_distribution(y_train)}")

    # Carve a validation split out of the training data.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]
    n_val = int(len(X_train) * config.VALIDATION_SPLIT)
    X_val, y_val = X_train[:n_val], y_train[:n_val]
    X_tr, y_tr = X_train[n_val:], y_train[n_val:]
    print(f"      fit={X_tr.shape[0]}  val={X_val.shape[0]}")

    print("\n[2/5] Building CNN ...")
    model = build_cnn((config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS))
    model.summary()

    class_weight = None
    if not args.no_class_weights:
        class_weight = compute_class_weights(y_tr)
        print(f"      class weights: { {config.EMOTIONS[k]: round(v, 2) for k, v in class_weight.items()} }")

    ckpt = keras.callbacks.ModelCheckpoint(
        config.MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    )
    callbacks = [
        ckpt,
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=4, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(config.RESULTS_DIR, "train_log.csv")),
        TimeStopping(args.minutes)._as_keras(),
    ]

    print(f"\n[3/5] Training (max {args.epochs} epochs, budget {args.minutes} min) ...")
    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = (time.time() - t0) / 60.0
    print(f"      training finished in {elapsed:.1f} min")

    # Restore best weights and persist.
    try:
        best = keras.models.load_model(config.MODEL_PATH)
        model.set_weights(best.get_weights())
    except Exception as exc:  # pragma: no cover
        print(f"      (could not reload best checkpoint: {exc})")

    print("\n[4/5] Saving artifacts ...")
    model.save(config.MODEL_PATH)
    try:
        model.save(config.MODEL_H5_PATH)
    except Exception as exc:  # pragma: no cover
        print(f"      (.h5 export skipped: {exc})")
    save_labels(config.LABELS_PATH, config.EMOTIONS)

    hist = {k: [float(x) for x in v] for k, v in history.history.items()
            if not k.startswith("elapsed")}
    with open(config.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    plot_history(hist, os.path.join(config.RESULTS_DIR, "training_curves.png"))

    print("\n[5/5] Quick test-set evaluation ...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"      test loss     : {test_loss:.4f}")
    print(f"      test accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")

    summary = {
        "epochs_ran": len(history.history.get("loss", [])),
        "training_minutes": round(elapsed, 2),
        "best_val_accuracy": round(float(max(history.history.get("val_accuracy", [0]))), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "num_params": int(model.count_params()),
        "dataset": "FER-2013 (msambare/fer2013)",
    }
    with open(os.path.join(config.RESULTS_DIR, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:", json.dumps(summary, indent=2))
    print("\nDone. Model ->", config.MODEL_PATH)
    return summary


if __name__ == "__main__":
    main()
