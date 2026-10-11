"""Generate real prediction-output artifacts for the documentation.

Runs the trained model over the bundled example faces and saves:

    results/output_examples/<name>_pred.png   annotated image (face box + label)
    results/prediction_output_examples.txt    CLI-style text output for each image

This produces the "actual output / prediction examples" evidence for the
project report.  Run after training:

    python scripts/make_output_examples.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src import config
from src.model import load_trained_model
from src.predict import predict_array


def main():
    out_dir = os.path.join(config.RESULTS_DIR, "output_examples")
    os.makedirs(out_dir, exist_ok=True)

    model = load_trained_model()
    samples = sorted(
        f for f in os.listdir(config.SAMPLES_DIR) if f.lower().endswith(".png")
    )

    lines = [
        "Actual model output on example faces (data/samples)",
        "=" * 55,
        "",
    ]

    for name in samples:
        path = os.path.join(config.SAMPLES_DIR, name)
        img = cv2.imread(path)
        if img is None:
            continue

        result = predict_array(model, img)
        true_label = name.replace("sample_", "").replace(".png", "")

        annotated = img.copy()
        if result["face_box"]:
            x, y, w, h = result["face_box"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(annotated, f"{result['emotion']} {result['confidence'] * 100:.1f}%",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        out_path = os.path.join(out_dir, f"{name[:-4]}_pred.png")
        cv2.imwrite(out_path, annotated)

        lines.append(f"file            : {name}  (true label: {true_label})")
        lines.append(f"predicted emotion: {result['emotion']}")
        lines.append(f"confidence       : {result['confidence'] * 100:.2f}%")
        lines.append("probabilities:")
        for e, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
            bar = "#" * int(round(p * 40))
            lines.append(f"  {e:>9}: {p * 100:6.2f}%  {bar}")
        lines.append("-" * 55)

    txt_path = os.path.join(config.RESULTS_DIR, "prediction_output_examples.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(samples)} annotated images -> {out_dir}")
    print("Wrote text output ->", txt_path)


if __name__ == "__main__":
    main()
