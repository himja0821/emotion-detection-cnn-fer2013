# Deep Learning Documentation Checklist — Evidence Map

Project: **Emotion Detection from Facial Expressions using CNNs (FER-2013)**
This file maps every mandatory checklist item to where it is covered.

| # | Checklist item | Where it is covered | Status |
|---|---|---|---|
| 1 | Dataset name, source URL, number of samples, classes, description | `docs/report.md` §3; `README.md`; `src/config.py` | ✅ |
| 2 | Data cleaning, preprocessing, normalization, augmentation, class balancing | `docs/report.md` §4.1–4.4; code: `src/dataset.py` (`load_fer2013_csv`, `compute_class_weights`), `src/model.py` (augmentation layers) | ✅ |
| 3 | Train/validation/test split and prevention of data leakage | `docs/report.md` §4.5; code: `src/train.py` (seeded split), `src/evaluate.py` (held-out test) | ✅ |
| 4 | Model architecture diagram and explanation of layers | `docs/architecture.png` + `docs/report.md` §5.1–5.2; generator `scripts/make_architecture_diagram.py` | ✅ |
| 5 | Model selection rationale and baseline/alternative comparison | `docs/report.md` §5.3 (baseline & alternatives table) | ✅ |
| 6 | Hyperparameters: learning rate, batch size, epochs, optimizer, loss | `docs/report.md` §5.4; `src/config.py`; `src/model.py` (`compile`) | ✅ |
| 7 | Training and validation accuracy/loss graphs | `results/training_curves.png`; data `results/history.json`; `docs/report.md` §7.4 | ✅ |
| 8 | Suitable task-specific evaluation metrics | `docs/report.md` §7.2; `results/classification_report.txt`; `results/metrics.json` | ✅ |
| 9 | Confusion matrix | `results/confusion_matrix.png`, `results/confusion_matrix_normalized.png`; `docs/report.md` §7.3 | ✅ |
| 10 | Actual output screenshots, prediction examples, and error analysis | `results/output_examples/*.png`, `results/prediction_output_examples.txt`, `results/sample_predictions.png`; `docs/report.md` §8 (error analysis) | ✅ |
| 11 | Hardware/software requirements and reproducibility details | `docs/report.md` §6 and §11; `README.md` (Setup); `requirements.txt` | ✅ |
| 12 | Declaration of originality and acknowledgement of external resources | `docs/DECLARATION.md` | ✅ |

## Supporting artifacts

| Artifact | Path |
|---|---|
| Full project report | `docs/report.md` |
| Architecture diagram | `docs/architecture.png` |
| Declaration & acknowledgements | `docs/DECLARATION.md` |
| Trained model | `models/emotion_cnn_fer2013.keras` |
| Evaluation metrics | `results/metrics.json` |
| Per-class report | `results/classification_report.txt` |
| Confusion matrices | `results/confusion_matrix.png`, `results/confusion_matrix_normalized.png` |
| Per-class accuracy | `results/per_class_accuracy.png` |
| Training curves | `results/training_curves.png` |
| Sample predictions | `results/sample_predictions.png` |
| Prediction output examples | `results/prediction_output_examples.txt`, `results/output_examples/` |
| Training log | `results/train_log.csv`, `results/train_summary.json` |

## How to regenerate the evidence

```powershell
python scripts\prepare_data.py            # dataset verify + folders
python -m src.train                        # train (writes curves, model, logs)
python -m src.evaluate                     # metrics + confusion matrices
python scripts\make_architecture_diagram.py  # architecture.png
python scripts\make_output_examples.py     # prediction output examples
```
