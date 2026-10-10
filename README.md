# Emotion Detection from Facial Expressions using Convolutional Neural Networks

A complete deep-learning project that classifies human emotions from facial
images using a Convolutional Neural Network (CNN) trained on the **FER-2013**
dataset (7 classes: angry, disgust, fear, happy, sad, surprise, neutral).

> **Dataset:** [FER-2013 — `msambare/fer2013`](https://www.kaggle.com/datasets/msambare/fer2013)
> (35,887 grayscale 48x48 face images).

## Results

| Metric | Value |
|---|---|
| Test accuracy | **63.78 %** |
| Best validation accuracy | 64.18 % |
| Parameters | 1,470,247 |
| Training time (CPU) | ~40 min |

Per-class highlights: **happy 85 % F1**, **surprise 75 % F1**, **neutral 61 % F1**.
See `results/classification_report.txt` and `results/confusion_matrix.png`.

## Project structure

```
Emotion Detection Project - DEEP LEARNING/
├── data/
│   ├── fer2013/                # icml_face_data.csv + train/ test/ class folders
│   └── samples/                # example faces (one per class)
├── src/
│   ├── config.py               # paths + hyper-parameters
│   ├── dataset.py              # FER-2013 loading / preprocessing / folder export
│   ├── model.py                # CNN architecture
│   ├── train.py                # training (with time budget + callbacks)
│   ├── evaluate.py             # accuracy, classification report, confusion matrix
│   ├── predict.py              # image / webcam inference
│   └── face_detector.py        # OpenCV DNN face detector
├── webapp/                     # Flask demo (upload / webcam)
├── notebooks/                  # executed walkthrough notebook
├── scripts/                    # prepare_data.py, make_samples.py
├── tests/                      # pytest suite
├── models/                     # trained model (.keras / .h5)
├── results/                    # metrics + plots
└── docs/report.md              # full project report
```

## Setup

```powershell
# 1. create a virtual environment (Python 3.13)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Get the dataset

Place the FER-2013 CSV at `data/fer2013/icml_face_data.csv` (same data as the
Kaggle `msambare/fer2013` release), then:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

This verifies the dataset and exports the msambare-style `train/` and `test/`
class folders.

## Train

```powershell
.\.venv\Scripts\python.exe -m src.train --epochs 100 --minutes 60
```

Artifacts: `models/emotion_cnn_fer2013.keras`, `results/training_curves.png`,
`results/train_summary.json`.

## Evaluate

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```

Produces `confusion_matrix.png`, `confusion_matrix_normalized.png`,
`classification_report.txt`, `per_class_accuracy.png`, `metrics.json`.

## Predict

```powershell
# single image
.\.venv\Scripts\python.exe -m src.predict --image data\samples\sample_happy.png
# live webcam (press q to quit)
.\.venv\Scripts\python.exe -m src.predict --webcam
```

### How to test (step by step)

1. Open **File Explorer** and go to the project folder:
   ```
   H:\Emotion Detection Project - DEEP LEARNING
   ```
2. Click the **address bar**, type `powershell` and press **Enter** (a window opens inside the folder).
3. Type one of the commands below and press **Enter**:

   **See your own face (webcam):**
   ```powershell
   .\.venv\Scripts\python.exe -m src.predict --webcam
   ```
   -> wait ~10–20 s, a camera window shows your face with a green box and the detected emotion + %.

   **Test with a saved photo:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.predict --image data\samples\sample_happy.png
   ```
   -> prints `Predicted emotion : happy ...` with a probability bar chart.

4. To close the webcam window, click it and press **q**.

**View the results without any code:** open the `results` folder and
double-click `confusion_matrix.png`, `training_curves.png` and
`sample_predictions.png`.

## Web demo

```powershell
.\.venv\Scripts\python.exe webapp\app.py
# open http://127.0.0.1:5000
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Tech stack

Python 3.13 · TensorFlow 2.21 / Keras 3 · OpenCV · NumPy · scikit-learn ·
Matplotlib / Seaborn · Flask · Jupyter.
