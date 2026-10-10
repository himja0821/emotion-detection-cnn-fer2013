# Emotion Detection from Facial Expressions using Convolutional Neural Networks

**Dataset:** FER-2013 (`msambare/fer2013`, Kaggle)
**Task:** 7-class facial expression classification
**Model:** Convolutional Neural Network (CNN) — TensorFlow / Keras
**Final test accuracy:** **63.78 %**

---

## 1. Problem Statement

Understanding human emotions from facial expressions is valuable in areas such
as customer-feedback analysis, mental-health monitoring, and human–computer
interaction. Manual analysis is subjective and does not scale. There is a need
for an automated system that classifies human emotions directly from facial
images.

## 2. Proposed Solution

A Convolutional Neural Network is trained on the **FER-2013** dataset, which
contains grayscale facial images labelled with seven emotions (*angry, disgust,
fear, happy, sad, surprise, neutral*). The CNN learns hierarchical facial
features through stacked convolutional and max-pooling layers, followed by
dense layers that classify the input face into one of the seven categories.
The model is evaluated using **accuracy** and a **confusion matrix** to analyse
per-class performance.

## 3. Dataset

| Property | Value |
|---|---|
| Name | FER-2013 (Facial Expression Recognition 2013) |
| Source | https://www.kaggle.com/datasets/msambare/fer2013 |
| Images | 35,887 grayscale, 48 × 48 px, faces roughly centred |
| Classes | 7 |
| Training set | 28,709 |
| Test set | 7,178 (PublicTest 3,589 + PrivateTest 3,589) |

**Training-set class distribution:** angry 3,995 · disgust 436 · fear 4,097 ·
happy 7,215 · sad 4,830 · surprise 3,171 · neutral 4,965.

The dataset is strongly imbalanced (disgust ≈ 1.5 % vs happy ≈ 25 %), which is
reflected in the per-class results.

## 4. Methodology

### 4.1 Preprocessing
- Pixels converted to `float32` and scaled to `[0, 1]`.
- Images reshaped to `(48, 48, 1)`.
- A 10 % validation split is carved out of the training set.

### 4.2 Data augmentation
Light geometric jitter is applied on-the-fly inside the model (rotation ±5 %,
translation ±5 %, zoom ±5 %). Horizontal flipping was deliberately **not** used
because facial expressions are asymmetric.

### 4.3 CNN architecture

```
Input (48×48×1)
 └─ Augmentation (rotation / translation / zoom)
 Block 1 : Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.2)
 Block 2 : Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.2)
 Block 3 : Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.3)
 Head    : Flatten → Dense(256) → BN → ReLU → Dropout(0.4) → Dense(7, softmax)
```

- **Parameters:** 1,470,247
- **Loss:** sparse categorical cross-entropy
- **Optimizer:** Adam (lr = 1e-3) with `ReduceLROnPlateau` (factor 0.5, patience 4)
- **Callbacks:** EarlyStopping (patience 12), ModelCheckpoint (best `val_accuracy`),
  CSVLogger and a wall-clock `TimeStopping` guard.

### 4.4 Training
Trained on CPU (TensorFlow 2.21 / Keras 3, oneDNN). 39 epochs in ≈ 40 minutes,
batch size 128.

## 5. Results

### 5.1 Overall accuracy

| Split | Accuracy |
|---|---|
| Validation (best) | 64.18 % |
| **Test (7,178 images)** | **63.78 %** |

### 5.2 Per-class performance (test set)

| Emotion | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| angry | 0.5504 | 0.5647 | 0.5574 | 958 |
| disgust | 0.7302 | 0.4144 | 0.5287 | 111 |
| fear | 0.4892 | 0.4189 | 0.4513 | 1024 |
| happy | 0.8350 | 0.8670 | **0.8507** | 1774 |
| sad | 0.5465 | 0.4050 | 0.4652 | 1247 |
| surprise | 0.7135 | 0.8002 | 0.7544 | 831 |
| neutral | 0.5485 | 0.6926 | 0.6122 | 1233 |
| **accuracy** | | | **0.6378** | 7178 |
| macro avg | 0.6305 | 0.5947 | 0.6029 | 7178 |
| weighted avg | 0.6327 | 0.6378 | 0.6305 | 7178 |

### 5.3 Confusion matrix (rows = true, columns = predicted)

```
             angry disgust  fear happy  sad surprise neutral
 angry   [   541     6     91    39   106    34     141  ]
 disgust [    38    46      6     1    10     4       6  ]
 fear    [   133     4    429    38   156   121     143  ]
 happy   [    46     0     40  1538    25    37      88  ]
 sad     [   145     4    168    86   505    41     298  ]
 surprise[    17     2     67    44     9   665      27  ]
 neutral [    63     1     76    96   113    30     854  ]
```

![confusion matrix](../results/confusion_matrix_normalized.png)

### 5.4 Observations
- **Happy** and **surprise** are recognised best — they have distinctive,
  high-contrast facial cues (smile, open mouth, raised brows).
- **Fear** and **sad** are the hardest; they are frequently confused with
  *neutral*, *angry* and each other, consistent with the literature on FER-2013.
- **Disgust** has high precision but low recall due to very few training
  examples (436).
- The dataset is imbalanced, so *weighted* average ≈ overall accuracy (~63 %).

## 6. Applications
Customer sentiment analysis, interactive/affective systems, driver-monitoring,
and mental-health monitoring tools.

## 7. Limitations & Future Work
- CPU-only training; a GPU and/or transfer learning (pre-trained face models)
  would push accuracy toward 70 %+.
- Small, imbalanced classes (disgust) limit per-class performance; oversampling
  or focal loss could help.
- FER-2013 images are pre-cropped; real-world use should combine the model with
  a face detector (the project ships one — see `src/face_detector.py`).

## 8. Conclusion
A CNN trained from scratch on FER-2013 reaches **63.78 % test accuracy**,
comfortably above the ~48 % that would be expected from always predicting the
majority class, and demonstrates the core techniques of deep learning for
affective computing: convolutional feature extraction, regularisation,
learning-rate scheduling, and rigorous evaluation via accuracy and a confusion
matrix.

## 9. How to Reproduce
See `README.md`. In short:

```powershell
python scripts\prepare_data.py      # verify dataset + build folders
python -m src.train                 # train the CNN
python -m src.evaluate              # accuracy + confusion matrix
python -m src.predict --image data\samples\sample_happy.png
python webapp\app.py                # web demo
```

## 10. References
1. Goodfellow et al. (2013). *Challenges in Representation Learning: FER-2013.*
2. Kaggle FER-2013 dataset — msambare/fer2013.
3. TensorFlow / Keras documentation.
