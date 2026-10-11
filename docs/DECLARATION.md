# Declaration of Originality and Acknowledgements

## Declaration of Originality

**Project:** Emotion Detection from Facial Expressions using Convolutional
Neural Networks (FER-2013)

**Student:** HIMAJA VUCHURU (HIMAJA V)
**Roll Number (URN):** 2401100CSE0115
**Class / Branch:** CSE — Section S2, Year II / Semester IV
**Institution:** Sreenidhi University, Hyderabad
**Course:** Deep Learning

I hereby declare that this project, *Emotion Detection from Facial Expressions
using Convolutional Neural Networks*, is my own original work carried out as
part of the Deep Learning course. All source code in this repository (`src/`,
`webapp/`, `scripts/`, `tests/`, `notebooks/`) was written by me, and the
training, evaluation, and results reported here (`results/`) were produced by
running this code on the FER-2013 dataset.

I further declare that:

- This work has **not been copied** from any other student or source, and has
  not been submitted elsewhere for assessment.
- All external data, libraries, models, and references that were used are
  **explicitly acknowledged** in the section below and cited in
  `docs/report.md` (References).
- Figures and result artifacts are generated directly by the scripts in this
  repository, except where an external resource is noted.

**Signature:** _______________________  **Date:** ______________

---

## Acknowledgement of External Resources

This project stands on the shoulders of open-source software and publicly
available datasets. The following external resources were used and are
gratefully acknowledged.

### 1. Dataset

| Resource | Source | License / Note |
|---|---|---|
| FER-2013 (Facial Expression Recognition 2013) | [Kaggle — `msambare/fer2013`](https://www.kaggle.com/datasets/msambare/fer2013) | Public dataset for research and education |

- Original data collection and benchmark introduced by **Goodfellow et al.
  (2013)**, *Challenges in Representation Learning: A report on three machine
  learning contests* (ICML 2013 Workshop).
- The dataset used in this repository is the `msambare/fer2013` Kaggle
  re-packaging (grayscale 48 × 48 images, 7 emotion classes).
- The dataset is **not** redistributed here (it exceeds GitHub's 100 MB
  file-size limit); the code in `scripts/prepare_data.py` regenerates the
  `train/` and `test/` folders from the CSV provided by the user.

### 2. Software libraries

| Library | Purpose | License |
|---|---|---|
| TensorFlow / Keras | CNN model, training | Apache 2.0 |
| OpenCV (`opencv-python`) | Image I/O, DNN face detector | Apache 2.0 |
| NumPy | Numerical arrays | BSD-3-Clause |
| scikit-learn | Metrics (classification report, confusion matrix) | BSD-3-Clause |
| Matplotlib / Seaborn | Plots and figures | PSF-based / BSD-3-Clause |
| pandas, SciPy | Data handling | BSD-3-Clause |
| h5py | Model serialization | BSD-3-Clause |
| Flask | Web demo | BSD-3-Clause |

### 3. Pretrained components

| Resource | Source | Use |
|---|---|---|
| OpenCV DNN SSD face detector (res10, 300×300) | [OpenCV `face_detector` samples](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) | Face localisation before emotion classification (`src/face_detector.py`) |

This face detector is a **pretrained external component** and is used only to
crop faces from full images/webcam frames; the emotion classifier itself is
trained from scratch in this project.

### 4. Learning references

- Goodfellow, Bengio, Courville — *Deep Learning* (MIT Press, 2016).
- TensorFlow / Keras official documentation and tutorials.
- Scikit-learn user guide (metrics).

### 5. AI assistance

Parts of the code, documentation, and figures were developed with the help of
an AI coding assistant. All generated content was reviewed, tested, and
validated by me before inclusion, and the design decisions and results are my
own responsibility.

---

## Data & Ethics Statement

- FER-2013 contains human faces and is provided for research/educational use.
  It is used here solely for academic coursework.
- No new personal data was collected; the webcam demo (`src/predict.py
  --webcam`) processes frames locally in real time and does **not** store or
  transmit any images.
- Facial-expression recognition can be sensitive; this project is intended for
  learning purposes only and is not suitable for surveillance or
  decision-making about individuals.
