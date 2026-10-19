# Week 7 Facial Emotion Recognition

A Jupyter laboratory using FER2013, frozen MobileNetV2 embeddings, Logistic Regression,
and Linear SVM. Start with `notebooks/week7_fer.ipynb`. No performance is assumed before
the notebook runs on the real dataset.

## Setup

Python 3.11 is recommended. On Intel macOS, requirements select TensorFlow 2.16.2;
newer TensorFlow builds do not support this platform. Use a project-local environment:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3 --display-name 'Python 3 (Week 7 FER)'
python -m jupyterlab notebooks/week7_fer.ipynb
```

Select this environment as the notebook kernel. Track B uses OpenCV for detection;
MediaPipe is optional (`requirements-landmarks.txt`) for a future landmark extension.
The first CNN run downloads ImageNet weights. CPU execution is supported. Package
versions are saved in `artifacts/environment.json`; a fixed seed does not guarantee
bitwise-identical outputs across hardware and library versions.

## Dataset

The public [Kaggle FER2013 folder-format release](https://www.kaggle.com/datasets/msambare/fer2013)
matches the lab's `data/train/<class>` and `data/test/<class>` layout. Keep source metadata
and the archive hash in `data/source.json`. The archive is a repackaged dataset; this
lab's test protocol is not described as original competition PrivateTest-only evaluation.

To use a different location, set `FER_DATA_ROOT` or edit `DATA_ROOT` in the notebook.
Read `data/README.md`. The audit writes evidence before rejecting unreadable images,
wrong sizes, conflicting labels, or train/test duplicates in strict mode. The notebook uses
an explicit alternative: report duplicates, keep the complete supplied test set, exclude
training images identical to test images, exclude conflicting training-label groups, and
keep one copy of other repeated training images. All exclusions and original/effective
counts are saved. Test-only duplicates and conflicting labels remain a reported limitation.
Source images are never modified. Exact matching does not detect all near duplicates.

## Workflow

1. Verify seven classes, counts, dimensions, and decoded-image hashes.
2. Inspect actual Haar detections on multiple training samples per class.
3. Extract 1,280-dimensional MobileNetV2 embeddings in batches; cache by source content,
   preprocessing settings, model weights, and runtime versions.
4. Make a stratified validation partition from training data, keeping identical-image groups
   together. Train `StandardScaler → LogisticRegression` and `StandardScaler → LinearSVC`.
5. Select by validation macro-F1, then refit both on the full training split. Evaluate once
   on the fixed test split. A convergence warning stops training instead of being hidden.
6. Save a complete classifier bundle, reload it, and demonstrate two deterministic test
   examples. Prediction labels must agree with evaluation.

Detection failures on already cropped FER-style images use the full image with an explicit
fallback flag. Never count these as successful detections. For a general photograph,
use strict detection mode. No confidence percentage is fabricated from SVM margins.

## Run and inspect

```sh
source .venv/bin/activate
python scripts/run_notebook.py
python -m src.predict path/to/face.png
python -m src.predict path/to/photo.jpg --strict-detection
MPLBACKEND=Agg MPLCONFIGDIR=.cache/matplotlib python -m unittest discover -s tests -v
```

`scripts/run_notebook.py` writes `outputs/week7_fer_executed.ipynb`, including diagnostic
outputs if a cell fails. Check the final notebook for errors before claiming a completed run.
The model must exist before invoking prediction. Only load trusted local joblib artifacts.

Outputs include:

- `artifacts/audit/`: manifest, original counts, duplicate records, and audit summary.
- `artifacts/features/`: cached embeddings and detection records.
- `artifacts/models/`: fitted pipelines, validation partition, selection evidence, deployment bundle.
- `outputs/metrics/`: test metrics, full reports, confusion matrices, predictions, detection rates.
- `outputs/figures/`: sample/detection grids, errors, and prediction proof.
- `outputs/submission/`: two-image prediction table.

Use `submission_checklist.md` to complete the supplied Word form. Personal details, team
roles, and reflection must be completed truthfully. The rubric adds to 110 despite its
100-point title. The notebook implements one of the two allowed tracks, not both.

## Technical references

- [Keras MobileNetV2](https://keras.io/2/api/applications/mobilenet/)
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
- [TensorFlow installation](https://www.tensorflow.org/install/pip)
