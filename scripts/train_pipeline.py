"""End-to-end execution of the Week 7 FER pipeline:
audit, detection check, feature extraction, training, evaluation, demo, and summary report.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Must set cache directories BEFORE importing matplotlib, tensorflow, or keras
os.environ['KERAS_HOME'] = str(ROOT / '.cache/keras')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

(ROOT / '.cache/keras/models').mkdir(parents=True, exist_ok=True)
(ROOT / '.cache/matplotlib').mkdir(parents=True, exist_ok=True)

import json
import random
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data import CLASSES, scan_dataset, audit_dataset, validation_partition, prepare_training_data
from src.features import FeatureConfig, FacePreprocessor, FeatureExtractor
from src.training import train_models, evaluate_models
from src.predict import save_bundle, EmotionPredictor, demonstrate
from src.plots import sample_grid, error_grid
from src.reporting import write_summary


import argparse

def main():
    parser = argparse.ArgumentParser(description='Week 7 FER End-to-End Pipeline')
    parser.add_argument('--sample-limit', type=int, default=None,
                        help='Limit number of samples per split for quick verification.')
    args = parser.parse_args()

    print("=== Week 7 Facial Emotion Recognition Pipeline ===")
    DATA_ROOT = Path(os.environ.get('FER_DATA_ROOT', ROOT / 'data')).expanduser()
    SEED = 42
    BATCH_SIZE = 32
    ARTIFACTS = ROOT / 'artifacts'
    OUTPUTS = ROOT / 'outputs'

    for folder in [ARTIFACTS / 'audit', ARTIFACTS / 'features', ARTIFACTS / 'models',
                   OUTPUTS / 'figures', OUTPUTS / 'metrics', OUTPUTS / 'submission']:
        folder.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    # Step 1: Environment log
    print("\n[1/7] Logging environment and software versions...")
    versions = {p: importlib.metadata.version(p)
                for p in ['numpy', 'pandas', 'scikit-learn', 'opencv-python', 'tensorflow', 'keras', 'matplotlib', 'joblib']}
    versions['python'] = sys.version
    (ARTIFACTS / 'environment.json').write_text(json.dumps(versions, indent=2))
    print(f"Dataset root: {DATA_ROOT}")
    print(f"TensorFlow devices: {tf.config.list_physical_devices()}")

    # Step 2: Dataset scan, audit, and deduplication
    print("\n[2/7] Scanning and auditing FER2013 dataset...")
    manifest = scan_dataset(DATA_ROOT)
    counts, audit = audit_dataset(manifest, ARTIFACTS / 'audit', duplicate_policy='report')
    print("Class counts (raw):")
    print(counts)
    print("Audit summary:", audit)

    train_rows, test_rows, effective_counts = prepare_training_data(manifest, ARTIFACTS / 'audit')
    print("Effective counts after deduplication policy:")
    print(effective_counts)

    if args.sample_limit:
        per_class = max(5, args.sample_limit // len(CLASSES))
        print(f"\n[NOTE] --sample-limit={args.sample_limit} specified: slicing {per_class} samples per class for test run.")
        train_rows = train_rows.groupby('label').head(per_class).reset_index(drop=True)
        test_rows = test_rows.groupby('label').head(per_class).reset_index(drop=True)
        print(f"Verification train rows: {len(train_rows)} | test rows: {len(test_rows)}")

    # Class sample grid
    fig = sample_grid(train_rows, per_class=3, seed=SEED)
    fig.savefig(OUTPUTS / 'figures/class_samples.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved sample grid -> outputs/figures/class_samples.png")

    # Step 3: Milestone 1 Face Detection & Visual Check
    print("\n[3/7] Milestone 1: Face detection and visual check...")
    config = FeatureConfig()
    preprocessor = FacePreprocessor(config)
    fig = sample_grid(train_rows, preprocessor=preprocessor, per_class=3, seed=SEED)
    fig.savefig(OUTPUTS / 'figures/face_detection.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved detection check -> outputs/figures/face_detection.png")

    # Step 4: Milestone 2 CNN Feature Extraction
    print("\n[4/7] Milestone 2: Feature extraction using MobileNetV2...")
    extractor = FeatureExtractor(config)
    print(f"Extracting training embeddings ({len(train_rows)} images)...")
    X_train, train_detection = extractor.extract(train_rows, ARTIFACTS / 'features', BATCH_SIZE)
    print(f"Extracting test embeddings ({len(test_rows)} images)...")
    X_test, test_detection = extractor.extract(test_rows, ARTIFACTS / 'features', BATCH_SIZE)
    print(f"Extracted train features: {X_train.shape} | Test features: {X_test.shape}")

    detections = pd.concat([train_detection, test_detection], ignore_index=True)
    detections.to_csv(OUTPUTS / 'metrics/detections.csv', index=False)
    detection_summary = detections.groupby(['split', 'label']).agg(
        images=('detected', 'size'), detected=('detected', 'sum'), fallbacks=('fallback', 'sum'))
    detection_summary['detection_rate'] = detection_summary.detected / detection_summary.images
    detection_summary.to_csv(OUTPUTS / 'metrics/detection_summary.csv')
    print("Detection summary:")
    print(detection_summary)

    # Step 5: Model Training and Validation-Based Selection
    print("\n[5/7] Training classical classifiers (Logistic Regression & Linear SVM)...")
    fit_idx, val_idx = validation_partition(train_rows, seed=SEED)
    print(f"Partition: {len(fit_idx)} fit images, {len(val_idx)} validation images.")
    models, selected_name, validation_scores = train_models(
        X_train, train_rows, fit_idx, val_idx, ARTIFACTS / 'models', seed=SEED)
    print("Validation metrics:")
    print(validation_scores)
    print(f"Selected model by validation macro-F1: {selected_name}")

    save_bundle(ARTIFACTS / 'models/emotion_pipeline.joblib', models[selected_name], extractor, selected_name)
    print("Saved deployment bundle -> artifacts/models/emotion_pipeline.joblib")

    # Step 6: Milestone 3 Test Evaluation
    print("\n[6/7] Milestone 3: Evaluating models on unchanged test split...")
    test_scores, test_predictions = evaluate_models(models, X_test, test_rows, OUTPUTS / 'metrics')
    print("Final Test Metrics:")
    print(test_scores)

    for name in models:
        print(f"\n--- {name} Classification Report ---")
        print((OUTPUTS / f'metrics/{name}_classification_report.txt').read_text())
        fig = error_grid(test_predictions, name)
        if fig is not None:
            fig.savefig(OUTPUTS / f'figures/{name}_errors.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

    # Step 7: Prediction Proof and Comparative Summary
    print("\n[7/7] Deployment proof on held-out test images & summary report...")
    predictor = EmotionPredictor(ARTIFACTS / 'models/emotion_pipeline.joblib', extractor=extractor)
    prediction_demo = demonstrate(predictor, test_rows, test_predictions, OUTPUTS / 'submission')
    print("Prediction demo on held-out images:")
    print(prediction_demo[['filename', 'predicted_label', 'true_label', 'detected', 'fallback', 'source']])

    fig, axes = plt.subplots(1, len(prediction_demo), figsize=(10, 4), squeeze=False)
    for ax, row in zip(axes.flat, prediction_demo.itertuples()):
        ax.imshow(cv2.imread(row.filename, cv2.IMREAD_GRAYSCALE), cmap='gray')
        ax.set_title(f"{Path(row.filename).name}\nPredicted: {row.predicted_label} | True: {row.true_label}", fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUTPUTS / 'figures/prediction_demo.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    summary_path = write_summary(test_scores, selected_name, OUTPUTS, ARTIFACTS / 'audit')
    print(f"Generated comparative summary -> {summary_path}")
    print("\n=== Pipeline Execution Successfully Completed! ===")


if __name__ == '__main__':
    import cv2
    main()
