"""Save and load a complete expression-classification pipeline."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

from .data import CLASSES
from .features import FeatureConfig, FeatureExtractor


def save_bundle(path, classifier, extractor, selected_name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'classifier': classifier, 'config': asdict(extractor.config),
                 'runtime': extractor.runtime, 'classes': list(CLASSES),
                 'selected_model': selected_name, 'schema_version': 1}, path)


class EmotionPredictor:
    def __init__(self, bundle_path, extractor=None):
        # joblib is for trusted local artifacts, not downloaded unknown models.
        bundle = joblib.load(bundle_path)
        if bundle.get('schema_version') != 1 or bundle['classes'] != list(CLASSES):
            raise ValueError('Unsupported model bundle.')
        self.extractor = extractor or FeatureExtractor(FeatureConfig(**bundle['config']))
        if asdict(self.extractor.config) != bundle['config'] or self.extractor.runtime != bundle['runtime']:
            raise ValueError('Extractor settings, weights, or runtime differ from training. Use the saved environment.')
        self.classifier = bundle['classifier']
        self.model_name = bundle['selected_model']

    def predict_emotion(self, image_path, fer_style=True):
        image, detection = self.extractor.preprocessor.prepare(image_path, fer_style=fer_style)
        features = self.extractor.embed([image])
        label = str(self.classifier.predict(features)[0])
        return {'filename': str(image_path), 'predicted_label': label,
                'model': self.model_name, **detection}


def demonstrate(predictor, rows, expected_predictions, output_dir):
    """Use two deterministic test samples, not cherry-picked correct examples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(rows) < 2 or set(rows.split) != {'test'}:
        raise ValueError('Demonstration needs at least two held-out test images.')
    chosen = rows.groupby('label', sort=True).head(1).head(2)
    if len(chosen) < 2:
        chosen = rows.head(2)
    results = []
    for row in chosen.itertuples():
        result = predictor.predict_emotion(row.path)
        expected = expected_predictions.loc[expected_predictions.sample_id == row.sample_id,
                                            predictor.model_name].iloc[0]
        if result['predicted_label'] != expected:
            raise AssertionError('Reloaded prediction differs from evaluation pipeline.')
        results.append(dict(**result, true_label=row.label, source='held-out test image'))
    table = pd.DataFrame(results)
    table.to_csv(output_dir / 'prediction_demo.csv', index=False)
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict a FER expression label with a trusted trained bundle.')
    parser.add_argument('image')
    parser.add_argument('--model', default='artifacts/models/emotion_pipeline.joblib')
    parser.add_argument('--strict-detection', action='store_true', help='Reject images when no face is detected.')
    args = parser.parse_args()
    predictor = EmotionPredictor(args.model)
    print(json.dumps(predictor.predict_emotion(args.image, fer_style=not args.strict_detection), indent=2))
