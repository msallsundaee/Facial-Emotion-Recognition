"""Small synthetic checks for data integrity and pipeline behavior, not FER accuracy."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from src.data import CLASSES, scan_dataset, audit_dataset, validation_partition, prepare_training_data
from src.features import FacePreprocessor
from src.training import train_models, evaluate_models


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self):
        rng = np.random.default_rng(42)
        for split in ('train', 'test'):
            for label in CLASSES:
                folder = self.root / 'data' / split / label
                folder.mkdir(parents=True)
                for i in range(5):
                    cv2.imwrite(str(folder / f'{i}.png'), rng.integers(0, 256, (48, 48), dtype=np.uint8))
        return scan_dataset(self.root / 'data')

    def test_cross_split_duplicates_stop_audit_and_leave_evidence(self):
        rows = self.dataset()
        source = self.root / 'data/train/angry/0.png'
        destination = self.root / 'data/test/angry/0.png'
        destination.write_bytes(source.read_bytes())
        rows = scan_dataset(self.root / 'data')
        with self.assertRaisesRegex(ValueError, 'across train/test'):
            audit_dataset(rows, self.root / 'audit')
        self.assertTrue((self.root / 'audit/duplicates.csv').exists())

    def test_duplicate_groups_stay_in_one_validation_partition(self):
        rows = self.dataset().query("split == 'train'").reset_index(drop=True)
        rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
        fit, val = validation_partition(rows)
        self.assertFalse(set(rows.iloc[fit].pixel_sha256) & set(rows.iloc[val].pixel_sha256))
        self.assertEqual(set(fit) | set(val), set(range(len(rows))))

    def test_filtering_preserves_test_and_documents_training_exclusions(self):
        rows = self.dataset()
        train = rows.query("split == 'train'")
        test = rows.query("split == 'test'")
        # Insert overlap, a conflicting training copy, and a same-label training copy.
        overlap = train.iloc[[0]].copy()
        overlap['pixel_sha256'] = test.iloc[0].pixel_sha256
        conflict = train.iloc[[1]].copy()
        conflict['label'] = 'sad'
        repeat = train.iloc[[2]].copy()
        augmented = pd.concat([rows, overlap, conflict, repeat], ignore_index=True)
        kept, actual_test, counts = prepare_training_data(augmented, self.root / 'policy')
        pd.testing.assert_frame_equal(actual_test, test.reset_index(drop=True))
        self.assertEqual(len(kept), len(train) - 1)
        self.assertFalse(kept.pixel_sha256.duplicated().any())
        self.assertFalse(set(kept.pixel_sha256) & set(actual_test.pixel_sha256))
        exclusions = pd.read_csv(self.root / 'policy/training_exclusions.csv')
        self.assertEqual(set(exclusions.exclusion_reason),
                         {'exact_test_overlap','conflicting_training_labels','repeated_training_image'})

    def test_no_face_is_explicit_and_strict_mode_rejects_it(self):
        path = self.root / 'blank.png'
        cv2.imwrite(str(path), np.zeros((48, 48), dtype=np.uint8))
        prep = FacePreprocessor()
        image, info = prep.prepare(path)
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertFalse(info['detected'])
        self.assertTrue(info['fallback'])
        self.assertIsNone(info['box'])
        with self.assertRaisesRegex(ValueError, 'No face detected'):
            prep.prepare(path, fer_style=False)

    def test_fit_evaluate_and_reload_on_synthetic_features(self):
        import joblib
        rows = self.dataset()
        train = rows.query("split == 'train'").reset_index(drop=True)
        test = rows.query("split == 'test'").reset_index(drop=True)
        # Distinct encoded classes make this a control-flow check, not an accuracy experiment.
        X = np.eye(7)[[CLASSES.index(label) for label in train.label]]
        Xt = np.eye(7)[[CLASSES.index(label) for label in test.label]]
        fit, val = validation_partition(train)
        models, selected, _ = train_models(X, train, fit, val, self.root / 'models')
        scores, pred = evaluate_models(models, Xt, test, self.root / 'metrics')
        self.assertEqual(len(scores), 2)
        self.assertTrue((scores.macro_f1 == 1).all())
        reloaded = joblib.load(self.root / 'models' / f'{selected}.joblib')
        np.testing.assert_array_equal(reloaded.predict(Xt), pred[selected])
        cm = pd.read_csv(self.root / 'metrics' / f'{selected}_confusion_matrix.csv', index_col=0)
        self.assertEqual(cm.shape, (7, 7))

    def test_training_rejects_test_rows(self):
        rows = self.dataset().query("split == 'test'").reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, 'only training-split'):
            train_models(np.ones((len(rows), 2)), rows, [], [], self.root / 'models')


if __name__ == '__main__':
    unittest.main()
