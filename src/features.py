import os
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

root_dir = Path(__file__).resolve().parents[1]
os.environ.setdefault('KERAS_HOME', str(root_dir / '.cache' / 'keras'))
os.environ.setdefault('MPLCONFIGDIR', str(root_dir / '.cache' / 'matplotlib'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    image_size: int = 224
    detector_size: int = 192
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_face_size: int = 30
    backbone: str = 'MobileNetV2'
    weights: str = 'imagenet'
    pooling: str = 'avg'
    preprocessing_version: int = 1


class FacePreprocessor:
    def __init__(self, config=None):
        self.config = config or FeatureConfig()
        cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
        self.cascade_sha256 = hashlib.sha256(cascade_path.read_bytes()).hexdigest()
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError('OpenCV Haar cascade failed to load.')

    def prepare(self, path, fer_style=True):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f'Cannot read image: {path}')
        h, w = gray.shape
        factor = self.config.detector_size / max(h, w)
        enlarged = cv2.resize(gray, (max(1, round(w * factor)), max(1, round(h * factor))))
        boxes = self.detector.detectMultiScale(enlarged, scaleFactor=self.config.scale_factor,
                                              minNeighbors=self.config.min_neighbors,
                                              minSize=(self.config.min_face_size,) * 2)
        metadata = {'detected': len(boxes) > 0, 'num_faces': len(boxes),
                    'fallback': False, 'box': None}
        if len(boxes):
            x, y, bw, bh = max(boxes, key=lambda b: int(b[2]) * int(b[3]))
            x0, y0 = max(0, int(x / factor)), max(0, int(y / factor))
            x1, y1 = min(w, int(np.ceil((x + bw) / factor))), min(h, int(np.ceil((y + bh) / factor)))
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f'Invalid detected crop for {path}')
            crop = gray[y0:y1, x0:x1]
            metadata['box'] = [x0, y0, x1 - x0, y1 - y0]
        elif fer_style:
            crop = gray
            metadata['fallback'] = True
        else:
            raise ValueError('No face detected. Use fer_style=True only for an already cropped face image.')
        resized = cv2.resize(crop, (self.config.image_size,) * 2, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32)
        return rgb, metadata


class FeatureExtractor:
    def __init__(self, config=None):
        import tensorflow as tf
        self.config = config or FeatureConfig()
        if (self.config.backbone, self.config.weights, self.config.pooling) != ('MobileNetV2', 'imagenet', 'avg'):
            raise ValueError('This lab supports frozen ImageNet MobileNetV2 with average pooling only.')
        self.preprocessor = FacePreprocessor(self.config)
        self.model = tf.keras.applications.MobileNetV2(
            weights='imagenet', include_top=False, pooling='avg',
            input_shape=(self.config.image_size, self.config.image_size, 3))
        self.model.trainable = False
        self.runtime = {'tensorflow': tf.__version__, 'opencv': cv2.__version__,
                        'cascade_sha256': self.preprocessor.cascade_sha256}
        digest = hashlib.sha256()
        for weights in self.model.get_weights():
            digest.update(weights.tobytes())
        self.runtime['backbone_sha256'] = digest.hexdigest()

    def embed(self, images):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        batch = np.stack(images).astype(np.float32)
        result = self.model(preprocess_input(batch), training=False).numpy()
        if result.ndim != 2 or result.shape[1] != 1280 or not np.isfinite(result).all():
            raise ValueError('Unexpected or non-finite MobileNetV2 embeddings.')
        return result.astype(np.float32)

    def extract(self, rows, cache_dir, batch_size=32):
        if len(rows) == 0 or batch_size < 1:
            raise ValueError('Extraction needs nonempty rows and a positive batch size.')
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = {'config': asdict(self.config), 'runtime': self.runtime,
                    'samples': rows[['sample_id', 'file_sha256', 'label']].to_dict('records')}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = cache_dir / f'{key}.npz'
        metadata_path = cache_dir / f'{key}.json'
        # Verify content before accepting a cache keyed by an earlier manifest.
        for row in rows.itertuples():
            if hashlib.sha256(Path(row.path).read_bytes()).hexdigest() != row.file_sha256:
                raise ValueError(f'Image changed since the audit: {row.path}. Re-run the audit.')
        if cache.exists() and metadata_path.exists():
            with np.load(cache, allow_pickle=False) as saved:
                X = saved['X']
                if saved['sample_ids'].tolist() != rows.sample_id.tolist():
                    raise ValueError('Cached sample order mismatch.')
            if X.shape != (len(rows), 1280) or not np.isfinite(X).all():
                raise ValueError('Invalid feature cache; delete this cache and rerun.')
            return X, pd.DataFrame(json.loads(metadata_path.read_text())['detections'])
        vectors, detections = [], []
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows.iloc[start:start + batch_size].itertuples():
                image, info = self.preprocessor.prepare(row.path, fer_style=True)
                images.append(image)
                detections.append(dict(sample_id=row.sample_id, label=row.label, split=row.split, **info))
            vectors.append(self.embed(images))
            current = min(start + batch_size, len(rows))
            pct = 100.0 * current / len(rows)
            if current % 1024 < batch_size or current == len(rows):
                print(f'Extracted {current}/{len(rows)} ({pct:.1f}%)', flush=True)
            else:
                print(f'Extracted {current}/{len(rows)} ({pct:.1f}%)', end='\r', flush=True)
        print(flush=True)
        X = np.concatenate(vectors)
        np.savez_compressed(cache, X=X, sample_ids=rows.sample_id.to_numpy(dtype=str))
        metadata_path.write_text(json.dumps({'identity': identity, 'detections': detections}, indent=2))
        return X, pd.DataFrame(detections)
