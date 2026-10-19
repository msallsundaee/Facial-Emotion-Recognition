"""Inspect FER-style class folders without silently discarding observations."""
from pathlib import Path
import hashlib
import json

import cv2
import pandas as pd

CLASSES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def scan_dataset(root):
    root = Path(root).expanduser().resolve()
    rows = []
    for split in ("train", "test"):
        folder = root / split
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing {folder}. Set DATA_ROOT to the folder containing train/ and test/.")
        unexpected = {p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith('.')} - set(CLASSES)
        if unexpected:
            raise ValueError(f"Unexpected class folders in {folder}: {sorted(unexpected)}")
        for label in CLASSES:
            class_dir = folder / label
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            paths = sorted(p for p in class_dir.rglob('*') if p.suffix.lower() in EXTENSIONS)
            if not paths:
                raise ValueError(f"No images in {class_dir}")
            for path in paths:
                raw = path.read_bytes()
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                readable = gray is not None
                height, width = gray.shape if readable else (0, 0)
                # Decoded-pixel hash catches identical images in different containers.
                pixel_hash = hashlib.sha256(str((height, width)).encode() + gray.tobytes()).hexdigest() if readable else ''
                rows.append(dict(sample_id=path.relative_to(root).as_posix(), path=str(path),
                                 split=split, label=label, readable=readable,
                                 width=width, height=height,
                                 file_sha256=hashlib.sha256(raw).hexdigest(), pixel_sha256=pixel_hash))
    return pd.DataFrame(rows)


def audit_dataset(manifest, output_dir, duplicate_policy='error'):
    """Write all evidence BEFORE rejecting unusable or leaking data."""
    if duplicate_policy not in {'error', 'report'}:
        raise ValueError('duplicate_policy must be error or report.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / 'manifest.csv', index=False)
    counts = pd.crosstab(manifest.label, manifest.split).reindex(CLASSES, fill_value=0)
    counts.to_csv(output_dir / 'class_counts.csv')
    valid = manifest[manifest.readable]
    duplicate_rows = valid[valid.duplicated('pixel_sha256', keep=False)]
    duplicate_rows.to_csv(output_dir / 'duplicates.csv', index=False)
    groups = valid.groupby('pixel_sha256')
    cross_split = set(groups.split.nunique().loc[lambda s: s > 1].index)
    conflicting = set(groups.label.nunique().loc[lambda s: s > 1].index)
    summary = dict(total=len(manifest), unreadable=int((~manifest.readable).sum()),
                   non_48x48=int(((manifest.width != 48) | (manifest.height != 48)).sum()),
                   cross_split_duplicate_groups=len(cross_split), conflicting_label_groups=len(conflicting),
                   duplicate_rows=len(duplicate_rows))
    (output_dir / 'audit.json').write_text(json.dumps(summary, indent=2))
    problems = []
    if summary['unreadable']:
        problems.append('unreadable images')
    if summary['non_48x48']:
        problems.append('images other than 48x48')
    if cross_split and duplicate_policy == 'error':
        problems.append('identical decoded images across train/test')
    if conflicting and duplicate_policy == 'error':
        problems.append('identical decoded images with conflicting labels')
    if problems:
        raise ValueError('Dataset audit stopped: ' + ', '.join(problems) +
                         f'. Inspect {output_dir}; resolve and document the data issue before training.')
    return counts, summary


def prepare_training_data(manifest, output_dir):
    """Keep the supplied test set unchanged; exclude documented training duplicates.

    Uses test pixels only for overlap checking, never test labels to select training
    labels. Test-only duplicate/conflicting groups remain a reported limitation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = manifest[manifest.split == 'train'].copy()
    test = manifest[manifest.split == 'test'].copy()
    reasons = pd.Series('', index=train.index)
    conflicts = train.groupby('pixel_sha256').label.nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    reasons.loc[train.pixel_sha256.isin(conflicts)] = 'conflicting_training_labels'
    reasons.loc[train.pixel_sha256.isin(set(test.pixel_sha256))] = 'exact_test_overlap'
    eligible = train.loc[reasons == ''].sort_values('sample_id')
    repeated = eligible.index[eligible.duplicated('pixel_sha256', keep='first')]
    reasons.loc[repeated] = 'repeated_training_image'
    excluded = train.loc[reasons != ''].copy()
    excluded['exclusion_reason'] = reasons.loc[excluded.index]
    kept = train.loc[reasons == ''].reset_index(drop=True)
    test = test.reset_index(drop=True)
    if set(kept.pixel_sha256) & set(test.pixel_sha256):
        raise AssertionError('Train/test overlap remains after filtering.')
    if set(kept.label) != set(CLASSES):
        raise ValueError('Duplicate handling removed an entire training class.')
    excluded.to_csv(output_dir / 'training_exclusions.csv', index=False)
    pd.concat([kept, test], ignore_index=True).to_csv(output_dir / 'effective_manifest.csv', index=False)
    counts = pd.DataFrame({'original_train': train.groupby('label').size(),
                           'effective_train': kept.groupby('label').size(),
                           'unchanged_test': test.groupby('label').size()}).reindex(CLASSES)
    counts.to_csv(output_dir / 'effective_counts.csv')
    (output_dir / 'duplicate_policy.json').write_text(json.dumps({
        'policy': 'preserve_test_filter_training_v1',
        'excluded_training_images': len(excluded), 'effective_training_images': len(kept),
        'test_images': len(test), 'exclusion_counts': excluded.exclusion_reason.value_counts().to_dict(),
        'note': 'Test-only duplicates and label conflicts remain in the fixed test set. '
                'No near-duplicate or identity-disjointness guarantee.'}, indent=2))
    return kept, test, counts


def validation_partition(train_rows, seed=42, fraction=0.2):
    """Stratify unique image groups; duplicates never straddle fit/validation."""
    from sklearn.model_selection import train_test_split
    groups = train_rows.drop_duplicates('pixel_sha256')
    if (groups.groupby('label').size().reindex(CLASSES, fill_value=0) < 2).any():
        raise ValueError('Need at least two distinct images per class for validation.')
    fit_hashes, val_hashes = train_test_split(groups.pixel_sha256, test_size=fraction,
                                            random_state=seed, stratify=groups.label)
    fit = train_rows.pixel_sha256.isin(fit_hashes).to_numpy().nonzero()[0]
    val = train_rows.pixel_sha256.isin(val_hashes).to_numpy().nonzero()[0]
    return fit, val
