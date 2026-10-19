"""Validation-based model selection and fixed-label final evaluation."""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data import CLASSES


def make_models(seed=42):
    return {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=seed)),
        'linear_svm': make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual='auto', max_iter=20000, random_state=seed)),
    }


def fit_checked(model, X, y):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X, y)
    failures = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
    if failures:
        raise RuntimeError('Classifier did not converge: ' + '; '.join(failures) +
                           '. Adjust iteration budget using validation data before final evaluation.')
    for warning in captured:
        warnings.warn(str(warning.message), warning.category)
    return model


def metrics(y, predictions):
    p, r, f, _ = precision_recall_fscore_support(y, predictions, labels=CLASSES,
                                               average='macro', zero_division=0)
    return dict(accuracy=accuracy_score(y, predictions), macro_precision=p, macro_recall=r, macro_f1=f)


def train_models(X, rows, fit_indices, val_indices, output_dir, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(X) != len(rows) or not np.isfinite(X).all():
        raise ValueError('Features must be finite and aligned with the training manifest.')
    if set(rows.split) != {'train'}:
        raise ValueError('Model selection accepts only training-split images.')
    fit_indices, val_indices = np.asarray(fit_indices), np.asarray(val_indices)
    if set(fit_indices) & set(val_indices) or set(fit_indices) | set(val_indices) != set(range(len(rows))):
        raise ValueError('Fit/validation indices must partition all training rows without overlap.')
    if set(rows.iloc[fit_indices].pixel_sha256) & set(rows.iloc[val_indices].pixel_sha256):
        raise ValueError('Duplicate images overlap fit and validation.')
    y = rows.label.to_numpy()
    candidates = make_models(seed)
    scores, fitted = [], {}
    for name, prototype in candidates.items():
        print(f'Fitting {name} on {len(fit_indices)} images; validating on {len(val_indices)}.')
        model = fit_checked(clone(prototype), X[fit_indices], y[fit_indices])
        scores.append(dict(model=name, **metrics(y[val_indices], model.predict(X[val_indices]))))
    comparison = pd.DataFrame(scores).sort_values(['macro_f1', 'model'], ascending=[False, True])
    selected = comparison.iloc[0].model
    comparison.to_csv(output_dir / 'validation_metrics.csv', index=False)
    partition = rows[['sample_id', 'pixel_sha256', 'label']].copy()
    partition['partition'] = 'fit'
    partition.iloc[val_indices, partition.columns.get_loc('partition')] = 'validation'
    partition.to_csv(output_dir / 'validation_partition.csv', index=False)
    settings = {name: {k: v for k, v in model.steps[-1][1].get_params().items()
                       if isinstance(v, (str, int, float, bool, type(None)))} for name, model in candidates.items()}
    (output_dir / 'selection.json').write_text(json.dumps(dict(
        selected_model=selected, selection_metric='validation macro-F1', seed=seed,
        tie_break='alphabetical model name', settings=settings), indent=2))
    for name, prototype in candidates.items():
        print(f'Refitting finalized {name} on all {len(rows)} training images.')
        fitted[name] = fit_checked(clone(prototype), X, y)
        joblib.dump(fitted[name], output_dir / f'{name}.joblib')
    return fitted, selected, comparison


def evaluate_models(models, X, rows, output_dir):
    from .plots import plot_confusion
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if set(rows.split) != {'test'} or len(X) != len(rows):
        raise ValueError('Final evaluation requires aligned test-split rows.')
    y = rows.label.to_numpy()
    scores, predictions = [], rows[['sample_id', 'path', 'label']].copy()
    for name, model in models.items():
        pred = model.predict(X)
        predictions[name] = pred
        scores.append(dict(model=name, **metrics(y, pred)))
        report = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(output_dir / f'{name}_classification_report.csv')
        (output_dir / f'{name}_classification_report.txt').write_text(
            classification_report(y, pred, labels=CLASSES, zero_division=0, digits=4))
        cm = confusion_matrix(y, pred, labels=CLASSES)
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(output_dir / f'{name}_confusion_matrix.csv')
        fig = plot_confusion(cm, name)
        fig.savefig(output_dir / f'{name}_confusion_matrix.png', dpi=160, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)
        off_diagonal = cm.copy()
        np.fill_diagonal(off_diagonal, 0)
        pairs = sorted(((int(off_diagonal[i, j]), CLASSES[i], CLASSES[j])
                        for i in range(7) for j in range(7) if i != j), reverse=True)
        notes = [f'{true} → {guess}: {n} test images.' for n, true, guess in pairs[:3] if n]
        (output_dir / f'{name}_common_confusions.txt').write_text('\n'.join(notes) or 'No test errors.')
    comparison = pd.DataFrame(scores)
    comparison.to_csv(output_dir / 'test_metrics.csv', index=False)
    predictions.to_csv(output_dir / 'test_predictions.csv', index=False)
    return comparison, predictions
