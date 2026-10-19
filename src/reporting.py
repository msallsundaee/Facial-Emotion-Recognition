"""Build a concise, measured-results draft for the laboratory submission."""
from pathlib import Path
import json
import pandas as pd


def write_summary(test_scores, selected_name, output_dir, audit_dir):
    output_dir, audit_dir = Path(output_dir), Path(audit_dir)
    lines = ['# Week 7 comparative summary', '',
             'Frozen ImageNet MobileNetV2 features were standardized and classified with '
             'Logistic Regression and Linear SVM. The deployment model was selected by '
             f'validation macro-F1: **{selected_name}**. Test results did not determine that selection.', '',
             '| Model | Accuracy | Macro-precision | Macro-recall | Macro-F1 |',
             '|---|---:|---:|---:|---:|']
    for row in test_scores.itertuples():
        lines.append(f'| {row.model} | {row.accuracy:.4f} | {row.macro_precision:.4f} | '
                     f'{row.macro_recall:.4f} | {row.macro_f1:.4f} |')
    lines.append('')
    for name in test_scores.model:
        report = pd.read_csv(output_dir / 'metrics' / f'{name}_classification_report.csv', index_col=0)
        class_rows = report.iloc[:7]
        hardest = class_rows['recall'].idxmin()
        confusions = (output_dir / 'metrics' / f'{name}_common_confusions.txt').read_text().splitlines()
        lines.append(f'- {name}: the lowest class recall was {hardest} '
                     f'({class_rows.loc[hardest, "recall"]:.4f}). '
                     f'The largest off-diagonal confusion was {confusions[0] if confusions else "none."}')
    best = test_scores.sort_values(['macro_f1', 'model'], ascending=[False, True]).iloc[0]
    lines.append(f'- The higher test macro-F1 was {best.model} ({best.macro_f1:.4f}); '
                 'this is a descriptive comparison, not evidence of statistically significant superiority.')
    policy = json.loads((audit_dir / 'duplicate_policy.json').read_text())
    lines.append(f'- Training used {policy["effective_training_images"]:,} images after '
                 f'{policy["excluded_training_images"]:,} documented exclusions. '
                 f'The supplied {policy["test_images"]:,}-image test set was unchanged; '
                 'test-only duplicates and label conflicts remain a limitation.')
    detection = pd.read_csv(output_dir / 'metrics/detections.csv')
    fallbacks = int(detection.fallback.sum())
    lines.append(f'- {fallbacks:,}/{len(detection):,} processed images used the full-face-image fallback '
                 'because Haar detection failed. These are not detection successes. '
                 'Error causes require visual interpretation; predictions are dataset expression categories.')
    lines += ['', 'Figures, complete reports, exclusions, and prediction filenames accompany this draft. '
              'Fit this comparison to at most one page in the provided submission form.']
    path = output_dir / 'submission/comparative_summary.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    return path
