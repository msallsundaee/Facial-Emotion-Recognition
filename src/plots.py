"""Notebook figures that can also be saved as submission evidence."""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .data import CLASSES


def sample_grid(rows, preprocessor=None, per_class=3, seed=42):
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 3, 16), squeeze=False)
    for i, label in enumerate(CLASSES):
        subset = rows[rows.label == label]
        subset = subset.sample(min(per_class, len(subset)), random_state=seed)
        for ax in axes[i]:
            ax.axis('off')
        for ax, row in zip(axes[i], subset.itertuples()):
            gray = cv2.imread(row.path, cv2.IMREAD_GRAYSCALE)
            ax.imshow(gray, cmap='gray', vmin=0, vmax=255)
            title = label
            if preprocessor:
                _, info = preprocessor.prepare(row.path)
                if info['detected']:
                    from matplotlib.patches import Rectangle
                    x, y, w, h = info['box']
                    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=1.5))
                    title += ' | detected'
                else:
                    title += ' | fallback (no detection)'
            ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion(cm, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True),
                           out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    for ax, values, title in zip(axes, (cm, normalized), ('Counts', 'Fraction within true class')):
        ax.imshow(values, cmap='Blues')
        for i in range(7):
            for j in range(7):
                text = str(cm[i, j]) if title == 'Counts' else f'{values[i, j]:.2f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=8,
                        color='white' if values[i, j] > values.max() / 2 else 'black')
        ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted label', ylabel='True label', title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.suptitle(name)
    fig.tight_layout()
    return fig


def error_grid(predictions, model_name, limit=3):
    errors = predictions[predictions.label != predictions[model_name]].head(limit)
    if errors.empty:
        print('No misclassified test examples.')
        return None
    fig, axes = plt.subplots(1, len(errors), figsize=(4 * len(errors), 4), squeeze=False)
    for ax, (_, row) in zip(axes.flat, errors.iterrows()):
        ax.imshow(cv2.imread(row.path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        ax.set_title(f'True: {row.label}\nPredicted: {row[model_name]}')
        ax.axis('off')
    fig.tight_layout()
    return fig
