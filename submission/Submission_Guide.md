# Week 7 submission guide

Use `Week7_Submission_Text.md` or `Week7_Submission_Text.txt` as the written report.
The Markdown version embeds seven figures with captions from the full saved run. The
plain-text version includes matching figure filenames and captions because `.txt` files
cannot display images. The images are bundled in `submission/figures/`; keep that folder
beside the Markdown file when copying or sharing the report. Open the Markdown preview
to see the figures. Paste the content into the
Laboratory Manual Submission Form at the end of the supplied Word document. Fill in
your name, section, submission date, and actual team roles. No personal information or
team contributions have been invented.

## Evidence to insert

| Required evidence | Where to get it | Suggested caption |
|---|---|---|
| Dataset split | Notebook cell beginning `manifest = scan_dataset(DATA_ROOT)` | Original and retained FER2013 class counts, with documented training exclusions. |
| Sample verification | `outputs/figures/class_samples.png` | Example 48×48 FER2013 images from all seven expression classes. |
| Face detection | `outputs/figures/face_detection.png` | Haar candidate boxes and explicitly labeled no-detection fallbacks. One disgust candidate localizes only the mouth; not every returned box is a correct face crop. |
| Feature dimensions | Notebook cell beginning `extractor = FeatureExtractor(config)` | Frozen MobileNetV2 produced 1,280 features for each of 26,912 training and 7,178 test images. |
| Both trained models | Notebook cell beginning `fit_idx, val_idx = validation_partition(...)` | Identical fit/validation partitions and validation-macro-F1 model selection. Include the displayed model settings. |
| Logistic Regression report | Evaluation-cell output or `outputs/metrics/logistic_regression_classification_report.txt` | Seven-class precision, recall, and F1 for Logistic Regression on 7,178 test images. |
| SVM report | Evaluation-cell output or `outputs/metrics/linear_svm_classification_report.txt` | Seven-class precision, recall, and F1 for Linear SVM on the same test images. |
| Logistic Regression matrix | `outputs/metrics/logistic_regression_confusion_matrix.png` | True labels are rows; predicted labels are columns. Left: counts. Right: within-class fractions. |
| SVM matrix | `outputs/metrics/linear_svm_confusion_matrix.png` | Linear SVM test confusions using the same class order. |
| Error examples | `outputs/figures/logistic_regression_errors.png` and `outputs/figures/linear_svm_errors.png` | Selected misclassified test examples; their appearance suggests possible explanations but does not establish causes. |
| Prediction proof | `outputs/figures/prediction_demo.png` plus the final notebook table | Two held-out test predictions: angry correctly recognized; disgust misclassified as fear. |

The supplied laboratory explicitly asks for screenshots. The written tables support the
answers but do not replace every required screenshot. Use the complete active notebook
`notebooks/week7_fer.ipynb`; the older `outputs/week7_fer_executed.ipynb` was from an
interrupted automated run and is incomplete. Do not present the older export as a full run.

## What to paste where

1. Paste the Objective and Dataset sections, followed by the split-count screenshot.
2. Paste Milestone 1 and insert the detection grid. Retain the false-localization observation.
3. Paste Milestone 2 and insert feature dimensions and training evidence.
4. Paste Milestone 3, both full reports, both confusion matrices, and their interpretations.
5. Keep the Comparative Summary table and five insights together, within one page. The
   page limit applies to that summary, not to the complete report. If the main metrics
   table is already nearby, avoid repeating it unnecessarily.
6. Paste Simple Deployment and Prediction Proof with both image filenames visible.
7. Paste Reflection, Conclusion, and References. Add truthful team roles.

## Short answers for explaining the work

**What does face detection do?** It proposes the location of a face. It does not assign
an emotion label. In this implementation, Haar sometimes returns no box and sometimes
returns an incorrect box.

**What are facial landmarks?** They are coordinate locations for facial structures such
as eye corners, the nose, and mouth. Landmark coordinates or geometric ratios can be
used as features in Track A. They were discussed in the lecture but were not extracted
in this Track B implementation.

**What are action units?** They describe visible facial movements, such as lip-corner
pulling or brow lowering, within the Facial Action Coding System. They are not emotion
labels themselves. This implementation does not estimate action units.

**Why did we use MobileNetV2?** The lab permits a pretrained CNN feature extractor. Here,
MobileNetV2 converts each image into 1,280 numbers that the two classical classifiers
can use. It is an implementation choice, not evidence that it is the best FER backbone.

**Did we train a CNN?** No. The ImageNet CNN weights stayed frozen. The trained models
were Logistic Regression and Linear SVM on the extracted embeddings.

**Why three channels for a grayscale image?** The selected pretrained model expects
three-channel input. Replicating grayscale satisfies the input shape; it does not add
color information. Resizing likewise does not recover detail absent from a 48×48 image.

**Why standardize features?** Scaling prevents feature dimensions with larger numeric
variation from dominating the classical model merely because of their scale. The scaler
is fitted inside the training pipeline so validation/test data do not determine it.

**Why macro-F1?** It averages the per-class F1 scores equally. This helps expose weak
classes such as disgust even when a model predicts the more common classes well.
Macro-F1 is not generally the harmonic mean of macro-precision and macro-recall.

**Which model is better?** For the chosen selection criterion, Logistic Regression:
validation macro-F1 0.4030 versus 0.4000. On test, it also has higher macro-F1 (0.4182
versus 0.4095). SVM has higher test accuracy (47.98% versus 47.77%). Neither result proves
universal or statistically significant superiority.

**Why remove some training images?** Exact images shared with the test set can inflate
evaluation, while duplicate training copies overweight those examples. Training-label
conflicts were excluded rather than arbitrarily relabeled. The 1,797 exclusions are
logged; the test set and source files remain unchanged.

**Is this an 80/20 split of the full dataset?** The supplied folders contain 28,709 train
and 7,178 test images. After training exclusions, the effective counts are 26,912 and
7,178. The additional 80/20 split applies only to the retained training set: 21,529 fit
and 5,383 validation images.

**Are the two demonstration images unseen?** They were excluded from classifier fitting
and model selection. They belong to the evaluated test set, so describe them as held-out
test examples, not newly collected external images.

**Why is one demonstration wrong?** The function runs correctly but the classifier is
imperfect. The disgust image was labeled fear. Its cause cannot be established from one
prediction. Keeping the failure makes the demonstration representative of actual behavior.

**Can we claim 55% face-detection accuracy?** No. About 55% of images produced a Haar box.
Without annotated reference boxes and a localization criterion, this is a box-return
rate, not an accuracy measurement.

**Can the model identify what a person truly feels?** No. It predicts one of seven
dataset expression labels from an image. Internal emotion and real-world reliability
were not established by this experiment.

## Before submitting

- Fill only the remaining personal/team fields and include the required screenshots.
- Keep reported numbers tied to the full 7,178-image test set.
- Do not use `outputs/smoke/` in the report: it is only a small software check.
- Preserve the correct and incorrect demo predictions; do not replace the failed example
  merely to make the demonstration appear perfect.
- Do not claim that landmark extraction, action-unit detection, CNN fine-tuning, perfect
  face detection, or identity-disjoint evaluation was performed.
- The rubric's listed categories total 110 although the heading says 100. Cover all categories.
