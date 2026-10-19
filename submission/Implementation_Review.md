# Week 7 implementation review

## Verdict

The completed notebook implements the required Track B workflow and has internally
consistent full-dataset results. This is a working baseline, with material detector and
dataset limitations. It should not be described as entirely accurate or deployment-ready.

## Findings

1. **Detector localization limitation:** visual inspection of the saved detection grid
   shows one disgust-image candidate around the mouth rather than the complete face.
   The `detected` flag means a box was returned, not that its localization was verified.
   Approximately 44.95% of processed images used fallback. The report discloses both
   issues. Changing the crop policy would require new embeddings, training, and evaluation;
   no such change was silently applied to the current measured results.
2. **Smoke-run overwrite risk, fixed:** `scripts/train_pipeline.py --sample-limit` previously
   wrote into the full run's model and results directories. It could also generate a
   summary using full audit counts for sampled results. Smoke runs now write exclusively
   to `artifacts/smoke` and `outputs/smoke`, carry an explicit smoke label, and skip the
   full-dataset submission summary. Negative or zero limits are rejected.
3. **Incomplete older notebook export:** `outputs/week7_fer_executed.ipynb` belongs to an
   interrupted automated run. The active `notebooks/week7_fer.ipynb` contains seven
   executed code cells and no saved error outputs. Use the latter for submission evidence.
4. **Dataset protocol limitations:** the 7,178-image test set remains unchanged, including
   86 repeated records beyond unique decoded images and four conflicting-label groups.
   Exact training/test overlap is removed, but near-duplicate and identity overlap are
   not ruled out. The test filenames include PublicTest and PrivateTest, so avoid
   PrivateTest-only benchmark claims.

## Verification performed

- Six pipeline tests passed, including exact-overlap rejection, validation grouping,
  training-only exclusions, explicit no-face behavior, synthetic classifier evaluation
  and reload, and rejection of test rows during model selection.
- The effective manifest contains 26,912 training and 7,178 test rows, with no exact
  decoded-pixel overlap; retained training images have unique decoded-pixel hashes.
- The saved validation partition matches the training manifest and contains 21,529 fit
  and 5,383 validation rows without exact-image overlap.
- Full feature caches match the manifests' ordered sample identifiers, labels, and
  recorded source hashes. Shapes are (26,912, 1,280) and (7,178, 1,280), with finite values.
- Both saved scalers record 26,912 fitted samples and match the training-feature means.
- Both saved classifiers reproduce every recorded test prediction. Recomputed accuracy,
  macro metrics, and all confusion-matrix entries match the saved outputs.
- Loading the deployment bundle in a fresh process reproduces both demonstration rows:
  angry → angry and disgust → fear.
- The active notebook's seven code cells have execution counts and no saved error outputs.
- All 35,887 current source-file hashes match the recorded manifest.
- A 35-image-per-split smoke run completed using the isolated directories. All 22 existing
  full model/result files remained byte-for-byte unchanged, and no misleading full-run
  summary was generated for the smoke run.

These checks establish consistency of the present implementation and artifacts; they do
not establish perfect labels, correct localization for every detected face, absence of
all forms of data leakage, or broad real-world generalization.

## Results to report

| Model | Test accuracy | Test macro-F1 | Validation macro-F1 |
|---|---:|---:|---:|
| Logistic Regression | 47.77% | 0.4182 | 0.4030 |
| Linear SVM | 47.98% | 0.4095 | 0.4000 |

Logistic Regression is the validation-selected deployment model. The text report and
guide retain both the successful and failed demonstration examples.
