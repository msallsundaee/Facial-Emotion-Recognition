# Week 7 laboratory submission checklist

Use the Laboratory Manual Submission Form at the end of the supplied DOCX.
The notebook creates evidence; screenshots and explanations must describe the actual run.

- Fill in name, section, date, topic, week, and team roles.
- Split verification: class counts, sample grid, source URL and archive hash, and any exclusions.
- Detection: actual bounding boxes across classes, detection rates, and explicit fallback counts.
- Feature extraction: frozen ImageNet MobileNetV2, 224×224 RGB conversion, its preprocessing,
  global average pooling, and feature matrix dimensions.
- Training: screenshots of both pipelines, standardization, C, iteration limit, seed,
  fit/validation sizes, and the validation-based selection result.
- Evaluation: full classification reports and 7×7 confusion matrices for both models.
- Comparative summary (at most one page): accuracy, macro-precision, macro-recall, macro-F1
  for both models, plus 3–5 findings supported by these outputs.
- Reflection: 2–3 sentences per model describing actual confusions. Inspect example images
  before suggesting causes; label possible causes as interpretations.
- Prediction proof: at least two held-out image filenames and predicted labels. State that
  these are test images if selected from that split. Include their true labels for context.
- Reproducibility: rerun from a fresh kernel, retain package versions and saved models.

Do not claim these categories establish a person's internal emotional state. The provided
rubric lists 110 points despite its 100-point heading; cover every listed category.
