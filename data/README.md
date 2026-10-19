# FER2013 data

Expected layout: `train/<class>/*` and `test/<class>/*`, with classes
`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, and `neutral`.

Public folder-format source: https://www.kaggle.com/datasets/msambare/fer2013
The downloaded archive and source metadata stay in this directory and are not committed.
Record its hash in `source.json`. The folder-format test split must not be described
as the original competition's PrivateTest split without a verified mapping.

The notebook audits decoded pixels, filenames, dimensions, and labels. Audit reports
are saved before a failure; do not silently skip corrupt files or edit source images.
