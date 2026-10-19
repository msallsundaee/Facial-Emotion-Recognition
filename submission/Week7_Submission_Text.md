# Week 7 Facial Emotion Recognition Laboratory Report

Name: [Enter your name]
Course and section: [Enter your course and section]
Date: [Enter submission date]
Team members and actual roles: [Enter the work completed by each member]

## Objective

This activity focused on classifying seven facial expressions using FER2013. It followed Track B, where MobileNetV2 extracts image features and Logistic Regression and Linear SVM use those features to predict a label. Comparing the two models helps show how well the same features work with different classifiers.

## Dataset and train-test verification

The public FER2013 dataset from Kaggle contained 28,709 training images and 7,178 test images. All images were readable and measured 48×48 pixels. Checking the image sizes and class folders helped confirm that the dataset matched the laboratory requirements.

A total of 1,797 images were excluded from training: 703 matched test images, 73 had conflicting training labels, and 1,021 were extra copies. This left 26,912 training images. Removing these copies helps prevent the model from being tested on an exact image it already learned from, while the original files remain unchanged.

All 7,178 test images were kept, including both PublicTest and PrivateTest images. Their labels were not used to train or choose the models, so they could be used for the final evaluation.


| Expression | Original training | Retained training | Test |
| --- | --- | --- | --- |
| angry | 3995 | 3776 | 958 |
| disgust | 436 | 350 | 111 |
| fear | 4097 | 3797 | 1024 |
| happy | 7215 | 7030 | 1774 |
| sad | 4830 | 4677 | 1247 |
| surprise | 3171 | 2460 | 831 |
| neutral | 4965 | 4822 | 1233 |
| Total | 28709 | 26912 | 7178 |


The retained training data was divided into 21,529 fitting images and 5,383 validation images, using seed 42. The validation set helped choose between the models before looking at test scores. Both models were then trained again on all 26,912 retained training images.

The classes were not equally represented. Happy had 7,030 training images, while disgust had only 350. Macro-F1 was included because it gives each class equal weight, making weak performance on smaller classes easier to notice.

![FER2013 samples from all seven expression classes](figures/class_samples.png)

*Figure 1. Sample training images from all seven classes, enlarged for viewing.*

## Milestone 1 Face detection and visual inspection

OpenCV’s Haar detector was used to locate faces before feature extraction. Images were enlarged to 192 pixels for detection, and the largest returned box was used for cropping. The settings were scale factor 1.1, minimum neighbors 5, and minimum face size 30×30.

The detector returned boxes for 54.83% of training images and 55.88% of test images. When no box was found, the whole image was used because FER2013 images were already cropped around faces. This fallback kept those images in the experiment and was needed for 44.95% of all processed images.

These rates show how often the detector returned a box, not whether every box was correct. For example, one disgust image had a box around only the mouth. A crop like this may leave out useful facial details, although its effect on the final scores was not measured separately.

![Haar candidate boxes and explicitly labeled detection fallbacks](figures/face_detection.png)

*Figure 2. Face boxes and fallbacks. One disgust image has an incorrect box around the mouth.*

## Milestone 2 Feature extraction and model training

Each crop was resized to 224×224, and its grayscale channel was repeated to make the three channels expected by MobileNetV2. Pixel values were scaled to −1 to 1 using the model’s preprocessing function. The ImageNet weights stayed frozen, with the classification head removed and global average pooling applied, so the CNN served as a feature extractor rather than being trained again.

MobileNetV2 represented each image using 1,280 numerical features. These became the inputs to the two classifiers, giving feature shapes of (26,912, 1,280) for training and (7,178, 1,280) for testing. Processing batches of 32 and saving the features allowed them to be reused without repeating extraction.

Two models were trained on the same features and partitions:

1. Logistic Regression: StandardScaler, C=1.0, lbfgs solver, up to 5,000 iterations, and seed 42.
2. Linear SVM: StandardScaler, C=1.0, squared-hinge loss, L2 regularization, dual='auto', up to 20,000 iterations, and seed 42.

Both models used StandardScaler to put the feature values on comparable scales, without class weighting. The scaler learned only from the training portion. This kept validation and test information from influencing the scaling step.

Logistic Regression was selected because its validation macro-F1 was 0.4030, slightly higher than SVM’s 0.4000. Macro-F1 was the chosen criterion because the dataset has unequal class sizes. The test scores were not used for this decision.

## Milestone 3 Evaluation

Both models were evaluated on the same 7,178 test images to make the comparison consistent. Accuracy shows the overall proportion of correct predictions. The macro scores use a 0–1 scale and average the results equally across the seven classes.


| Model | Accuracy (%) | Macro-precision | Macro-recall | Macro-F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 47.77 | 0.4173 | 0.4239 | 0.4182 |
| Linear SVM | 47.98 | 0.4133 | 0.4202 | 0.4095 |


SVM correctly classified 3,444 images, compared with 3,429 for Logistic Regression. Although SVM had slightly higher accuracy, Logistic Regression had higher macro-F1. This shows why overall accuracy alone does not fully describe performance across all classes.

### Per-class classification reports


#### Logistic Regression

```text
              precision    recall  f1-score   support

       angry     0.3579    0.3392    0.3483       958
     disgust     0.1524    0.2252    0.1818       111
        fear     0.3362    0.2646    0.2962      1024
       happy     0.6424    0.7311    0.6839      1774
         sad     0.3824    0.3480    0.3644      1247
    surprise     0.6088    0.5692    0.5883       831
     neutral     0.4412    0.4899    0.4643      1233

    accuracy                         0.4777      7178
   macro avg     0.4173    0.4239    0.4182      7178
weighted avg     0.4695    0.4777    0.4717      7178
```


#### Linear SVM

```text
              precision    recall  f1-score   support

       angry     0.3839    0.3278    0.3536       958
     disgust     0.1127    0.2072    0.1460       111
        fear     0.3720    0.2129    0.2708      1024
       happy     0.6074    0.7621    0.6760      1774
         sad     0.3938    0.3448    0.3677      1247
    surprise     0.5864    0.5800    0.5832       831
     neutral     0.4371    0.5069    0.4694      1233

    accuracy                         0.4798      7178
   macro avg     0.4133    0.4202    0.4095      7178
weighted avg     0.4675    0.4798    0.4672      7178
```


### Confusion matrices

Rows represent true labels and columns represent predicted labels. Entries are image counts.


#### Logistic Regression

![Logistic Regression confusion matrices with counts and within-class fractions](figures/logistic_regression_confusion_matrix.png)

*Figure 3. Logistic Regression confusion matrices: counts on the left and within-class fractions on the right.*

| True / predicted | angry | disgust | fear | happy | sad | surprise | neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| angry | 325 | 31 | 127 | 139 | 156 | 47 | 133 |
| disgust | 37 | 25 | 16 | 15 | 10 | 2 | 6 |
| fear | 142 | 18 | 271 | 121 | 209 | 119 | 144 |
| happy | 78 | 24 | 67 | 1297 | 104 | 59 | 145 |
| sad | 175 | 36 | 130 | 186 | 434 | 33 | 253 |
| surprise | 45 | 9 | 93 | 87 | 40 | 473 | 84 |
| neutral | 106 | 21 | 102 | 174 | 182 | 44 | 604 |


#### Linear SVM

![Linear SVM confusion matrices with counts and within-class fractions](figures/linear_svm_confusion_matrix.png)

*Figure 4. Linear SVM confusion matrices: counts on the left and within-class fractions on the right.*

| True / predicted | angry | disgust | fear | happy | sad | surprise | neutral |
| --- | --- | --- | --- | --- | --- | --- | --- |
| angry | 314 | 39 | 83 | 165 | 152 | 58 | 147 |
| disgust | 31 | 23 | 12 | 21 | 10 | 5 | 9 |
| fear | 132 | 26 | 218 | 154 | 201 | 135 | 158 |
| happy | 68 | 30 | 46 | 1352 | 88 | 56 | 134 |
| sad | 145 | 44 | 105 | 220 | 430 | 36 | 267 |
| surprise | 48 | 17 | 61 | 95 | 38 | 482 | 90 |
| neutral | 80 | 25 | 61 | 219 | 173 | 50 | 625 |


### Confusion-matrix interpretation

Logistic Regression most often predicted sad as neutral, with 253 cases. It also predicted fear as sad in 209 cases and sad as happy in 186 cases. These patterns show difficulty separating the labeled expressions, but the matrix alone cannot explain the exact cause of each mistake.

SVM also most often predicted sad as neutral, with 267 cases. It predicted sad as happy in 220 cases and neutral as happy in 219 cases. SVM recognized more happy images correctly, but its lower F1 for disgust and fear helps explain why its macro-F1 was below Logistic Regression’s.

![Misclassified test images from Logistic Regression](figures/logistic_regression_errors.png)

*Figure 5. Logistic Regression errors, showing the true and predicted labels.*

![Misclassified test images from Linear SVM](figures/linear_svm_errors.png)

*Figure 6. Linear SVM errors, showing the true and predicted labels.*

## Comparative summary

The table and points below summarize the two models.


| Model | Accuracy (%) | Macro-precision | Macro-recall | Macro-F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 47.77 | 0.4173 | 0.4239 | 0.4182 |
| Linear SVM | 47.98 | 0.4133 | 0.4202 | 0.4095 |


- Logistic Regression had higher test macro-F1: 0.4182 compared with 0.4095 for SVM.
- SVM had slightly higher accuracy: 47.98% compared with 47.77%. This small difference does not prove it is always better.
- Happy had the highest F1 for both models, while disgust had the lowest.
- Both models most often confused sad with neutral.
- Unequal class sizes, low image resolution, and incorrect face boxes are limitations. Almost 45% of images needed the full-image fallback.

## Simple deployment and prediction proof

The predict_emotion function applies the same preprocessing and feature extraction used during training, then returns a label from the saved Logistic Regression model. Keeping these steps the same helps avoid differences between training and prediction. The output also records whether a face box or the full image was used.

The function was demonstrated on two test images that were not used for training or model selection. They were selected without checking whether the predictions would be correct, allowing the demonstration to show actual behavior rather than only successful examples.

| Filename | True label | Predicted label | Preprocessing | Outcome |
|---|---|---|---|---|
| PrivateTest_10131363.jpg | angry | angry | Detected face crop | Correct |
| PrivateTest_11895083.jpg | disgust | fear | Full-image fallback | Incorrect |

The angry image was classified correctly, while the disgust image was predicted as fear. Reloading the saved model produced the same results, confirming that the saved pipeline could reproduce these predictions. The incorrect example also shows that a working prediction function does not guarantee a correct label.

![Two held-out predictions, one correct and one incorrect](figures/prediction_demo.png)

*Figure 7. Two test predictions: angry was correct, while disgust was predicted as fear.*

## Reflection and limitations

The activity showed how pretrained CNN features can be combined with simple classifiers to build a facial-expression system. The selected model reached 47.77% accuracy, so many expressions were still misclassified. Its outputs describe dataset expression labels and should not be treated as proof of what a person actually feels.

Exact training/test overlap was removed, but similar images or the same people may still appear in both sets. The test set also contains 86 repeated copies beyond unique images and four groups with conflicting labels. Since the experiment used only one random seed and one dataset, its results do not show how well the system would perform in other settings.

Possible improvements include better face detection, class weighting to give smaller classes more attention, and fine-tuning the CNN for facial expressions. These ideas were not tested in this activity. They should be compared using validation data before making claims about improvement.

## Conclusion

The Track B pipeline successfully extracted features, trained two classifiers, and produced predictions on test images. Logistic Regression was selected using validation macro-F1 and achieved 47.77% test accuracy with a macro-F1 of 0.4182. The results provide a starting point for facial-expression classification, while showing that face detection and recognition errors still need improvement.

## References

1. Supplied Week 7 Facial Emotion Recognition laboratory manual.
2. Supplied Week 7 and 8 Facial Expression Analysis lecture slides.
3. FER2013 folder-format dataset: https://www.kaggle.com/datasets/msambare/fer2013
4. Keras MobileNetV2 documentation: https://keras.io/2/api/applications/mobilenet/
5. scikit-learn common pitfalls and data leakage: https://scikit-learn.org/stable/common_pitfalls.html
