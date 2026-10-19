# Week 7 Facial Emotion Recognition Laboratory Report

Name: [Enter your name]
Course and section: [Enter your course and section]
Date: [Enter submission date]
Team members and actual roles: [Enter the work completed by each member]

## Objective

This activity used FER2013 to classify seven facial expressions. I followed Track B, using MobileNetV2 to extract features and Logistic Regression and Linear SVM to predict the labels.

## Dataset and train-test verification

I used the public FER2013 dataset from Kaggle. It contained 28,709 training images and 7,178 test images. All images were readable and had a size of 48×48 pixels.

I excluded 1,797 training images: 703 matched test images, 73 had conflicting training labels, and 1,021 were extra copies. This left 26,912 training images. The original files were kept.

All 7,178 test images were kept. This set includes both PublicTest and PrivateTest images. Test labels were not used to train or choose the models.


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


I used 21,529 images for fitting and 5,383 for validation, with seed 42. After choosing the model using validation results, both models were trained again on all 26,912 retained training images.

Some classes had more images than others. For example, happy had 7,030 training images, while disgust had only 350. I included macro-F1 because it gives each class equal weight.

![FER2013 samples from all seven expression classes](figures/class_samples.png)

*Figure 1. Sample training images from all seven classes, enlarged for viewing.*

## Milestone 1 Face detection and visual inspection

I used OpenCV’s Haar detector to find faces. Images were enlarged to 192 pixels for detection, and the largest box was used for cropping. The settings were scale factor 1.1, minimum neighbors 5, and minimum face size 30×30.

The detector returned boxes for 54.83% of training images and 55.88% of test images. When no box was found, I used the whole image because FER2013 images were already cropped around faces. This fallback was used for 44.95% of all processed images.

These rates show how often a box was returned, not how accurate it was. Some boxes were incorrect, such as the box around only the mouth in one disgust image.

![Haar candidate boxes and explicitly labeled detection fallbacks](figures/face_detection.png)

*Figure 2. Face boxes and fallbacks. One disgust image has an incorrect box around the mouth.*

## Milestone 2 Feature extraction and model training

I resized each crop to 224×224 and repeated the grayscale channel to make three channels. MobileNetV2 preprocessing scaled the pixels to −1 to 1. I used frozen ImageNet weights, removed the classification head, and applied global average pooling; I did not train the CNN.

MobileNetV2 produced 1,280 features per image. The feature shapes were (26,912, 1,280) for training and (7,178, 1,280) for testing. I processed batches of 32 and saved the features for reuse.

Two models were trained on the same features and partitions:

1. Logistic Regression: StandardScaler, C=1.0, lbfgs solver, up to 5,000 iterations, and seed 42.
2. Linear SVM: StandardScaler, C=1.0, squared-hinge loss, L2 regularization, dual='auto', up to 20,000 iterations, and seed 42.

Both models used StandardScaler without class weighting. The scaler learned only from the training portion, so validation and test data did not affect scaling.

I selected Logistic Regression because its validation macro-F1 was 0.4030, slightly higher than SVM’s 0.4000. The test scores were not used to make this choice.

## Milestone 3 Evaluation

I tested both models on the same 7,178 images. Accuracy is shown as a percentage, while the macro scores use a 0–1 scale and give equal weight to each class.


| Model | Accuracy (%) | Macro-precision | Macro-recall | Macro-F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 47.77 | 0.4173 | 0.4239 | 0.4182 |
| Linear SVM | 47.98 | 0.4133 | 0.4202 | 0.4095 |


SVM correctly classified 3,444 images, while Logistic Regression correctly classified 3,429. SVM had slightly higher accuracy, but Logistic Regression had higher macro-F1.

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

Logistic Regression most often confused sad with neutral, with 253 cases. It also predicted fear as sad in 209 cases and sad as happy in 186 cases. The matrix shows these mistakes but does not explain their exact causes.

SVM also most often confused sad with neutral, with 267 cases. It predicted sad as happy in 220 cases and neutral as happy in 219 cases. It recognized more happy images correctly but had lower F1 for disgust and fear than Logistic Regression.

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

The predict_emotion function uses the same preprocessing and features as training, then predicts a label using the saved Logistic Regression model. It also shows whether a face was detected or the full image was used.

I demonstrated the function on two test images that were not used for training or model selection. They were selected without checking whether the predictions would be correct.

| Filename | True label | Predicted label | Preprocessing | Outcome |
|---|---|---|---|---|
| PrivateTest_10131363.jpg | angry | angry | Detected face crop | Correct |
| PrivateTest_11895083.jpg | disgust | fear | Full-image fallback | Incorrect |

The angry image was predicted correctly, but the disgust image was predicted as fear. Reloading the saved model gave the same results. This shows that the function works, although it still makes mistakes.

![Two held-out predictions, one correct and one incorrect](figures/prediction_demo.png)

*Figure 7. Two test predictions: angry was correct, while disgust was predicted as fear.*

## Reflection and limitations

The activity showed that CNN features can be used with simple classifiers. However, the selected model reached only 47.77% accuracy, so it still needs improvement. Its labels describe facial expressions, not proof of what someone actually feels.

I removed exact training/test overlap, but similar images or the same people may still appear in both sets. The test set also has 86 repeated copies beyond unique images and four groups with conflicting labels. I used one random seed and did not test another dataset.

Possible improvements include better face detection, class weighting, and fine-tuning the CNN. These were not tested here. Any changes should be chosen using validation data, not the test results.

## Conclusion

I completed the Track B pipeline and compared two classifiers. Logistic Regression was selected using validation macro-F1 and reached 47.77% test accuracy with a macro-F1 of 0.4182. The system can predict labels, but detection problems and classification errors remain.

## References

1. Supplied Week 7 Facial Emotion Recognition laboratory manual.
2. Supplied Week 7 and 8 Facial Expression Analysis lecture slides.
3. FER2013 folder-format dataset: https://www.kaggle.com/datasets/msambare/fer2013
4. Keras MobileNetV2 documentation: https://keras.io/2/api/applications/mobilenet/
5. scikit-learn common pitfalls and data leakage: https://scikit-learn.org/stable/common_pitfalls.html
