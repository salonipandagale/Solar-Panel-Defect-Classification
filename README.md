# Solar Panel Defect Classification using Deep Learning

A deep learning-based multi-class image classification system for identifying different conditions and defects in solar panels.

**Live Demo:** [solar-panel-defect-classification](https://solar-panel-defect-classification-1.onrender.com/)

---

## 1. Overview

Solar panels can be affected by conditions such as dust, bird drops, snow, electrical damage, and physical damage. Automatically classifying these conditions from images can support faster inspection and maintenance.

This project develops a **multi-class image classification system** that takes a solar-panel image as input and predicts one of six categories:

* Bird-drop
* Clean
* Dusty
* Electrical-damage
* Physical-Damage
* Snow-Covered

The project explores a progression from a baseline CNN to transfer-learning approaches using **MobileNetV2** and **EfficientNetB0**, followed by hyperparameter optimization using **Keras Tuner**.

---

## 2. Dataset

The dataset is publicly available on Kaggle:

`kaggle.com/datasets/salonipandagale/solar-panel-defect-classification-dl-project`

The dataset contains **757 labeled images** belonging to six classes.

### Classes

| Class             | Description                                |
| ----------------- | ------------------------------------------ |
| Bird-drop         | Solar panels affected by bird droppings    |
| Clean             | Clean solar-panel surface                  |
| Dusty             | Solar panels affected by dust accumulation |
| Electrical-damage | Images showing electrical damage           |
| Physical-Damage   | Images showing physical damage             |
| Snow-Covered      | Solar panels covered by snow               |

### Dataset Split

An **80:20 train-validation split** was created using `image_dataset_from_directory`.

| Set        | Images |
| ---------- | -----: |
| Training   |    606 |
| Validation |    151 |
| Total      |    757 |

No separate test set was used in this notebook. Therefore, the reported evaluation results should be interpreted as **validation-set results**, not test-set performance.

---

## 3. Data Preprocessing

The following preprocessing steps were used:

* Images resized to **224 × 224 pixels**
* Batch size of **32**
* Fixed random seed of **42** for reproducibility
* 80:20 training-validation split
* Image rescaling where used in the CNN and MobileNetV2 pipelines

Data augmentation was also used to improve generalization:

* Random horizontal flipping
* Random rotation
* Random zoom

---

## 4. Model Development

The project follows an experimental progression from a custom CNN to transfer learning.

### 4.1 Baseline CNN

A custom CNN was first developed as a baseline.

The architecture consists of:

* 3 convolutional layers
* Max pooling after each convolutional block
* Flatten layer
* Dense layer with 128 units
* Softmax output layer for six-class classification

The baseline model was trained using:

* Adam optimizer
* Sparse categorical cross-entropy loss
* Accuracy as the evaluation metric

The baseline provided a reference point for comparing more advanced approaches.

---

### 4.2 Improved CNN

A more regularized CNN was then experimented with using:

* Data augmentation
* Batch Normalization
* Dropout
* L2 regularization
* Multiple convolutional blocks
* Early Stopping
* ReduceLROnPlateau

This experiment showed that simply increasing CNN complexity did not necessarily lead to better validation performance. The training run exhibited poor validation generalization, motivating the exploration of pretrained architectures.

---

## 5. Transfer Learning

To improve feature extraction and generalization, pretrained CNN architectures were explored.

### 5.1 MobileNetV2

MobileNetV2 pretrained on **ImageNet** was used as a feature extractor.

The original ImageNet classification head was removed using:

```python
include_top=False
```

The pretrained backbone was frozen:

```python
base_model.trainable = False
```

A task-specific classification head was added consisting of:

* Global Average Pooling
* Dense layer with 128 units
* Six-class Softmax output

The model achieved a best validation accuracy of approximately **76.27%**.

---

### 5.2 EfficientNetB0

EfficientNetB0 pretrained on **ImageNet** was then explored.

Similar to MobileNetV2:

* The ImageNet classification head was removed.
* The pretrained EfficientNetB0 backbone was frozen.
* A custom classification head was added.
* Global Average Pooling was used to convert feature maps into a compact feature vector.
* A Dense layer with 128 units was added.
* A six-class Softmax layer produced the final predictions.

EfficientNetB0 achieved a best validation accuracy of approximately **81.36%**, which was higher than the MobileNetV2 experiment.

Therefore, EfficientNetB0 was selected for further optimization.

---

## 6. Class Imbalance Handling

The class distribution was examined before hyperparameter tuning.

Class weights were calculated using:

```python
class_weight =
    total_images / (number_of_classes × class_count)
```

The calculated class weights were:

```text
Class 0: 0.7530
Class 1: 0.7491
Class 2: 0.7649
Class 3: 1.4110
Class 4: 2.1063
Class 5: 1.1816
```

The variation in these weights indicates that the six classes are not equally represented.

Classes with fewer training examples receive higher weights, giving their samples a greater contribution to the training loss.

The calculated class weights were passed to the **Keras Tuner search** using:

```python
class_weight=class_weights
```

This was done to reduce the influence of class imbalance during model training.

---

## 7. Hyperparameter Optimization

Keras Tuner was used to optimize the EfficientNetB0-based model using **Random Search**.

A total of **20 trials** were performed.

The following hyperparameters were tuned:

| Hyperparameter  | Search Range |
| --------------- | ------------ |
| Rotation factor | 0.05 – 0.30  |
| Zoom factor     | 0.05 – 0.30  |
| Dropout rate    | 0.0 – 0.5    |
| Dense units     | 64 – 512     |
| Learning rate   | 1e-4 – 1e-2  |

The EfficientNetB0 backbone remained frozen during the tuning process.

### Best Hyperparameters

The best configuration found by Keras Tuner was:

| Hyperparameter  | Best Value |
| --------------- | ---------: |
| Rotation factor |       0.05 |
| Zoom factor     |       0.05 |
| Dropout rate    |        0.4 |
| Dense units     |        128 |
| Learning rate   |    0.00291 |

The best trial achieved a validation accuracy of approximately **83.05%**.

The complete tuning process took approximately **6 hours 49 minutes** for 20 trials.

---

## 8. Results

The main experimental results are summarized below:

| Approach             | Best Validation Accuracy |
| -------------------- | -----------------------: |
| Baseline CNN         |                     ~61% |
| MobileNetV2          |                   76.27% |
| EfficientNetB0       |                   81.36% |
| Tuned EfficientNetB0 |               **83.05%** |

The tuned EfficientNetB0 configuration achieved the highest validation accuracy among the approaches experimented with in this project.

A subsequent evaluation of the selected model on the same validation dataset produced approximately **85.78% accuracy**.

> **Note:** Since the notebook does not contain a separate held-out test set, these values represent validation-set performance and should not be described as test accuracy.

---

## 9. Final Model

The selected tuned model was saved as:

```text
solar_model.keras
```

The final pipeline can be summarized as:

```text
Input Solar Panel Image
          ↓
      224 × 224
          ↓
    Data Augmentation
          ↓
   Pretrained EfficientNetB0
      (ImageNet Weights)
          ↓
    Frozen Feature Extractor
          ↓
Global Average Pooling
          ↓
   Dense Layer (128)
          ↓
    Softmax (6 Classes)
          ↓
   Predicted Class
```

---

## 10. Technologies Used

* Python
* TensorFlow / Keras
* EfficientNetB0
* MobileNetV2
* Keras Tuner
* NumPy
* Matplotlib
* Streamlit
* Git / GitHub
* Render

---

## 11. Key Learning Outcomes

Through this project, I explored:

* Building CNNs for image classification
* Data augmentation for improving generalization
* Batch Normalization and Dropout
* L2 regularization
* Early Stopping and learning-rate scheduling
* Transfer learning using ImageNet-pretrained models
* Feature extraction using MobileNetV2 and EfficientNetB0
* Handling class imbalance using class weights
* Hyperparameter optimization using Keras Tuner
* Deploying a deep learning classification model using Streamlit and Render

---

## 12. Deployment

The trained model was integrated into a Streamlit application and deployed using Render.

The application allows users to provide a solar-panel image and obtain a predicted class from the six categories.

**Live Demo:** solar-panel-defect-classification-1.onrender.com

---

## 13. Project Structure

```text
Solar-Panel-Defect-Classification/
│
├── Solar_Panel_Classification.ipynb
├── solar_model.keras
├── app.py
├── requirements.txt
├── README.md
└── ...
```

---

## 14. Author

**Saloni Pandagale**

B.Tech — Metallurgical & Materials Engineering
IIT Bhubaneswar
