# Solar Panel Defect Detection using Deep Learning

### Live demo : https://solar-panel-defect-classification-1.onrender.com/

## 1. Introduction 
This project focuses on detecting defects in solar panels using deep learning techniques. Identifying defects early is crucial for maintaining efficiency and reducing operational costs in solar energy systems.

The project explores multiple approaches including Convolutional Neural Networks (CNNs), transfer learning, and hyperparameter optimization to achieve improved performance.

## 2. Dataset

The dataset used in this project is publicly available on Kaggle :
https://www.kaggle.com/datasets/salonipandagale/solar-panel-defect-classification-dl-project

It consists of labeled images of solar panels categorized into different defect classes.


## 3. Methodology

### 3.1 Data Preprocessin
* Image resizing to 224 × 224
* Normalization using rescaling
* Train-validation split (80:20)

### 3.2 Baseline Model (CNN)
* Multiple Conv2D + MaxPooling layers
* Fully connected dense layers
* Softmax output for classification

### 3.3 Overfitting Mitigation
* Dropout layers
* Batch Normalization
* Early Stopping
* Learning rate scheduling

### 3.4 Data Augmentation
* Random Flip
* Random Rotation
* Random Zoom

### 3.5 Transfer Learning

Two pretrained architectures were used:
* MobileNetV2
* EfficientNetB0

These models were fine-tuned for improved generalization.


### 3.6 Hyperparameter Optimization

* Implemented using Keras Tuner
* Tuned parameters:
* Learning rate
* Dropout rate
* Dense layer units
* Augmentation factors


## 4. Results

| Model           | Training Accuracy | Validation Accuracy |

| --------------- | ----------------- | ------------------- |

| CNN (Baseline)  | \~0.99             | \~0.61               |

| Regularized CNN | Improved          | Improved            |

| MobileNetV2     | Higher            | Moderate            |

| EfficientNetB0  | \~0.94             | \~0.81               |

EfficientNetB0 provided the best performance among all models.

## 6.Author 
Saloni Pandagale


