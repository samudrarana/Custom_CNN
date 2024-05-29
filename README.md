# Deep Learning Based Diagnosis of Tuberculosis Using X-Ray Images

This repository contains the code and resources for the project "Deep Learning Based Diagnosis of Tuberculosis Using X-Ray Images". We utilized the [Tuberculosis (TB) Chest X-ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) to train and evaluate various pre-trained models, including our Custom_CNN model.

## Overview

We experimented with several pre-trained models like VGG16, VGG19, InceptionV3, and ResNet50 etc. Among these, VGG16 provided the best validation accuracy. However, our custom CNN model outperformed all pre-trained models in terms of both accuracy and training time.

## Dataset

The dataset used in this project can be found [here](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset). It consists of X-ray images categorized into Tuberculosis and Normal classes.

## Models and Results

### Pre-trained Models

We trained the following pre-trained models on the dataset:
- **VGG16**
- **VGG19**
- **InceptionV3**
- **ResNet50**

### Custom CNN Model

We designed a custom CNN model that achieved higher validation accuracy compared to the pre-trained models. The architecture of our custom model is illustrated below.

#### Custom CNN Architecture
![Custom CNN Architecture](https://github.com/samudrarana/Custom_CNN/blob/main/Custom_CNN_Architecture.jpeg)

### Validation Accuracy Comparison

Here are the validation accuracies of the different models:

| Model            | Validation Accuracy |
|------------------|----------------------|
| VGG16            | 0.961905             |
| VGG19            | 0.928571             |
| InceptionV3      | 0.938095             |
| ResNet50         | 0.823810             |
| **Custom CNN**   | **0.988095**         |

![Validation Accuracy of Different Models](https://github.com/samudrarana/Custom_CNN/blob/main/Comparison_Analysis_Bar_Graph.png)

![Model Accuracies](https://github.com/samudrarana/Custom_CNN/blob/main/Model_Accuracies.png)

## Custom CNN Model Code
Here is the code for our [Custom CNN model](https://github.com/samudrarana/Custom_CNN/blob/main/Custom_CNN_Model.py).
