# USC-CSCI-467-FINAL_PROJECT
This repository contains the code and resources used to develop an ovarian cancer subtype classifier using machine learning and deep learning techniques.
# Instructions for Training and Running Models

This project contains three subfolders with different models for training and testing. Below is a description of the folders and the necessary steps to execute the corresponding scripts.

## Project Structure

The project is divided into the following subfolders:

- **BASELINE_MODEL**: Contains the base model for training using an SVM classifier.
- **DENSENET201**: Contains the multi-class classification model based on DenseNet201.
- **INCEPTION_V3**: Contains the CNN model based on InceptionV3 architecture.

## General Steps for Running the Models

### 1. BASELINE_MODEL

This folder contains the code for training the baseline model using an SVM classifier.

- **Main File**: `svm.py`
- **Steps**:
  1. Make sure you have downloaded the required data and adjust the image paths in the code.
  2. Run the script `svm.py` to train the baseline model.

### 2. INCEPTION_V3

This folder contains the CNN model based on the InceptionV3 architecture.

- **Main File**: `CNN-InceptionV3.py`
- **Steps**:
  1. Make sure the image paths are correctly set in the code.
  2. Run the script `CNN-InceptionV3.py` to train the InceptionV3 model.

### 3. DENSENET201

This folder contains the multi-class classification model based on DenseNet201.

- **Main Files**:
  - `DENSENET.py`: To train the model.
  - `DENSENET-TEST.py`: To run tests using a pre-trained model.
- **Steps**:
  1. Ensure the image paths are correctly set in the `DENSENET.py` file before training the model.
  2. In the same folder, you will find a `.keras` file with the pre-trained model. To run tests with this model, simply execute the `DENSENET-TEST.py` script.
  3. Make sure to adjust the image paths in the `DENSENET-TEST.py` file before running it.
### 4. DATASET
#### Data for Training and Testing SVM and InceptionV3
This dataset was used for training and testing the SVM and InceptionV3 models. It contains pre-processed and labeled images to evaluate the performance of these models.

- [Download Data (SVM and InceptionV3)](https://shorturl.at/zKdNS)

#### Data for Training and Testing DenseNet201
This dataset is used for training and testing the DenseNet201 model. It consists of three categories of images for classification, allowing the model to differentiate between these classes.

- [Download Data (DenseNet201)](https://shorturl.at/WewfD)

## Requirements

Before running the scripts, make sure you have the necessary dependencies installed. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt

