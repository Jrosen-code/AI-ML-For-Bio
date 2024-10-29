# Logistic Regression Model for Protein Pathology Prediction

This project demonstrates how to create a logistic regression classifier to predict pathology conditions using synthetic protein sequence data. Key functionalities include data generation, model training, evaluation with metrics, and visualization of model performance through an ROC curve and confusion matrix. This is a WIP so expect further updates. 

## Table of Contents

- [Overview](#overview)
- [Dataset Generation](#dataset-generation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Usage](#usage)
- [Installation and Requirements](#installation-and-requirements)
- [Code Structure](#code-structure)

## Overview

This project creates synthetic data representing patients' protein sequences and their associated pathology classification. The model uses logistic regression to predict pathology labels, where performance is evaluated using the model accuracy, ROC curve, and confusion matrix.

## Dataset Generation

A synthetic dataset of protein sequences is generated with the following features:
- **Protein Sequences**: Randomly generated sequences of amino acids.
- **Pathology Label**: Indicates the pathology condition of the patient as "PD" or "PRCR".
- **Random Count Values**: Assigned to each protein sequence for each patient to simulate clinical data.

The generated dataset is stored in a Pandas DataFrame, which is printed for verification.

## Model Training and Evaluation

The project employs logistic regression for binary classification, using the following steps:
1. **Label Encoding**: Converts pathology labels to binary form.
2. **Logistic Regression Model**: Fits the model on the protein count data to predict pathology labels.
3. **Performance Metrics**:
   - **Accuracy**: Reports the model's prediction accuracy on the training data.
   - **ROC Curve**: Displays the model's performance in classifying positive cases.
   - **Confusion Matrix**: Visualizes model performance by comparing true and predicted labels.

## Usage

1. **Run the Code**:
   The code performs the following tasks:
   - Generates and displays synthetic protein sequence data.
   - Trains a logistic regression model on the generated dataset.
   - Outputs model accuracy, ROC curve, and confusion matrix.

2. **Functionality**:
   - **`generate_random_protein_seq(length)`**: Generates synthetic protein sequences.
   - **`gen_patient_dataset()`**: Creates a dataset of patients with protein counts and pathology labels.
   - **`ROC_curve_create(model, X, y)`**: Generates and displays the ROC curve.
   - **`Confusion_matrix_create(model, X, y, display_labels)`**: Displays the confusion matrix.

## Installation and Requirements

Ensure you have the required libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn
