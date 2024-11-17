# Smart Grid Stability Prediction

## Project Overview

The Smart Grid Stability Prediction project aims to predict the stability of smart grids using various machine learning algorithms. The project leverages different supervised learning models to classify whether the smart grid will remain stable or unstable based on various input features. 

The goal of the project is to evaluate the performance of these algorithms in predicting grid stability and to understand their strengths and weaknesses when applied to real-world data.

## Problem Definition

In the context of smart grids, predicting stability is crucial for ensuring efficient and reliable power distribution. The instability of smart grids can lead to inefficiencies, power outages, and potential system failures. The goal of this project is to build a model that can predict whether a smart grid will remain stable or not based on various input parameters.

### Data Description

The dataset contains several features representing different parameters that affect smart grid stability, including:

- **tau1, tau2, tau3, tau4**: Reaction times of the energy producer and consumers.
- **p1, p2, p3, p4**: Power balance of the energy producer and consumers.
- **g1, g2**: Price elasticity coefficients of the energy producer and consumers.

These features are used to predict the stability of the grid, with the target variable indicating whether the grid is stable (label `1`) or unstable (label `0`).

## Algorithms Used

We employed the following machine learning algorithms for classification in this project:

1. **Logistic Regression**: A simple yet effective model used for binary classification.
2. **K-Nearest Neighbors (KNN)**: A non-parametric algorithm that classifies instances based on the majority class of the k-nearest neighbors.
3. **Support Vector Machines (SVM)**:
   - **Linear Kernel**: SVM with a linear kernel.
   - **Polynomial Kernel**: SVM with a polynomial kernel.
   - **Radial Basis Function (RBF) Kernel**: SVM with the RBF kernel.

### Evaluation Metrics

To evaluate the performance of the models, we used several metrics, which are widely accepted in the machine learning community for classification problems:

- **Accuracy**: The proportion of true results (both true positives and true negatives) among the total number of cases examined.
  
- **Sensitivity (Recall)**: The proportion of actual positives that are correctly identified by the model. In this case, it indicates the model's ability to correctly identify stable grids.
  
- **Specificity**: The proportion of actual negatives that are correctly identified. Here, it indicates the model's ability to identify unstable grids.
  
- **F1-Score**: The harmonic mean of precision and recall, providing a single metric that balances both concerns.
  
- **Confusion Matrix**: A table used to describe the performance of a classification algorithm, showing true positives, true negatives, false positives, and false negatives.
  
- **ROC-AUC Curve**: A graphical representation of the model's performance, showing the trade-off between sensitivity and specificity for different thresholds.

### Dataset Preprocessing

Before training the models, the dataset underwent several preprocessing steps:

1. **Missing Data Handling**: Missing values were imputed using the mean for numerical columns.
2. **Feature Scaling**: Features were normalized or standardized to ensure the algorithms perform efficiently and converge faster.
3. **Train-Test Split**: The dataset was split into training and testing sets, with 80% for training and 20% for testing.
4. **Encoding Categorical Data**: If any categorical features were present, they were encoded using appropriate methods such as One-Hot Encoding or Label Encoding.

## Model Training and Evaluation

The models were trained on the training data, and their performance was evaluated on the test data using the aforementioned metrics. Here's a summary of the training process:

1. **Model Training**:
   - Models were trained using various algorithms like Logistic Regression, KNN, and SVM with different kernel types.
   - Hyperparameters were tuned using techniques like Grid Search or Random Search (where applicable).

2. **Early Stopping**: 
   - For certain models, early stopping was applied to prevent overfitting and improve generalization by halting training once the validation loss stopped improving.

3. **Model Evaluation**:
   - After training, models were evaluated using the test set, and the following evaluation metrics were calculated:
     - **Confusion Matrix**: For visualizing the true positive, true negative, false positive, and false negative counts.
     - **ROC-AUC Curve**: For assessing the model's ability to distinguish between stable and unstable grids.
     - **Precision, Recall, F1-Score**: For a more detailed evaluation of model performance on imbalanced datasets.
