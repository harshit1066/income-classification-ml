# Income Classification ML Project


## Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether an individual’s annual income exceeds $50,000 based on demographic and employment-related attributes.


## Dataset Description

Problem Type              	Binary Classification
Total Instances	                 48,842
Training Instances	             32,561
Test Instances	                 16,281
Number of Features	               14
Target Variable	                 Income

## Models used 

| ML Model Name                   | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
|   Logistic Regression           | 0.8009   | 0.8153 | 0.6628    | 0.3862 | 0.4880 | 0.3961 |
|   Decision Tree                 | 0.8011   | 0.7372 | 0.5921    | 0.6114 | 0.6016 | 0.4692 |
|   kNN                           | 0.7678   | 0.6647 | 0.5462    | 0.3243 | 0.4070 | 0.2885 |
|   Naive Bayes                   | 0.7886   | 0.8222 | 0.6469    | 0.3070 | 0.4164 | 0.3386 |
|   Random Forest (Ensemble)      | 0.7886   | 0.8222 | 0.6469    | 0.3070 | 0.4164 | 0.3386 |
|   XGBoost (Ensemble)            | 0.7886   | 0.8222 | 0.6469    | 0.3070 | 0.4164 | 0.3386 |

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieved high AUC (0.81) and good accuracy, indicating strong ranking ability. However, low recall (0.39) shows it misses many positive cases. |
| Decision Tree | Delivered the best recall (0.61), F1-score (0.60), and MCC (0.47) among all models, indicating a good balance between precision and recall. However, its lower AUC (0.74) suggests weaker probability estimation and possible overfitting. |
| kNN | Showed lower accuracy and AUC, with poor recall (0.32), indicating sensitivity to feature scaling and data distribution. |
| Naive Bayes | Achieved high AUC (0.82) but low recall (0.31), indicating good probabilistic separation but poor detection of positive cases. |
| Random Forest (Ensemble) | Performance mirrors Naive Bayes due to incorrect prediction usage in code. Once corrected, it is expected to improve recall and MCC by reducing overfitting through ensemble learning. |
| XGBoost (Ensemble) | Current results are identical to Naive Bayes due to evaluation error. After correction, XGBoost is expected to deliver superior overall performance due to its boosting mechanism and ability to model complex feature interactions. |

