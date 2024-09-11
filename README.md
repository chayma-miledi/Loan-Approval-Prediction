# Loan Approval Prediction

This project presents an in-depth analysis and predictive modeling for loan approval evaluation. Using a dataset comprising various financial and personal characteristics of loan applicants, the objective is to develop machine learning models that can accurately predict whether a loan application will be approved or rejected.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)
- [Future Work](#future-work)
- [License](#license)

## Introduction
The primary goal of this project is to predict the outcome of loan approval using machine learning techniques. The dataset used contains information such as the applicant's income, CIBIL score, employment status, loan amount, and loan duration.

## Dataset
The dataset used for this project is the **Loan Approval Prediction Dataset** available on Kaggle, contributed by Archit Sharma. It contains various financial and personal information about loan applicants, which is used to determine their eligibility for a loan.

Link to the dataset: [Loan Approval Dataset](https://www.kaggle.com/datasets/architsharma/loan-approval-dataset)

The target variable is `loan_status`, indicating whether a loan is approved or rejected.

## Data Exploration and Preprocessing
- **Data Loading and Inspection**: The dataset was inspected for missing values, outliers, and data quality issues.
- **Preprocessing Steps**:
  - Handled missing values and erroneous data entries.
  - Encoded categorical variables.
  - Applied feature scaling to normalize numerical values.
  - Removed unnecessary spaces in column names and handled negative values in financial fields like `residential_assets_value`.

Key libraries used:
- **Pandas**: Data manipulation and cleaning.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn**: Model building and evaluation.
- **XGBoost**: Advanced gradient boosting algorithm for performance.
- **Statsmodels**: Statistical model estimation.

## Machine Learning Models
The dataset was split into training and testing sets. Multiple machine learning models were used to predict loan approvals, including:

1. **Random Forest Classifier**: Captures complex relationships in mixed datasets.
2. **XGBoost**: High-performance gradient boosting algorithm.
3. **Logistic Regression**: Provides insight into the individual contribution of features.

### Model Optimization
- Hyperparameters were tuned using **GridSearchCV** to improve model performance.

## Results
Each model was evaluated using various performance metrics such as accuracy, precision, recall, and the area under the ROC curve (AUC). Confusion matrices were used to illustrate the performance of each model.

- **Random Forest**:
  - High accuracy and balanced performance in predicting both approvals and rejections.

- **XGBoost**:
  - Slightly better accuracy and AUC compared to Random Forest, making it the most reliable model for this task.

- **Logistic Regression**:
  - Lower accuracy and AUC, prone to higher false positives, making it less suitable for this problem.

## Conclusion
After evaluating the models, **XGBoost** emerged as the best-performing model due to its high accuracy and ability to generalize well on unseen data. This project demonstrates the potential of machine learning models in predicting loan approval outcomes based on financial and personal attributes of applicants.
