# Website Classification with Machine Learning

This project implements a machine learning pipeline for classifying websites based on textual features extracted from their content. The solution uses **XGBoost** for classification, and various techniques are applied for model evaluation and interpretation, including **SHAP** values, **Permutation Importance**, and **Partial Dependence Plots**.

## Features

- **Data Preprocessing**: 
  - Cleans the website text and extracts useful features like word count, sentiment, and special character count.
  - Extracts additional features such as URL length and subdomain count.
- **Model Training**: 
  - Uses **XGBoost** to classify websites into categories.
  - Trains and evaluates the model with a train-validation split of the data.
- **Model Evaluation**:
  - Provides performance metrics such as accuracy, recall, precision, F1 score, and confusion matrix.
- **Model Interpretability**:
  - Visualizes feature importance using **Permutation Importance** and **Partial Dependence Plots**.
  - Uses **SHAP** values to explain the model’s decisions.

## Requirements

To run this project, you need to have Python 3.x and the following libraries installed:

- `pandas`
- `xgboost`
- `shap`
- `eli5`
- `textblob`
- `sklearn`
- `seaborn`
- `matplotlib`

You can install the required libraries by running:

```bash
pip install -r requirements.txt
