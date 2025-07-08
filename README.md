# ML-Classification_and_Regression

This repository contains a machine learning assignment project completed on Kaggle. It includes both a classification and a regression task using real-world datasets. The implementation demonstrates data preprocessing, model training, hyperparameter tuning, evaluation, and comparison of different models.

---

## 📌 Assignment Objectives

- Understand the difference between **classification** and **regression** tasks.
- Learn how to split data into training, validation, and testing sets.
- Apply and evaluate different machine learning models:
  - K-Nearest Neighbors (KNN)
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
- Measure performance using metrics like accuracy, precision, recall, F1-score, MSE, MAE.
- Gain experience using ML libraries: **Pandas**, **NumPy**, **Scikit-Learn**, **Seaborn**, **Matplotlib**.

---

## 🧠 Classification Task: MAGIC Gamma Telescope

- **Dataset**: MAGIC gamma telescope dataset
- **Problem**: Binary classification – distinguish between Gamma and Hadron particles
- **Model**: K-Nearest Neighbors (KNN)
- **Steps**:
  - Balanced the imbalanced classes (Gamma vs Hadron)
  - Split the dataset into Train (70%), Validation (15%), Test (15%)
  - Scaled features using `StandardScaler`
  - Compared:
    - KNN with validation split
    - KNN with 10-fold cross-validation
  - Evaluated using:
    - Accuracy
    - Confusion matrix
    - Classification report (Precision, Recall, F1-score)

📁 **Notebook**: `KNN_Classification.ipynb`

---

## 🏡 Regression Task: California Housing Prices

- **Dataset**: California housing dataset (1990 census-based)
- **Problem**: Predict median house values using multiple features
- **Models**:
  - Linear Regression
  - Lasso Regression
  - Ridge Regression (with cross-validation for alpha)
- **Steps**:
  - Preprocessed and scaled features
  - Split data: Train (70%), Validation (15%), Test (15%)
  - Compared models using:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
- **Model Comparison**:
  - Linear Regression: High sensitivity to noise, best test results
  - Lasso: Performs feature selection, useful with redundant features
  - Ridge: Regularization helps avoid overfitting, but may underperform with noisy data

📁 **Notebook**: `Regression.ipynb`

---

## 📊 Evaluation Summary

| Task           | Model            | Validation MSE | Test MSE | Notes |
|----------------|------------------|----------------|----------|-------|
| Classification | KNN (k=7, CV)    | ~0.81 accuracy | 0.8042   | Balanced, good generalization |
| Regression     | Linear           | ↓ Lowest       | ↓ Lowest | Best performance |
| Regression     | Lasso            | Moderate       | Moderate | Some feature elimination |
| Regression     | Ridge            | ↑ Highest      | ↑ Highest | More affected by noise |

---

## 🧰 Libraries Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/KarenSamehS/ML-Classification_and_Regression.git
