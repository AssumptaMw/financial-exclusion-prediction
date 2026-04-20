# Predicting Financial Exclusion in Kenya (Kenya - FinAccess 2024)

## Overview

This project uses machine learning to predict **financial exclusion in Kenya** using the FinAccess 2024 dataset.

It combines data preprocessing, classification models, and a Streamlit web application for interactive predictions.

---

## Objectives

* Predict whether an individual is financially excluded
* Compare multiple machine learning models
* Handle class imbalance using SMOTE
* Provide model explainability using SHAP
* Deploy an interactive app using Streamlit

---

## Models Used

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

---

## Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost, LightGBM
* SHAP (Explainability)
* Streamlit (Web App)

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd financial-exclusion-prediction
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## Dataset

* Source: FinAccess 2024 Survey (Kenya)
* Contains demographic and financial behavior indicators

---

## Features

* Data preprocessing and feature engineering
* Class imbalance handling (SMOTE)
* Model comparison and evaluation
* SHAP-based model explainability
* Interactive prediction interface

---

## Future Improvements

* Deploy to Streamlit Cloud
* Add real-time data input forms
* Improve model performance with hyperparameter tuning
