# Heart Disease Prediction Using Machine Learning

## Project Overview

This project focuses on predicting the presence of heart disease using patient clinical and diagnostic data. The goal is to build a health informatics decision-support system that can help identify patients who may be at risk of heart disease based on measurable health indicators.

The dataset contains patient information such as age, sex, chest pain type, resting blood pressure, cholesterol level, fasting blood sugar, ECG results, maximum heart rate, exercise-induced angina, ST depression, and other clinical features.

## Objective

The main objective of this project is to use machine learning to predict whether a patient has heart disease.

The target variable is:

- `target = 0`: No heart disease
- `target = 1`: Heart disease present

## Project Structure

```text
heart-disease-prediction/
│
├── data/
│   ├── heart_disease.csv
│   ├── heart_disease_cleaned.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── scaler.pkl
│       └── feature_names.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── best_model.pkl
│
├── outputs/
│   ├── step_1_data_overview/
│   ├── step_2_data_cleaning/
│   ├── step_3_eda/
│   ├── step_4_feature_preparation/
│   ├── step_5_model_training/
│   ├── step_6_model_evaluation/
│   └── step_7_model_interpretation/
│
├── src/
│   ├── step_1_data_overview.py
│   ├── step_2_data_cleaning.py
│   ├── step_3_eda.py
│   ├── step_4_feature_preparation.py
│   ├── step_5_model_training.py
│   ├── step_6_model_evaluation.py
│   └── step_7_model_interpretation.py
│
├── .gitignore
├── requirements.txt
└── README.md
```
## Dataset Summary

The original dataset contains:

- 1,025 rows
- 14 columns
- No missing values
- A balanced target variable

During the data overview stage, 723 duplicate records were identified. These duplicate records were removed during data cleaning, leaving 302 unique patient records for analysis and modelling.

Removing duplicates is important because repeated patient records can cause the machine learning model to overestimate its performance. By using only unique patient records, the project produces a more reliable and realistic heart disease prediction system.

## Step 1: Data Overview

The first step loads the dataset and checks:

- Dataset shape
- Column names
- First few records
- Missing values
- Duplicate rows
- Target distribution
- Summary statistics

## Step 2: Data Cleaning

The second step removes duplicate patient records from the dataset and saves a cleaned version of the data.

The cleaned dataset is saved as:

```text
data/heart_disease_cleaned.csv
```

The cleaning report is saved in:

```text
outputs/step_2_data_cleaning/
```

## How to Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run Step 1:

```bash
python src/step_1_data_overview.py
```

Run Step 2:

```bash
python src/step_2_data_cleaning.py
```
## Step 3: Exploratory Data Analysis

The third step explores the cleaned heart disease dataset using visual charts and summary tables. Exploratory Data Analysis helps identify patterns, trends, and relationships between patient health indicators and the presence of heart disease.

This step analyzes important clinical features such as age, sex, chest pain type, cholesterol level, resting blood pressure, maximum heart rate, and exercise-induced angina.

The EDA process includes:

- Heart disease distribution
- Age distribution of patients
- Heart disease distribution by sex
- Heart disease distribution by chest pain type
- Cholesterol distribution
- Resting blood pressure distribution
- Maximum heart rate distribution
- Exercise-induced angina compared with heart disease status
- Correlation heatmap of numerical and encoded features

The charts and summary reports are saved in:

```text
outputs/step_3_eda/
```

## Step 4: Feature Preparation

The fourth step prepares the cleaned dataset for machine learning. The target variable, `target`, is separated from the input features. The data is then split into training and testing sets using an 80/20 split.

A stratified split is used to preserve the balance between patients with heart disease and patients without heart disease in both the training and testing datasets.

Numerical features such as age, resting blood pressure, cholesterol, maximum heart rate, and ST depression are scaled using StandardScaler. The processed training and testing files are saved in the `data/processed/` folder.

## Step 5: Model Training

The fifth step trains multiple machine learning models to predict whether a patient has heart disease.

The models trained include Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. These models are trained using the prepared training dataset from Step 4 and evaluated using the testing dataset.

The models are compared using accuracy, precision, recall, F1-score, and ROC-AUC. Recall is especially important in this health informatics project because missing a patient who actually has heart disease can be risky.

The best model is selected based on recall and saved for later evaluation and interpretation.

## Step 6: Model Evaluation

The sixth step evaluates the best machine learning model using the testing dataset. This step checks how well the selected model performs on data it has not seen during training.

The model is evaluated using accuracy, precision, recall, F1-score, ROC-AUC, a confusion matrix, and a classification report.

The confusion matrix shows how many patients were correctly and incorrectly classified as having heart disease or not having heart disease. The ROC curve shows how well the model separates patients with heart disease from patients without heart disease.

In this health informatics project, recall is especially important because it measures how well the model identifies patients who actually have heart disease. A high recall helps reduce the risk of missing patients who may need further medical attention.

The evaluation outputs are saved in:

```text
outputs/step_6_model_evaluation/
```

## Step 7: Model Interpretation

The seventh step explains which patient health factors had the strongest influence on the model's heart disease predictions.

Permutation importance is used to measure how important each feature is. This method works by shuffling one feature at a time and checking how much the model's performance changes. If the model performs worse after a feature is shuffled, that feature is considered important.

This step helps make the heart disease prediction model more explainable. In a health informatics setting, explainability is important because healthcare professionals need to understand which clinical factors contribute most to a patient's predicted heart disease risk.

The interpretation outputs are saved in:

```text
outputs/step_7_model_interpretation/
```
## Current Progress

- [x] Step 1: Data overview
- [x] Step 2: Data cleaning and duplicate handling
- [x] Step 3: Exploratory data analysis
- [x] Step 4: Feature preparation
- [x] Step 5: Model training
- [x] Step 6: Model evaluation
- [x] Step 7: Model interpretation
- [ ] Step 8: Health informatics dashboard

## Health Informatics Relevance

This project is relevant to health informatics because it applies data analytics and machine learning to clinical data. The final system can support early risk identification and help healthcare professionals make more informed decisions.