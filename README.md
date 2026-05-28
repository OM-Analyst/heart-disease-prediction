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
│   └── heart_disease_cleaned.csv
│
├── outputs/
│   ├── step_1_data_overview/
│   ├── step_2_data_cleaning/
│   └── step_3_eda/
│
├── src/
│   ├── step_1_data_overview.py
│   ├── step_2_data_cleaning.py
│   └── step_3_eda.py
│
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

## Current Progress

- [x] Step 1: Data overview
- [x] Step 2: Data cleaning and duplicate handling
- [x] Step 3: Exploratory data analysis
- [ ] Step 4: Feature preparation
- [ ] Step 5: Model training
- [ ] Step 6: Model evaluation
- [ ] Step 7: Model interpretation
- [ ] Step 8: Health informatics dashboard

## Health Informatics Relevance

This project is relevant to health informatics because it applies data analytics and machine learning to clinical data. The final system can support early risk identification and help healthcare professionals make more informed decisions.