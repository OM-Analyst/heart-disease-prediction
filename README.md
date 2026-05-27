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
│   └── heart_disease.csv
│
├── outputs/
│   └── step_1_data_overview/
│
├── src/
│   └── step_1_data_overview.py
│
├── requirements.txt
└── README.md