import pandas as pd
import streamlit as st
import joblib
from pathlib import Path


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide"
)


# -----------------------------
# File Paths
# -----------------------------
DATA_PATH = Path("data/heart_disease_cleaned.csv")
MODEL_PATH = Path("models/best_model.pkl")
SCALER_PATH = Path("data/processed/scaler.pkl")
METRICS_PATH = Path("outputs/step_6_model_evaluation/evaluation_metrics.csv")
FEATURE_IMPORTANCE_PATH = Path("outputs/step_7_model_interpretation/permutation_feature_importance.csv")
CONFUSION_MATRIX_PATH = Path("outputs/step_6_model_evaluation/confusion_matrix.png")
ROC_CURVE_PATH = Path("outputs/step_6_model_evaluation/roc_curve.png")
FEATURE_IMPORTANCE_IMAGE_PATH = Path("outputs/step_7_model_interpretation/top_10_feature_importance.png")


# -----------------------------
# Load Resources
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None


@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return None


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


df = load_data()
model = load_model()
scaler = load_scaler()
metrics_df = load_metrics()
feature_importance_df = load_feature_importance()


# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Data Summary",
        "Model Performance",
        "Feature Importance",
        "Patient Risk Prediction"
    ]
)


# -----------------------------
# Project Overview Page
# -----------------------------
if page == "Project Overview":
    st.title("❤️ Heart Disease Prediction Dashboard")

    st.write(
        """
        This dashboard is a health informatics machine learning application for predicting
        whether a patient may have heart disease based on clinical and diagnostic features.
        """
    )

    st.subheader("Project Objective")

    st.write(
        """
        The goal of this project is to support early heart disease risk identification
        using patient health information such as age, cholesterol, blood pressure,
        chest pain type, maximum heart rate, and exercise-induced angina.
        """
    )

    st.subheader("Health Informatics Relevance")

    st.write(
        """
        This project is relevant to health informatics because it uses clinical data,
        machine learning, and model interpretation to support healthcare decision-making.
        The dashboard is not a replacement for a medical professional, but it shows how
        data analytics can help identify patients who may need further screening.
        """
    )

    st.warning(
        "Disclaimer: This dashboard is for educational purposes only and should not be used as medical advice."
    )


# -----------------------------
# Data Summary Page
# -----------------------------
elif page == "Data Summary":
    st.title("Data Summary")

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Target Column", "target")

    st.subheader("Heart Disease Distribution")

    target_counts = df["target"].value_counts().rename(index={
        0: "No Heart Disease",
        1: "Heart Disease"
    })

    st.bar_chart(target_counts)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())


# -----------------------------
# Model Performance Page
# -----------------------------
elif page == "Model Performance":
    st.title("Model Performance")

    st.write(
        """
        This page shows how the selected best model performed on the testing dataset.
        In this health informatics project, recall is especially important because
        it shows how well the model identifies patients who actually have heart disease.
        """
    )

    if metrics_df is not None:
        st.subheader("Evaluation Metrics")

        metric_cols = st.columns(len(metrics_df))

        for index, row in metrics_df.iterrows():
            with metric_cols[index]:
                st.metric(row["Metric"], round(row["Score"], 4))

        st.dataframe(metrics_df)
    else:
        st.warning("Evaluation metrics file not found.")

    st.subheader("Confusion Matrix")

    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH))
    else:
        st.warning("Confusion matrix image not found.")

    st.subheader("ROC Curve")

    if ROC_CURVE_PATH.exists():
        st.image(str(ROC_CURVE_PATH))
    else:
        st.warning("ROC curve image not found.")


# -----------------------------
# Feature Importance Page
# -----------------------------
elif page == "Feature Importance":
    st.title("Feature Importance")

    st.write(
        """
        This page shows which patient health factors were most important in the model's
        heart disease predictions.
        """
    )

    if FEATURE_IMPORTANCE_IMAGE_PATH.exists():
        st.subheader("Top 10 Important Features")
        st.image(str(FEATURE_IMPORTANCE_IMAGE_PATH))

    if feature_importance_df is not None:
        st.subheader("Feature Importance Table")
        st.dataframe(feature_importance_df)
    else:
        st.warning("Feature importance file not found.")


# -----------------------------
# Patient Risk Prediction Page
# -----------------------------
elif page == "Patient Risk Prediction":
    st.title("Patient Risk Prediction")

    st.write(
        """
        Enter patient health values below to generate a heart disease prediction.
        """
    )

    st.warning(
        "This prediction is for educational purposes only and should not be used as medical advice."
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)

        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal Pain",
                3: "Asymptomatic"
            }[x]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=80,
            max_value=250,
            value=130
        )

        chol = st.number_input(
            "Serum Cholesterol",
            min_value=100,
            max_value=700,
            value=240
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        restecg = st.selectbox(
            "Resting ECG Result",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x]
        )

    with col2:
        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=60,
            max_value=250,
            value=150
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        oldpeak = st.number_input(
            "ST Depression",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }[x]
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            options=[0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thalassemia / Thallium Stress Test Result",
            options=[0, 1, 2, 3]
        )

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    numerical_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    input_scaled = input_data.copy()
    input_scaled[numerical_columns] = scaler.transform(input_data[numerical_columns])

    st.subheader("Patient Input Summary")
    st.dataframe(input_data)

    if st.button("Predict Heart Disease Risk"):
        prediction = model.predict(input_scaled)[0]

        if hasattr(model, "predict_proba"):
            prediction_probability = model.predict_proba(input_scaled)[0][1]
        else:
            prediction_probability = None

        if prediction == 1:
            st.error("Prediction: Heart Disease Risk Detected")
        else:
            st.success("Prediction: No Heart Disease Risk Detected")

        if prediction_probability is not None:
            st.write(f"Estimated probability of heart disease: **{prediction_probability:.2%}**")

        st.info(
            "This result should be interpreted as a machine learning prediction, not a medical diagnosis."
        )