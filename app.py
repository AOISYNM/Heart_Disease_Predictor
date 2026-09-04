import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Load model artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("LR_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

# ----------------------------
# Light styling
# ----------------------------
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.5rem;
    }
    .stButton > button, .stFormSubmitButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        background-color: #e63946;
        color: white;
        border: none;
        box-shadow: 0 4px 10px rgba(230, 57, 70, 0.35);
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #d62839;
        color: white;
        box-shadow: 0 6px 14px rgba(230, 57, 70, 0.45);
        transform: translateY(-1px);
    }
    .stButton > button:active, .stFormSubmitButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 6px rgba(230, 57, 70, 0.35);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This tool estimates the likelihood of heart disease from basic "
        "clinical measurements using a Logistic Regression model."
    )
    st.markdown("---")
    st.caption(
        "⚠️ This is not a medical diagnosis. Always consult a qualified "
        "healthcare professional for real medical advice."
    )

# ----------------------------
# Header
# ----------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.write("Fill in the details below and click **Predict** to estimate risk.")
st.markdown("---")

# ----------------------------
# Input form
# ----------------------------
with st.form("prediction_form"):

    st.subheader("👤 Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 40)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])

    st.subheader("🩺 Vitals")
    col1, col2 = st.columns(2)
    with col1:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120
        )
        max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    with col2:
        cholesterol = st.number_input(
            "Cholesterol (mg/dl)", min_value=100, max_value=600, value=200
        )
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Y", "N"])

    st.subheader("💓 Heart Function")
    col1, col2 = st.columns(2)
    with col1:
        chest_pain = st.selectbox(
            "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
            help="ATA: Atypical Angina · NAP: Non-Anginal Pain · TA: Typical Angina · ASY: Asymptomatic",
        )
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
    with col2:
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    st.markdown("")
    submitted = st.form_submit_button("Predict")

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Y" else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])
    input_df = pd.get_dummies(input_df, drop_first=True)

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    proba = (
        model.predict_proba(scaled_input)[0]
        if hasattr(model, "predict_proba")
        else None
    )

    st.markdown("---")
    st.subheader("Result")

    result_col, metric_col = st.columns([2, 1])

    with result_col:
        if prediction == 1:
            st.error("⚠️ **High Risk of Heart Disease**")
        else:
            st.success("✅ **Low Risk of Heart Disease**")

    if proba is not None:
        risk_pct = proba[1] * 100
        with metric_col:
            st.metric("Estimated Risk", f"{risk_pct:.1f}%")
        st.progress(min(int(risk_pct), 100))

    with st.expander("See the values used for this prediction"):
        st.dataframe(
            pd.DataFrame([raw_input]).T.rename(columns={0: "Value"}),
            use_container_width=True,
        )