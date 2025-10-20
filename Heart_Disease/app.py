import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import numpy as np
import os


def main():
    st.set_page_config(page_title="💓 Heart Disease Prediction", layout="centered")
    st.title("💓 Heart Disease Prediction App")


# Load Model and Scaler

base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, "heart_disease_model.pkl"))
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
with open(os.path.join(base_path, "feature_columns.json"), "r") as f:
    feature_columns = json.load(f)

# App Title

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.markdown("Predict likelihood of heart disease using clinical parameters.")


# User Input

st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", 20, 90, 50)
sex = st.sidebar.selectbox("Sex", (0, 1), format_func=lambda x: "Female" if x==0 else "Male")
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type", [1, 2, 3, 4],
    format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x-1]
)
bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 250)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120mg/dL", (0, 1))
ecg = st.sidebar.selectbox(
    "Resting ECG Results", [0, 1, 2],
    format_func=lambda x: ["Normal", "ST-T Abnormality", "LVH"][x]
)
max_hr = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)
ex_angina = st.sidebar.selectbox("Exercise Induced Angina", (0, 1), format_func=lambda x: "No" if x==0 else "Yes")
oldpeak = st.sidebar.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
slope = st.sidebar.selectbox(
    "ST Slope", [1, 2, 3],
    format_func=lambda x: ["Upward", "Flat", "Downward"][x-1]
)


# Create Input DataFrame

patient_input = {
    "age": age, "sex": sex, "chest pain type": chest_pain,
    "resting bp s": bp, "cholesterol": chol, "fasting blood sugar": fbs,
    "resting ecg": ecg, "max heart rate": max_hr, "exercise angina": ex_angina,
    "oldpeak": oldpeak, "ST slope": slope
}

df_input = pd.DataFrame([patient_input])

# --- Data Preprocessing ---
patient_input = {
    "age": age, "sex": sex, "chest pain type": chest_pain,
    "resting bp s": bp, "cholesterol": chol, "fasting blood sugar": fbs,
    "resting ecg": ecg, "max heart rate": max_hr,
    "exercise angina": ex_angina, "oldpeak": oldpeak, "ST slope": slope
}

df_input = pd.DataFrame([patient_input])
df_input = pd.get_dummies(df_input, columns=["chest pain type", "resting ecg", "ST slope"], drop_first=True)

for col in feature_columns:
    if col not in df_input.columns:
        df_input[col] = 0
df_input = df_input[feature_columns]

num_cols = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
df_input[num_cols] = scaler.transform(df_input[num_cols])

# --- Prediction ---
if st.button("🔍 Predict"):
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    if pred == 1:
        st.error(f"⚠️ High likelihood of Heart Disease (Probability={prob:.2f})")
    else:
        st.success(f"✅ Likely Normal Heart (Probability={1-prob:.2f})")


# Footer

st.markdown("---")
st.caption("Developed as part of a Machine Learning Portfolio Project by Amit Kharote.")

if __name__ == "__main__":
main()
