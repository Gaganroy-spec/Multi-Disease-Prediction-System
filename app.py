# app.py
import streamlit as st

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Multi‑Disease Predictor", layout="wide")

import pandas as pd
import joblib
import numpy as np

# Load all models
@st.cache_resource
def load_models():
    fever_model = joblib.load("fever_best_model.pkl")
    diabetes_model = joblib.load("diabetes_best_model.pkl")
    heart_model = joblib.load("heart_best_model.pkl")
    return fever_model, diabetes_model, heart_model

fever_model, diabetes_model, heart_model = load_models()

st.title("🩺 Disease Prediction System")
st.markdown("Predict risk of **Fever**, **Diabetes**, or **Heart Attack** using machine learning.")

# Sidebar for disease selection
disease = st.sidebar.selectbox("Select Disease to Predict", ("Fever", "Diabetes", "Heart Attack"))

# ------------------- Fever Inputs -------------------
if disease == "Fever":
    st.header("Fever Prediction")
    st.write("Enter the following symptoms / measurements:")
    
    temperature = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=106.0, value=98.6, step=0.1)
    cough = st.selectbox("Cough", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    headache = st.selectbox("Headache", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    fatigue = st.selectbox("Fatigue", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    sore_throat = st.selectbox("Sore Throat", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    muscle_ache = st.selectbox("Muscle Ache", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    
    input_data = pd.DataFrame([{
        'temperature': temperature,
        'cough': cough,
        'headache': headache,
        'fatigue': fatigue,
        'sore_throat': sore_throat,
        'muscle_ache': muscle_ache
    }])
    
    if st.button("Predict Fever Risk"):
        prediction = fever_model.predict(input_data)[0]
        proba = fever_model.predict_proba(input_data)[0][1]
        if prediction == 1:
            st.error(f"⚠️ High risk of fever! (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Low risk of fever. (Probability: {proba:.2f})")

# ------------------- Diabetes Inputs -------------------
elif disease == "Diabetes":
    st.header("Diabetes Prediction")
    st.write("Enter the following medical metrics:")
    
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=25)
    insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=500, value=80)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=28.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=35)
    
    input_data = pd.DataFrame([{
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'diabetes_pedigree': dpf,
        'age': age
    }])
    
    if st.button("Predict Diabetes Risk"):
        prediction = diabetes_model.predict(input_data)[0]
        proba = diabetes_model.predict_proba(input_data)[0][1]
        if prediction == 1:
            st.error(f"⚠️ High risk of diabetes! (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Low risk of diabetes. (Probability: {proba:.2f})")

# ------------------- Heart Attack Inputs -------------------
else:  # Heart Attack
    st.header("Heart Attack Risk Prediction")
    st.write("Enter the following clinical features:")
    
    age = st.number_input("Age (years)", min_value=20, max_value=100, value=55)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=130)
    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                           format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                        format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])
    
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }])
    
    if st.button("Predict Heart Attack Risk"):
        prediction = heart_model.predict(input_data)[0]
        proba = heart_model.predict_proba(input_data)[0][1]
        if prediction == 1:
            st.error(f"⚠️ High risk of heart attack! (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Low risk of heart attack. (Probability: {proba:.2f})")