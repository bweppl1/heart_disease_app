import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dataclasses import dataclass

from joblib import load

# Optimized model
model = load("heart_disease_predictor.joblib")

# Title and description
st.title("Heart Disease Predictor")
st.write("Random Forest Classifier Model to Predict Heart Disease")

with st.form(key='patient_form'):
    age = st.number_input("Age", min_value=0, max_value=110, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"], index=0)
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], index=0)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, value=120)
    #check min/max chol values
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"], index=0)
    restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], index=0)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["True", "False"], index=0)
    oldpeak = st.number_input("Oldpeak (depression induced by exercise relative to rest)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"], index=0)
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"], index=0)

    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Convert categorical variables to numerical
        patient_data = pd.DataFrame({
            "Age": [age],
            "Sex": [1 if sex == "Male" else 0],
            "Chest pain type": [1 if cp == "Typical Angina" else 2 if cp == "Atypical Angina" else 3 if cp == "Non-Anginal Pain" else 4],
            "BP": [trestbps],
            "Cholesterol": [chol],
            "FBS over 120": [1 if fbs == "True" else 0],
            "EKG results": [0 if restecg == "Normal" else 1 if restecg == "ST-T Wave Abnormality" else 2],
            "Max HR": [thalach],
            "Exercise angina": [1 if exang == "True" else 0],
            "ST depression": [oldpeak],
            "Slope of ST": [1 if slope == "Upsloping" else 2 if slope == "Flat" else 3],
            "Number of vessels fluro": [ca],
            "Thallium": [3 if thal == "Normal" else 6 if thal == "Fixed Defect" else 7]
        })
    
        # Make prediction
        prediction = model.predict(patient_data)[0]
        prediction_proba = model.predict_proba(patient_data)[0]
        
        # Convert probabilities to percentage
        heart_disease_prob = prediction_proba[1]*100
        no_heart_disease_prob = prediction_proba[0]*100

        # Display prediction
        st.write("### Prediction Result")
        if prediction == 1:
            st.error("The model predicts that the patient has heart disease.")
        else:
            st.success("The model predicts that the patient does not have heart disease.")
        st.write("### Prediction Probability")
        st.write(f"Probability of heart disease: {heart_disease_prob:.2f}%")
        st.write(f"Probability of no heart disease: {no_heart_disease_prob:.2f}%")

# Save patient data to CSV
def save_patient_data(patient_data):
    patient_df = pd.DataFrame(patient_data)
    patient_df.to_csv("data/new_patient_data.csv", mode='a', header=False, index=False)
    st.success("Patient data saved to CSV file.")

# Save patient data when the form is submitted
if submit_button:
    save_patient_data(patient_data)

# Show raw input data
    st.json(patient_data.to_dict(orient='records'))