import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and expected feature columns
model = joblib.load("xgb_churn_model.pkl")
expected_features = joblib.load("model_features.pkl")

st.title("📉 Customer Churn Prediction")

# UI inputs
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", 
    "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

# Construct input dict
input_dict = {
    'SeniorCitizen': SeniorCitizen,
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'gender': gender,
    'Partner': Partner,
    'Dependents': Dependents,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One-hot encode to match training format
input_encoded = pd.get_dummies(input_df)

# Add any missing columns from training
for col in expected_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure correct order
input_encoded = input_encoded[expected_features]

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.markdown("## 🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **Churn** (Risk Score: {probability:.2f})")
    else:
        st.success(f"✅ This customer is likely to **Stay** (Risk Score: {probability:.2f})")
