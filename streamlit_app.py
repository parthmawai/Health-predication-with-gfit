import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import load_model, predict_heart_disease

# Load model with caching
@st.cache_data  # Updated caching function for Streamlit
def get_model():
    return load_model()

model = get_model()

# Title and description
st.title("Heart Disease and Heart Rate Analysis App")
st.write("This app predicts heart disease likelihood and provides heart rate analysis.")

# Section for Heart Disease Prediction
st.header("Heart Disease Prediction")
st.write("Enter the following details to predict the likelihood of heart disease:")

# Input fields for user features
age = st.number_input("Age", min_value=1, max_value=120, step=1)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, step=1)
chol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=400, step=1)
thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, step=1)
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)

# Predict button for heart disease
if st.button('Predict Heart Disease'):
    user_data = pd.DataFrame([[age, trestbps, chol, thalch, oldpeak]], columns=['age', 'trestbps', 'chol', 'thalch', 'oldpeak'])
    prediction = predict_heart_disease(model, user_data)
    st.write(f'Prediction: {prediction}')

# Section for Heart Rate Data Analysis
st.header("Heart Rate Data Analysis")

# Load heart rate data
@st.cache_data  # Updated caching function for data
def load_data():
    data = pd.read_csv('heart_rate_data.csv')
    data['date'] = pd.to_datetime(data['date'])
    return data

data = load_data()

# Display raw data
st.subheader("Raw Heart Rate Data")
st.write(data)

# Plot heart rate data over time
st.subheader("Heart Rate Over Time")
plt.figure(figsize=(10, 5))
plt.plot(data['date'], data['heart_rate'], color='blue')
plt.xlabel("Date")
plt.ylabel("Heart Rate (bpm)")
plt.title("Daily Heart Rate")
st.pyplot(plt)
