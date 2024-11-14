import streamlit as st
import pandas as pd
from main import load_model, predict_heart_disease

# Title and description
st.title("Heart Disease Prediction App")
st.write("Enter the following details to predict the likelihood of heart disease:")

# Input fields for user features
age = st.number_input("Age", min_value=1, max_value=120, step=1)
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, step=1)
chol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=400, step=1)
thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, step=1)
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)

# Load model with caching
@st.cache_data
def get_model():
    return load_model()

model = get_model()

# Predict button
if st.button('Predict Heart Disease'):
    # Combine inputs into a DataFrame (or a 2D list) to pass to the model
    user_data = pd.DataFrame([[age, trestbps, chol, thalch, oldpeak]], 
                             columns=['age', 'trestbps', 'chol', 'thalch', 'oldpeak'])
    
    # Run prediction
    prediction = predict_heart_disease(model, user_data)
    
    # Display result
    st.write(f'Prediction: {prediction}')
