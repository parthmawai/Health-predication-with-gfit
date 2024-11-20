import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load('sleep_disorder_model.pkl')
    preprocessor = model.named_steps['preprocessor']
    return model, preprocessor

# Predict sleep disorder
def predict_sleep_disorder(input_data, model, preprocessor):
    # Convert input into a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Transform features
    transformed_features = preprocessor.transform(input_df)
    
    # Predict
    prediction = model.named_steps['classifier'].predict(transformed_features)
    prediction_proba = model.named_steps['classifier'].predict_proba(transformed_features)
    return prediction[0], prediction_proba[0]

# Streamlit App
def main():
    st.title("Sleep Disorder Prediction")
    st.markdown("Predict sleep disorders using lifestyle and health parameters.")

    # Input fields for user data
    st.sidebar.header("Enter Input Details")
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    age = st.sidebar.number_input("Age", min_value=10, max_value=100, step=1, value=30)
    occupation = st.sidebar.selectbox(
        "Occupation",
        options=["Software Engineer", "Doctor", "Sales Representative", "Other"]
    )
    sleep_duration = st.sidebar.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, step=0.1, value=6.0)
    quality_of_sleep = st.sidebar.slider("Quality of Sleep (1-10)", min_value=1, max_value=10, value=6)
    physical_activity = st.sidebar.slider("Physical Activity Level (1-100)", min_value=0, max_value=100, value=50)
    stress_level = st.sidebar.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
    bmi_category = st.sidebar.selectbox("BMI Category", options=["Underweight", "Normal", "Overweight", "Obese"])
    blood_pressure = st.sidebar.selectbox("Blood Pressure", options=["Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2"])
    heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=30, max_value=200, step=1, value=72)
    daily_steps = st.sidebar.number_input("Daily Steps", min_value=0, max_value=50000, step=100, value=5000)

    # Prepare input data
    input_data = {
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality_of_sleep,
        "Physical Activity Level": physical_activity,
        "Stress Level": stress_level,
        "BMI Category": bmi_category,
        "Blood Pressure": blood_pressure,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps
    }

    # Predict button
    if st.sidebar.button("Predict"):
        model, preprocessor = load_model_and_preprocessor()
        prediction, prediction_proba = predict_sleep_disorder(input_data, model, preprocessor)

        st.header("Prediction Results")
        st.write(f"Predicted Sleep Disorder: **{prediction}**")
        st.write("Prediction Probabilities:")
        st.write(pd.DataFrame(prediction_proba, index=model.named_steps['classifier'].classes_, columns=["Probability"]).T)

        st.success("Prediction completed!")

# Run the app
if __name__ == "__main__":
    main()
