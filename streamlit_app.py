import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the heart rate data from CSV
@st.cache  # Cache data for performance
def load_data():
    data = pd.read_csv('heart_rate_data.csv')
    data['date'] = pd.to_datetime(data['date'])  # Ensure 'date' is datetime type
    return data

# App title and description
st.title("Heart Rate Analysis and Prediction")
st.write("This app displays heart rate data and provides insights based on your ML model.")

# Load data
data = load_data()

# Display raw data
st.subheader("Raw Heart Rate Data")
st.write(data)

# Plot the heart rate over time
st.subheader("Heart Rate Over Time")
plt.figure(figsize=(10, 5))
plt.plot(data['date'], data['heart_rate'], color='blue')
plt.xlabel("Date")
plt.ylabel("Heart Rate (bpm)")
plt.title("Daily Heart Rate")
st.pyplot(plt)
