import os
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Fit Authentication and Data Retrieval Functions
SCOPES = ['https://www.googleapis.com/auth/fitness.activity.read']

def authenticate_google_fit():
    """Authenticate and return the service object."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Use environment variables for Google OAuth client ID and secret
            client_id = os.getenv("GOOGLE_CLIENT_ID")
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

            flow = InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token"
                    }
                },
                SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    service = build('fitness', 'v1', credentials=creds)
    return service

def get_heart_rate_data(service):
    """Retrieve heart rate data from Google Fit."""
    end_time = int(time.time() * 1000)  # Current time in milliseconds
    start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)  # Last 30 days
    
    dataset_id = f"{start_time}-{end_time}"
    
    dataset = service.users().dataSources().datasets().get(
        userId='me',
        dataSourceId='derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm',
        datasetId=dataset_id
    ).execute()
    
    heart_rate_data = []
    for point in dataset['point']:
        for value in point['value']:
            heart_rate_data.append({
                'timestamp': point['endTimeNanos'],
                'heart_rate': value['fpVal']  # Heart rate in bpm
            })
    
    return heart_rate_data

# Function to load Google Fit data
@st.cache_data  # Cache data for efficiency
def load_google_fit_data():
    """Load heart rate data from Google Fit."""
    service = authenticate_google_fit()  # Authenticate and get service
    heart_rate_data = get_heart_rate_data(service)  # Get heart rate data
    return heart_rate_data

# Load the heart disease dataset and train the model
df = pd.read_csv('heart_disease_uci.csv')  # Ensure you have this CSV in the correct path

# Preprocess the data
df['fbs'] = df['fbs'].astype('category')
df['exang'] = df['exang'].astype('category')
df['slope'] = df['slope'].astype('category')
df['ca'] = df['ca'].astype('category')
df['thal'] = df['thal'].astype('category')

# Fill missing values
df['trestbps'] = df['trestbps'].fillna(df['trestbps'].mean())
df['thalch'] = df['thalch'].fillna(df['thalch'].mean())
df['chol'] = df['chol'].fillna(df['chol'].mean())
df['fbs'] = df['fbs'].fillna(df['fbs'].mode()[0])
df['exang'] = df['exang'].fillna(df['exang'].mode()[0])
df['oldpeak'] = df['oldpeak'].fillna(df['oldpeak'].mean())
df['slope'] = df['slope'].fillna(df['slope'].mode()[0])
df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

# Standardize numeric columns
numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Train the model function
def train_model(df):
    # Splitting dataset into features and target
    features = df[['age', 'trestbps', 'chol', 'thalch', 'oldpeak']]
    target = df['num']  # Assuming 'num' is the target column
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # Save the model
    joblib.dump(model, 'heart_disease_model.pkl')
    print("Model trained and saved.")

# Load model function (from session state)
@st.cache_data
def load_model():
    """Load the trained heart disease prediction model."""
    return joblib.load('heart_disease_model.pkl')

# Predict heart disease function
def predict_heart_disease(model, user_data):
    """Predict heart disease likelihood."""
    prediction = model.predict(user_data)
    return "Heart disease likely" if prediction[0] == 1 else "No heart disease"

# Add this check to optimize model loading
if 'model' not in st.session_state:
    st.session_state.model = load_model()

model = st.session_state.model

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

# Section for Google Fit Heart Rate Data
st.header("Heart Rate Data from Google Fit")
try:
    heart_rate_data = load_google_fit_data()
    if heart_rate_data:
        st.subheader("Recent Heart Rate Data")
        st.write(heart_rate_data)  # Display raw heart rate data
        
        # Optionally, you can plot the heart rate data over time
        if heart_rate_data:
            df = pd.DataFrame(heart_rate_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')  # Convert to readable datetime
            
            st.subheader("Heart Rate Over Time")
            plt.figure(figsize=(10, 5))
            plt.plot(df['timestamp'], df['heart_rate'], color='blue')
            plt.xlabel("Date")
            plt.ylabel("Heart Rate (bpm)")
            plt.title("Heart Rate Data from Google Fit")
            st.pyplot(plt)
            plt.clf()  # Clear the plot for next display
    else:
        st.write("No heart rate data found.")
except Exception as e:
    st.error(f"Error fetching heart rate data: {str(e)}")
