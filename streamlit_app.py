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

SCOPES = [
    'https://www.googleapis.com/auth/fitness.activity.read',
    'https://www.googleapis.com/auth/fitness.sleep.read',
    'https://www.googleapis.com/auth/fitness.heart_rate.read',
]

def authenticate_google_fit():
    """Authenticate and return the Google Fit API service object."""
    creds = None
    
    # Load saved credentials if available
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # Refresh or initiate new authentication if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Load client credentials from environment variables
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
            
            # Manual URL generation for environments without a GUI browser
            auth_url, _ = flow.authorization_url(prompt='consent')
            print("Please visit this URL to authorize the application:", auth_url)
            code = input("Enter the authorization code: ")
            creds = flow.fetch_token(code=code)

        # Save the credentials for future use
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

# Function to load the trained model
def load_model():
    """Load the trained heart disease prediction model."""
    with open('heart_disease_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Train the model function
def train_model(df):
    """Train a model using the provided dataframe."""
    # Assuming `df` is your heart disease data with 'target' column as labels
    X = df.drop('num', axis=1)  # Features (drop target column)
    y = df['num']  # Labels (target column)

    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the model to a file
    with open('heart_disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved.")

# Load the heart disease dataset
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

# Train the model if it has not been trained yet
if not os.path.exists('heart_disease_model.pkl'):
    train_model(df)

# Load model into session state
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
    prediction = model.predict(user_data)
    st.write(f'Prediction: {"Heart disease likely" if prediction[0] == 1 else "No heart disease"}')

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
