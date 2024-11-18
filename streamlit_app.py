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
import json
from streamlit_authenticator import Authenticate
import dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch credentials from environment variables
client_id = os.getenv("GOOGLE_CLIENT_ID")
client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
project_id = os.getenv("GOOGLE_PROJECT_ID")
private_key_id = os.getenv("GOOGLE_PRIVATE_KEY_ID")
private_key = os.getenv("GOOGLE_PRIVATE_KEY").replace("\\n", "\n")  # Ensure the private key is correctly formatted
client_email = os.getenv("GOOGLE_CLIENT_EMAIL")
client_x509_cert_url = os.getenv("GOOGLE_CLIENT_X509_CERT_URL")
auth_uri = os.getenv("GOOGLE_AUTH_URI")
token_uri = os.getenv("GOOGLE_TOKEN_URI")
auth_provider_x509_cert_url = os.getenv("GOOGLE_AUTH_PROVIDER_X509_CERT_URL")

# Ensure the required environment variables are present
if not all([client_id, client_secret, project_id, private_key, client_email]):
    st.error("Error: Missing some environment variables for Google API credentials.")
else:
    # Construct the credentials using the environment variables
    credentials_info = {
        "type": "service_account",
        "project_id": project_id,
        "private_key_id": private_key_id,
        "private_key": private_key,
        "client_email": client_email,
        "client_id": client_id,
        "auth_uri": auth_uri,
        "token_uri": token_uri,
        "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
        "client_x509_cert_url": client_x509_cert_url,
    }

    credentials = service_account.Credentials.from_service_account_info(credentials_info)

    # Initialize Google Fit API client
    try:
        service = build('fitness', 'v1', credentials=credentials)
        st.success("Successfully authenticated with Google Fit API!")
    except Exception as e:
        st.error(f"Error while authenticating: {e}")


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
            redirect_uri = "http://localhost:8502"

            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json',
                SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save the credentials for future use
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('fitness', 'v1', credentials=creds)
    return service


def get_heart_rate_data(service):
    """Retrieve heart rate data from Google Fit."""
    # Calculate the dataset time range
    end_time = int(time.time() * 1000)  # Current time in milliseconds
    start_time = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)  # Last 30 days

    dataset_id = f"{start_time}-{end_time}"
    
    try:
        dataset = service.users().dataSources().datasets().get(
            userId='me',
            dataSourceId='derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm',
            datasetId=dataset_id
        ).execute()
        
        heart_rate_data = []
        if 'point' in dataset:
            for point in dataset['point']:
                for value in point['value']:
                    heart_rate_data.append({
                        'timestamp': point['endTimeNanos'],
                        'heart_rate': value['fpVal']  # Heart rate in bpm
                    })
        else:
            print("No data points found in the response.")
        
        return heart_rate_data
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return []


# Function to load Google Fit data
@st.cache_data
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
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset."""
    # Load the heart disease dataset
    df = pd.read_csv('heart_disease_uci.csv')  # Ensure you have this CSV in the correct path

    # Display columns for debugging
    st.write("Columns in the dataset:", df.columns)

    # Preprocessing steps
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

    st.write("Preprocessing completed successfully!")

    return df


# Train the model if it has not been trained yet
if not os.path.exists('heart_disease_model.pkl'):
    df = load_and_preprocess_data()
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
age = st.number_input("Age", min_value=1, max_value=120, step=1, key="age_input")
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, step=1, key="bp_input")
chol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=400, step=1, key="chol_input")
thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, step=1, key="thalch_input")
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1, key="oldpeak_input")


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
