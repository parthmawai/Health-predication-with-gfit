import os
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/fitness.activity.read https://www.googleapis.com/auth/fitness.sleep.read https://www.googleapis.com/auth/fitness.heart_rate.read']

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
    
    try:
        # Fetch data from Google Fit API
        dataset = service.users().dataSources().datasets().get(
            userId='me',
            dataSourceId='derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm',
            datasetId=dataset_id
        ).execute()
        
        # Print the raw API response for debugging
        print(f"API Response: {dataset}")  # <-- Debugging line
        
        if 'point' not in dataset:
            raise ValueError("No 'point' data found in the response.")
        
        heart_rate_data = []
        for point in dataset['point']:
            for value in point['value']:
                heart_rate_data.append({
                    'timestamp': point['endTimeNanos'],
                    'heart_rate': value['fpVal']  # Heart rate in bpm
                })
        return heart_rate_data
    except Exception as e:
        st.error(f"Error fetching heart rate data: {str(e)}")
        return []

# Function to load Google Fit data
@st.cache_data  # Cache data for efficiency
def load_google_fit_data():
    """Load heart rate data from Google Fit."""
    try:
        service = authenticate_google_fit()  # Authenticate and get service
        heart_rate_data = get_heart_rate_data(service)  # Get heart rate data
        return heart_rate_data
    except Exception as e:
        st.error(f"Error during Google Fit authentication or data retrieval: {str(e)}")
        return []

# Add this check to optimize model loading
if 'model' not in st.session_state:
    if os.path.exists('heart_disease_model.pkl'):
        st.session_state.model = load_model()
    else:
        train_model(df)  # Train the model if not already trained
        st.session_state.model = load_model()

model = st.session_state.model

# Title and description
st.title("Heart Disease and Heart Rate Analysis App")
st.write("This app predicts heart disease likelihood and provides heart rate analysis.")

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
            
            plt.figure(figsize=(10, 6))
            plt.plot(df['timestamp'], df['heart_rate'], label='Heart Rate (bpm)')
            plt.xlabel('Timestamp')
            plt.ylabel('Heart Rate (bpm)')
            plt.title('Heart Rate Over Time')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)
    else:
        st.warning("No heart rate data available.")
except Exception as e:
    st.error(f"Error displaying heart rate data: {str(e)}")
