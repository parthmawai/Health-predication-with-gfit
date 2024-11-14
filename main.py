import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
def load_data():
    # Replace 'heart_disease_uci.csv' with the correct path to your dataset
    df = pd.read_csv('heart_disease_uci.csv') 

    # Check for missing values and fill as needed
    df['fbs'] = df['fbs'].astype('category')
    df['exang'] = df['exang'].astype('category')
    df['slope'] = df['slope'].astype('category')
    df['ca'] = df['ca'].astype('category')
    df['thal'] = df['thal'].astype('category')

    # Fill missing values for numeric columns with mean
    df['trestbps'] = df['trestbps'].fillna(df['trestbps'].mean())
    df['thalch'] = df['thalch'].fillna(df['thalch'].mean())
    df['chol'] = df['chol'].fillna(df['chol'].mean())

    # Fill missing values for categorical columns with mode
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

    return df

# Train a RandomForest model and save it
def train_model():
    df = load_data()

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

# Load the trained model from a file
def load_model():
    model = joblib.load('heart_disease_model.pkl')
    return model

# Calling the training function (uncomment this when you want to train the model)
train_model()
