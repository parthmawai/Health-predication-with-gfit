import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_model():
    df = pd.read_csv('heart_disease_uci.csv')
    features = df[['age', 'trestbps', 'chol', 'thalch', 'oldpeak']]
    target = df['num']
    model = RandomForestClassifier()
    model.fit(features, target)
    joblib.dump(model, 'heart_disease_model.pkl')  # Save model
    
def load_model():
    return joblib.load('heart_disease_model.pkl')  # Load model

def predict_heart_disease(features):
    model = load_model()
    prediction = model.predict(features)
    return "Positive" if prediction[0] == 1 else "Negative"
