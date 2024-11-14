# Required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
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

# Ready to use `df` for ML model training and prediction
print("Data prepared for model training.")
