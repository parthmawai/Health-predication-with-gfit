import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load and preprocess the dataset
def load_data():
    # Replace 'heart_disease_uci.csv' with the correct path to your dataset
    df = pd.read_csv('./data/uci_heart_disease.csv')

    # Check for missing values and fill as needed
    df['fbs'] = df['fbs'].astype('category')
    df['exang'] = df['exang'].astype('category')
    df['slope'] = df['slope'].astype('category')
    df['ca'] = df['ca'].astype('category')
    df['thal'] = df['thal'].astype('category')

    # Fill missing values for numeric columns with mean
    numeric_columns = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())

    # Fill missing values for categorical columns with mode
    categorical_columns = ['fbs', 'exang', 'slope', 'ca', 'thal']
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

# Preprocess data and split into training/testing sets
def preprocess_data(df):
    # Separate features and target
    features = df.drop(columns=['num'])  # Assuming 'num' is the target column
    target = df['num']

    # Identify numeric and categorical columns
    numeric_features = features.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = features.select_dtypes(include=['category']).columns

    # Preprocessing pipelines
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

    return X_train, X_test, y_train, y_test, preprocessor

# Train a RandomForest model and save it
def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Create a pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best model accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save the model
    joblib.dump(best_model, 'heart_disease_model.pkl')
    print("Best model trained and saved.")

# Load the trained model from a file
def load_model():
    model = joblib.load('heart_disease_model.pkl')
    return model

# Train the model (uncomment this line to run training)
train_model()
