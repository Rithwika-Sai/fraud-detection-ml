import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ----------------------------
# Load Dataset
# ----------------------------

def load_data(path):

    data = pd.read_csv(path)

    print("Dataset Shape:", data.shape)
    print("First Rows:")
    print(data.head())

    return data


# ----------------------------
# Data Preprocessing
# ----------------------------

def preprocess_data(data):

    X = data.drop("is_fraud", axis=1)
    y = data["is_fraud"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# ----------------------------
# Train Model
# ----------------------------

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return model


# ----------------------------
# Save Model
# ----------------------------

def save_model(model, scaler):

    joblib.dump(model, "fraud_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Model saved successfully")


# ----------------------------
# Fraud Prediction Function
# ----------------------------

def predict_transaction(transaction_data):

    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")

    data_scaled = scaler.transform([transaction_data])

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        return "Fraudulent Transaction"
    else:
        return "Legitimate Transaction"


# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":

    dataset_path = "dataset_sample.csv"

    data = load_data(dataset_path)

    X, y, scaler = preprocess_data(data)

    model = train_model(X, y)

    save_model(model, scaler)

    # Example transaction prediction
    example_transaction = [3500, 0.7, 0.4]

    result = predict_transaction(example_transaction)

    print("\nPrediction Result:", result)
