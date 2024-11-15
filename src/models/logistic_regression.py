# logistic_regression.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd 
import os
def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    return accuracy, report

def main(data_path):
    """Main function to load data, train model, and evaluate it."""
    df = pd.read_csv(data_path)

    # Features and labels
    X = df.drop(columns=['isFraud', 'isFlaggedFraud'])
    y = df['isFraud']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = train_logistic_regression(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_regression_model.pkl")
    print("Model saved as logistic_regression_model.pkl")

if __name__ == "__main__":
    main("data\processed\preprocessed_data.csv")
