# train_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

def train_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def main():
    data = pd.read_csv("../data/processed/processed_data.csv")
    X = data.drop(columns=["is_fraud"])
    y = data["is_fraud"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = train_logistic_regression(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    print("Model Accuracy:", accuracy)
    joblib.dump(model, "../models/logistic_regression_model.pkl")

if __name__ == "__main__":
    main()
