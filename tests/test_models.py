# test_models.py
import pandas as pd
from src.models.logistic_regression import train_logistic_regression
from src.models.neural_network import train_neural_network
from src.models.evaluate_model import evaluate_model_function

def test_train_logistic_regression():
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'is_fraud': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    model, X_test, y_test = train_logistic_regression(df, target_column="is_fraud")
    assert model is not None

def test_evaluate_model_function():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    results = evaluate_model_function(y_true, y_pred)
    assert 'accuracy' in results
    assert 'f1_score' in results
