# test_explainers.py
import pandas as pd
from src.explainability.explain_model import explain_shap  # Assuming a SHAP explanation function

def test_explain_shap():
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'is_fraud': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    # Assume we have a trained model and a sample
    explanation = explain_shap(model=None, sample=df.iloc[0:1, :-1])  # Replace with actual model
    assert explanation is not None
