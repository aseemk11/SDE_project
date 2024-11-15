# shap_explainer.py
import shap
import joblib
import pandas as pd

def explain_shap(model, X):
    explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)
    return shap_values

def main():
    model = joblib.load("models\logistic_regression_model.pkl")
    X = pd.read_csv("data\processed\preprocessed_data.csv")
    
    shap_values = explain_shap(model, X)
    shap.summary_plot(shap_values, X, plot_type="bar")

if __name__ == "__main__":
    main()
