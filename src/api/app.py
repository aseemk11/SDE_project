from flask import Flask, request, jsonify
from models.logistic_regression import load_model
from explainability.shap_explainer import generate_shap_values

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data['features'])
    explanation = generate_shap_values(model, data['features'])
    return jsonify({'prediction': int(prediction[0]), 'explanation': explanation})

if __name__ == '__main__':
    app.run(debug=True)
