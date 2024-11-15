# Importing necessary functions
from src.models.logistic_regression import train_logistic_regression
from src.models.neural_network import train_neural_network
from src.models.evaluate_model import evaluate_model_function
import pandas as pd
# Load your dataset
df = pd.read_csv("data\Transactions Data.csv")

# Train Logistic Regression Model
logistic_model, X_test, y_test = train_logistic_regression(df, target_column="is_fraud")

# Evaluate Logistic Regression Model
logistic_y_pred = logistic_model.predict(X_test)
logistic_results = evaluate_model_function(y_test, logistic_y_pred)
print("Logistic Regression Results:", logistic_results)

# Train Neural Network Model
nn_model, X_test, y_test = train_neural_network(df, target_column="is_fraud")

# Evaluate Neural Network Model
nn_y_pred = (nn_model.predict(X_test) > 0.5).astype(int)
nn_results = evaluate_model_function(y_test, nn_y_pred)
print("Neural Network Results:", nn_results)
