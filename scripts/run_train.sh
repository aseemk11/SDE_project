#!/bin/bash
# run_train.sh
echo "Training logistic regression model..."
python -m src.models.logistic_regression

echo "Training neural network model..."
python -m src.models.neural_network

echo "Model training completed."
