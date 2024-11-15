#!/bin/bash
# run_preprocess.sh
echo "Running data preprocessing..."
python -m src.data_preprocessing.feature_engineering
echo "Data preprocessing completed."
