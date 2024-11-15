# test_preprocessing.py
import pandas as pd
from src.data_preprocessing.feature_engineering import create_transaction_features, encode_categorical_features, scale_numeric_features

def test_create_transaction_features():
    data = {'transaction_date': ['2024-11-10 12:45:00', '2024-11-11 08:30:00']}
    df = pd.DataFrame(data)
    df = create_transaction_features(df)
    assert 'transaction_month' in df.columns
    assert 'transaction_hour' in df.columns

def test_encode_categorical_features():
    data = {'transaction_type': ['debit', 'credit', 'debit']}
    df = pd.DataFrame(data)
    df = encode_categorical_features(df)
    assert 'transaction_type_encoded' in df.columns

def test_scale_numeric_features():
    data = {'transaction_amount': [100, 200, 300], 'account_age': [5, 10, 15]}
    df = pd.DataFrame(data)
    df = scale_numeric_features(df)
    assert df['transaction_amount'].std() == 1.0
    assert df['account_age'].mean() == 0.0
