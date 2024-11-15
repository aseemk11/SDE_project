# feature_engineering.py
import pandas as pd

def create_transaction_features(df):
    """Creates features based on the transaction type and balances."""
    # Feature 1: Transaction type encoded as categorical
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    # Feature 2: Difference between old and new balance for the origin account
    df['balance_diff_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    
    # Feature 3: Difference between old and new balance for the destination account
    df['balance_diff_dest'] = df['oldbalanceDest'] - df['newbalanceDest']
    
    return df

def create_custom_features(df):
    """Creates custom features specifically for fraud detection."""
    # Feature: Ratio of transaction amount to the origin account's old balance
    df['amount_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)  # Add 1 to avoid division by zero
    
    # Feature: Flag if origin account balance goes to zero after transaction
    df['orig_balance_zeroed'] = (df['newbalanceOrig'] == 0).astype(int)
    
    # Feature: Flag if destination account balance starts at zero and increases
    df['dest_balance_start_zero'] = (df['oldbalanceDest'] == 0).astype(int)
    
    return df
