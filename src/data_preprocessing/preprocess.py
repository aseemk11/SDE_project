import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path):
    """Loads data from a CSV file and returns a DataFrame."""
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Cleans the dataset by removing invalid or unnecessary columns.
    """
    # Drop columns that are not useful for training
    df = df.drop(columns=["nameOrig", "nameDest"])  # Remove non-numeric ID columns
    
    # Filter out rows with invalid or missing data
    df = df.dropna()
    df = df[df["amount"] > 0]
    return df

def handle_missing_values(df):
    """Handles missing values by filling them with appropriate strategies."""
    df.fillna(0, inplace=True)  # Fill any potential missing values with 0
    return df

def encode_categorical_features(df):
    """Encodes categorical features to numeric."""
    # Label encode 'type' column
    label_encoder = LabelEncoder()
    df['type'] = label_encoder.fit_transform(df['type'])
    return df

def scale_numeric_features(df, numeric_columns):
    """Scales numeric features to have mean 0 and variance 1."""
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def preprocess_data(file_path, output_path):
    """Loads, cleans, preprocesses the data, and saves it to output_path."""
    df = load_data(file_path)
    df = clean_data(df)
    df = handle_missing_values(df)

    # Encode categorical columns
    df = encode_categorical_features(df)


    # Define numeric columns to scale
    numeric_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    df = scale_numeric_features(df, numeric_columns)
    return(df)
