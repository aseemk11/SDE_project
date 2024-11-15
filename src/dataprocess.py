from data_preprocessing.preprocess import preprocess_data
from data_preprocessing.feature_engineering import create_transaction_features, create_custom_features
import os 
# Load and preprocess the data
file_path = "data\Transactions Data.csv"
output_path ="data\processed\preprocessed_data.csv" 
df = preprocess_data(file_path,output_path)

# Apply feature engineering 
df = create_transaction_features(df)
df = create_custom_features(df)
# Save the preprocessed data to a new CSV file
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Preprocessed data saved to {output_path}")

# Display the first few rows to verify
print(df.head())
