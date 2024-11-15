# neural_network.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
def create_neural_network(input_dim):
    """Define a simple neural network model."""
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(X_train, y_train):
    """Train a neural network model."""
    input_dim = X_train.shape[1]
    model = create_neural_network(input_dim)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    return accuracy, report

def main(data_path):
    """Main function to load data, train model, and evaluate it."""
    df = pd.read_csv(data_path)

    # Features and labels
    X = df.drop(columns=['isFraud', 'isFlaggedFraud'])
    y = df['isFraud']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = train_neural_network(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Save the model
    model.save('neural_network_model.h5')
    print("Model saved as neural_network_model.h5")

if __name__ == "__main__":
    main("data\processed\preprocessed_data.csv")
