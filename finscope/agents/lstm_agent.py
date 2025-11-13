import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
# from dotenv import load_dotenv
import os

# load_dotenv()  # Load environment variables from .env file

class LSTMAgent:
    """
    The LSTM Agent for quantitative analysis of stock prices.
    """
    def __init__(self, sequence_length=60, model_path="models/lstm_model.h5"):
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _preprocess_data(self, data):
        """
        Prepares the data for the LSTM model.
        Scales the data and creates sequences.
        """
        # We only need the 'Close' price for this model
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        X_train = []
        y_train = []
        
        for i in range(self.sequence_length, len(scaled_data)):
            X_train.append(scaled_data[i-self.sequence_length:i, 0])
            y_train.append(scaled_data[i, 0])
            
        return np.array(X_train), np.array(y_train)

    def build_model(self, input_shape):
        """
        Builds the LSTM model architecture.
        """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        print("LSTM model built successfully.")

    def train(self, data):
        """
        Trains the LSTM model on the provided historical data.
        """
        print("Training LSTM model...")
        X_train, y_train = self._preprocess_data(data)
        
        # Reshape data for LSTM [samples, timesteps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        self.build_model((X_train.shape[1], 1))
        self.model.fit(X_train, y_train, epochs=25, batch_size=32)
        
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"Model trained and saved to {self.model_path}")

    def load(self):
        """
        Loads a pre-trained model from the specified path.
        """
        if os.path.exists(self.model_path):
            print(f"Loading pre-trained model from {self.model_path}...")
            self.model = load_model(self.model_path)
            print("Model loaded successfully.")
            return True
        else:
            print(f"Error: Model file not found at {self.model_path}")
            return False

    def predict(self, data):
        """
        Predicts the next day's closing price.
        
        Args:
            data (pandas.DataFrame): A DataFrame containing at least the last
                                     `sequence_length` days of stock data.
        
        Returns:
            float: The predicted (scaled) closing price.
        """
        if self.model is None:
            print("Error: Model is not loaded or trained.")
            return None
            
        print("Making a prediction with the LSTM model...")
        # Get the last `sequence_length` days of closing prices
        last_sequence = data['Close'][-self.sequence_length:].values.reshape(-1, 1)
        scaled_sequence = self.scaler.transform(last_sequence)
        
        # Reshape for the model
        X_test = np.array([scaled_sequence.flatten()])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predicted_price_scaled = self.model.predict(X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price_scaled)
        
        return predicted_price[0][0]
