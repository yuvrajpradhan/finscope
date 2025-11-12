import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime


class LSTMAgent:
    
    def __init__(self, sequence_length=60, model_path="models/lstm_model.keras", scaler_path="models/lstm_scaler.pkl"):
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features_list = ['Open', 'High', 'Low', 'Close', 'Sentiment']
        self.target_feature = 'Close'

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        print("LSTMAgent initialized.")

    
    def _preprocess_data(self, data):
        
        feature_data = data[self.features_list].values
        scaled_data = self.scaler.fit_transform(feature_data)

        target_col_index = self.features_list.index(self.target_feature)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i, :])
            y.append(scaled_data[i, target_col_index])

        return np.array(X), np.array(y)

    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),

            LSTM(100, return_sequences=True),
            Dropout(0.2),

            LSTM(100, return_sequences=False),
            Dropout(0.2),

            Dense(50),
            Dense(1)
        ])


        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

        print("LSTM model built.")
        # print(f"Model Input Shape: (None, {input_shape[0]}, {input_shape[1]})")

    
    def train(self, data):
        print("Training LSTM model...")

        X_train, y_train = self._preprocess_data(data)
        
        self.build_model((X_train.shape[1], X_train.shape[2])) 

        self.model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)

        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"Model saved to {self.model_path}")
        print(f"Scaler saved to {self.scaler_path}")

    
    def load(self):
        if not os.path.exists(self.model_path):
            print("Model file not found.")
            return False

        print("Loading model...")
        self.model = load_model(self.model_path)

        print("Loading scaler...")
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            print("Error: scaler.pkl not found.")
            return False

        print("Model + scaler loaded successfully.")
        return True

    
    def predict(self, data):
        if self.model is None:
            print("Error: Model is not loaded.")
            return None

        if len(data) < self.sequence_length:
            print("Error: Not enough data for prediction.")
            return None

        print("Predicting next closing price...")

        last_seq = data[self.features_list][-self.sequence_length:].values
        scaled_seq = self.scaler.transform(last_seq)

        X_test = scaled_seq.reshape(1, self.sequence_length, last_seq.shape[1])

        pred_scaled = self.model.predict(X_test, verbose=0)
        
        dummy_array = np.zeros((pred_scaled.shape[0], len(self.features_list)))
        
        target_col_index = self.features_list.index(self.target_feature)
        dummy_array[:, target_col_index] = pred_scaled[:, 0]
        
        pred_array = self.scaler.inverse_transform(dummy_array)
        
        pred_price = pred_array[:, target_col_index]

        return float(pred_price[0])

    
    def _plot_evaluation_results(self, actual, predicted):
        """Generates and saves a series of diagnostic plots."""
        
        os.makedirs("plots", exist_ok=True)
        # ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate residuals for error plots
        residuals = actual.flatten() - predicted.flatten()
        
        print("\nSaving visualizations to /plots directory...")

        # 1. Actual vs Predicted
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label="Actual Price", linewidth=2)
        plt.plot(predicted, label="Predicted Price", linewidth=2)
        plt.title("Actual vs Predicted Stock Price")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        file1 = f"plots/actual_vs_predicted.png"
        plt.savefig(file1)
        plt.close()

        # 2. Residual Plot
        plt.figure(figsize=(12, 5))
        plt.plot(residuals, label="Residual (Actual - Predicted)")
        plt.title("Prediction Residuals Over Time")
        plt.xlabel("Time")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)
        file2 = f"plots/residual_plot.png"
        plt.savefig(file2)
        plt.close()

        # 3. Histogram of Errors
        plt.figure(figsize=(10, 5))
        plt.hist(residuals, bins=30, edgecolor='black')
        plt.title("Error Distribution (Residual Histogram)")
        plt.xlabel("Error Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        file3 = f"plots/error_histogram.png"
        plt.savefig(file3)
        plt.close()

        # 4. Scatter: Actual vs Predicted
        plt.figure(figsize=(6, 6))
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot(actual, actual, color='red')
        plt.title("Actual vs Predicted Scatter Plot")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.grid(True)
        file4 = f"plots/scatter_actual_predicted.png"
        plt.savefig(file4)
        plt.close()

        # 5. Rolling RMSE
        errors = (actual.flatten() - predicted.flatten())**2
        rolling_rmse = np.sqrt(pd.Series(errors).rolling(10).mean())
        plt.figure(figsize=(12, 5))
        plt.plot(rolling_rmse, color="purple", label="10-Day Rolling RMSE")
        plt.title("Rolling RMSE Over Time")
        plt.xlabel("Time")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        file5 = f"plots/rolling_rmse.png"
        plt.savefig(file5)
        plt.close()

        # 6. Cumulative Error
        cumulative_error = np.cumsum(residuals)
        plt.figure(figsize=(12, 5))
        plt.plot(cumulative_error, color="orange", label="Cumulative Error")
        plt.title("Cumulative Prediction Error")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Error")
        plt.legend()
        plt.grid(True)
        file6 = f"plots/cumulative_error.png"
        plt.savefig(file6)
        plt.close()
        
        print(f"6 plots saved.")


    def evaluate(self, data):
        if self.model is None:
            print("Error: Model is not loaded.")
            return None

        # --- Data Preparation ---
        feature_data = data[self.features_list].values
        scaled_data = self.scaler.transform(feature_data)
        target_col_index = self.features_list.index(self.target_feature)

        X_test, y_test = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X_test.append(scaled_data[i - self.sequence_length:i, :])
            y_test.append(scaled_data[i, target_col_index])

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # --- Prediction ---
        predictions = self.model.predict(X_test, verbose=0)

        # --- Inverse Transform Actual ---
        actual_dummy = np.zeros((y_test.shape[0], len(self.features_list)))
        actual_dummy[:, target_col_index] = y_test
        actual = self.scaler.inverse_transform(actual_dummy)[:, target_col_index].reshape(-1, 1)

        # --- Inverse Transform Predicted ---
        predicted_dummy = np.zeros((predictions.shape[0], len(self.features_list)))
        predicted_dummy[:, target_col_index] = predictions[:, 0]
        predicted = self.scaler.inverse_transform(predicted_dummy)[:, target_col_index].reshape(-1, 1)

        # --- Metrics Calculation ---
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        r2 = r2_score(actual, predicted)

        print("\nEvaluation Results:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R2 Score: {r2:.4f}")

        # --- Visualization ---
        self._plot_evaluation_results(actual, predicted)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "R2 Score": r2
        }