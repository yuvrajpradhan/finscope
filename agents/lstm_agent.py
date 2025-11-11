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
    """
    Clean, prediction-safe LSTM Agent for stock price forecasting.
    """

    def __init__(self, sequence_length=60, model_path="models/lstm_model.keras", scaler_path="models/lstm_scaler.pkl"):
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.scaler_path = scaler_path

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        print("LSTMAgent initialized.")

    # -----------------------------
    # DATA PREPROCESSING
    # -----------------------------
    def _preprocess_data(self, data):
        """
        Turns raw 'Close' prices into LSTM-ready sequences.
        """
        close_prices = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    # -----------------------------
    # MODEL BUILDING
    # -----------------------------
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

    # -----------------------------
    # TRAINING
    # -----------------------------
    def train(self, data):
        print("Training LSTM model...")

        X_train, y_train = self._preprocess_data(data)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        self.build_model((X_train.shape[1], 1))

        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)

        # Save scaler
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"Model saved to {self.model_path}")
        print(f"Scaler saved to {self.scaler_path}")

    # -----------------------------
    # LOADING MODEL + SCALER
    # -----------------------------
    def load(self):
        if not os.path.exists(self.model_path):
            print("Error: Model file not found.")
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

    # -----------------------------
    # PREDICTION
    # -----------------------------
    def predict(self, data):
        if self.model is None:
            print("Error: Model is not loaded.")
            return None

        if len(data) < self.sequence_length:
            print("Error: Not enough data for prediction.")
            return None

        print("Predicting next closing price...")

        last_seq = data['Close'][-self.sequence_length:].values.reshape(-1, 1)
        scaled_seq = self.scaler.transform(last_seq)

        X_test = scaled_seq.reshape(1, self.sequence_length, 1)

        pred_scaled = self.model.predict(X_test)
        pred_price = self.scaler.inverse_transform(pred_scaled)

        return float(pred_price[0][0])

    # -----------------------------
    # EVALUATION
    # -----------------------------
    def evaluate(self, data):
        if self.model is None:
            print("Error: Model is not loaded.")
            return None

        # Make output directory
        # os.makedirs("plots", exist_ok=True)

        # Timestamp for unique file names
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        close_prices = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(close_prices)

        X_test, y_test = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X_test.append(scaled_data[i - self.sequence_length:i, 0])
            y_test.append(scaled_data[i, 0])

        X_test = np.array(X_test).reshape(-1, self.sequence_length, 1)
        y_test = np.array(y_test)

        predictions = self.model.predict(X_test)

        actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        predicted = self.scaler.inverse_transform(predictions)

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

        # # ------------------------------------------------------------
        # # 1. Actual vs Predicted
        # plt.figure(figsize=(12, 6))
        # plt.plot(actual, label="Actual Price", linewidth=2)
        # plt.plot(predicted, label="Predicted Price", linewidth=2)
        # plt.title("Actual vs Predicted Stock Price")
        # plt.xlabel("Time")
        # plt.ylabel("Price")
        # plt.legend()
        # plt.grid(True)
        # file1 = f"../plots/actual_vs_predicted_{ts}.png"
        # plt.savefig(file1)
        # plt.close()

        # # ------------------------------------------------------------
        # # 2. Residual Plot
        # residuals = actual.flatten() - predicted.flatten()
        # plt.figure(figsize=(12, 5))
        # plt.plot(residuals, label="Residual (Actual - Predicted)")
        # plt.title("Prediction Residuals Over Time")
        # plt.xlabel("Time")
        # plt.ylabel("Error")
        # plt.legend()
        # plt.grid(True)
        # file2 = f"../plots/residual_plot_{ts}.png"
        # plt.savefig(file2)
        # plt.close()

        # # ------------------------------------------------------------
        # # 3. Histogram of Errors
        # plt.figure(figsize=(10, 5))
        # plt.hist(residuals, bins=30, edgecolor='black')
        # plt.title("Error Distribution (Residual Histogram)")
        # plt.xlabel("Error Value")
        # plt.ylabel("Frequency")
        # plt.grid(True)
        # file3 = f"../plots/error_histogram_{ts}.png"
        # plt.savefig(file3)
        # plt.close()

        # # ------------------------------------------------------------
        # # 4. Scatter: Actual vs Predicted
        # plt.figure(figsize=(6, 6))
        # plt.scatter(actual, predicted, alpha=0.5)
        # plt.plot(actual, actual, color='red')
        # plt.title("Actual vs Predicted Scatter Plot")
        # plt.xlabel("Actual Price")
        # plt.ylabel("Predicted Price")
        # plt.grid(True)
        # file4 = f"../plots/scatter_actual_predicted_{ts}.png"
        # plt.savefig(file4)
        # plt.close()

        # # ------------------------------------------------------------
        # # 5. Rolling RMSE
        # errors = (actual.flatten() - predicted.flatten())**2
        # rolling_rmse = np.sqrt(pd.Series(errors).rolling(10).mean())
        # plt.figure(figsize=(12, 5))
        # plt.plot(rolling_rmse, color="purple", label="10-Day Rolling RMSE")
        # plt.title("Rolling RMSE Over Time")
        # plt.xlabel("Time")
        # plt.ylabel("RMSE")
        # plt.legend()
        # plt.grid(True)
        # file5 = f"../plots/rolling_rmse_{ts}.png"
        # plt.savefig(file5)
        # plt.close()

        # # ------------------------------------------------------------
        # # 6. Cumulative Error
        # cumulative_error = np.cumsum(residuals)
        # plt.figure(figsize=(12, 5))
        # plt.plot(cumulative_error, color="orange", label="Cumulative Error")
        # plt.title("Cumulative Prediction Error")
        # plt.xlabel("Time")
        # plt.ylabel("Cumulative Error")
        # plt.legend()
        # plt.grid(True)
        # file6 = f"../plots/cumulative_error_{ts}.png"
        # plt.savefig(file6)
        # plt.close()

        # print("\nVisualization files saved:")
        # print(file1)
        # print(file2)
        # print(file3)
        # print(file4)
        # print(file5)
        # print(file6)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "R2 Score": r2
        }

