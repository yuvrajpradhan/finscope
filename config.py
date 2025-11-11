from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# --- API Keys ---
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")
API_KEY_NEWS = os.getenv("API_KEY_NEWS")
API_KEY_AV = os.getenv("API_KEY_AV")

# --- Stock Settings ---
TICKER = os.getenv("TICKER", "AAPL")
START_DATE = os.getenv("START_DATE", "2020-01-01")
END_DATE = os.getenv("END_DATE", "2024-01-01")

# --- Model Settings ---
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", 60))
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "models/lstm_model.h5")
ENABLE_TRAINING = os.getenv("ENABLE_TRAINING", "True").lower() in ("true", "1", "yes")

# Optional: Print summary when debugging (comment this out in production)
if __name__ == "__main__":
    print("API_KEY_GEMINI:", bool(API_KEY_GEMINI))
    print("API_KEY_NEWS:", bool(API_KEY_NEWS))
    print("API_KEY_AV:", bool(API_KEY_AV))
    print("TICKER:", TICKER)
    print("LSTM_MODEL_PATH:", LSTM_MODEL_PATH)
    print("ENABLE_TRAINING:", ENABLE_TRAINING)
