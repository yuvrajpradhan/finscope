from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Load variables from .env file
load_dotenv()

# --- API Keys ---
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")
API_KEY_NEWS = os.getenv("API_KEY_NEWS")
API_KEY_AV = os.getenv("API_KEY_AV")

# --- Stock Settings ---
TICKER = os.getenv("TICKER", "AAPL")

# --- Dynamic Date Calculation ---
end_date_obj = datetime.now()
END_DATE = end_date_obj.strftime('%Y-%m-%d')

start_date_obj = end_date_obj - timedelta(days=10)
START_DATE = start_date_obj.strftime('%Y-%m-%d')
# --------------------------------------------------

# --- Model Settings ---
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", 60))
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "models/lstm_model.h5")
ENABLE_TRAINING = os.getenv("ENABLE_TRAINING", "True").lower() in ("true", "1", "yes")