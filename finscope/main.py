import config
from data_collection.fetcher import fetch_stock_data, fetch_news_data
from agents.lstm_agent import LSTMAgent
from agents.llm_agent import LLMAgent
from agents.master_agent import MasterAgent
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

TICKER = os.getenv("TICKER")
START_DATE = os.getenv("START_DATE")
END_DATE = os.getenv("END_DATE")
LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH"))
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH")
ENABLE_TRAINING = os.getenv("ENABLE_TRAINING") == 'True'

def run():
    """
    Main function to run the entire prediction pipeline.
    """
    # --- 1. Data Collection ---
    stock_data = fetch_stock_data(config.TICKER, config.START_DATE, config.END_DATE)
    if stock_data is None:
        return # Exit if stock data fetching fails
        
    news_articles = fetch_news_data(config.TICKER)

    # --- 2. LSTM Agent ---
    lstm_agent = LSTMAgent(
        sequence_length=config.LSTM_SEQUENCE_LENGTH, 
        model_path=config.LSTM_MODEL_PATH
    )
    
    # Train or load the model
    if config.ENABLE_TRAINING:
        lstm_agent.train(stock_data)
    else:
        if not lstm_agent.load():
            print("Cannot proceed without a trained LSTM model. Please enable training or provide a model file.")
            return

    # Make a prediction
    predicted_price = lstm_agent.predict(stock_data)
    if predicted_price is None:
        return # Exit if prediction fails

    # --- 3. LLM Agent ---
    sentiments = []
    events = []
    if news_articles:
        try:
            llm_agent = LLMAgent()
            for article in news_articles:
                # Use title for sentiment, content for event extraction
                title = article.get('title', '')
                content = article.get('content', '') or title # Fallback to title if content is empty
                
                if title:
                    sentiments.append(llm_agent.analyze_sentiment(title))
                if content:
                    events.append(llm_agent.extract_events(content))
        except ValueError as e:
            print(f"Error initializing LLM Agent: {e}")
            # Continue without LLM analysis
            sentiments, events = ['Neutral'], [{'event_detected': False, 'event_type': 'none'}]
    else:
        sentiments, events = ['Neutral'], [{'event_detected': False, 'event_type': 'none'}]


    # Aggregate LLM results (simple approach: take the most common sentiment and first detected event)
    final_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "Neutral"
    final_event = next((e for e in events if e.get('event_detected')), {'event_detected': False, 'event_type': 'none'})

    # --- 4. Master Agent ---
    master_agent = MasterAgent()
    # FIX: Use .item() to extract the standard Python float from the numpy value.
    current_price = stock_data['Close'].values[-1].item()
    
    decision = master_agent.decide(
        lstm_prediction=predicted_price,
        current_price=current_price,
        sentiment=final_sentiment,
        event=final_event
    )
    
    # --- 5. Final Output ---
    print("\n===================================")
    print(f"      Stock Analysis for {config.TICKER}")
    print("===================================")
    print(f"Final Recommendation: {decision}")
    print("===================================\n")


if __name__ == "__main__":
    # To run the project:
    # 1. Create a project directory.
    # 2. Create subdirectories: `data_collection`, `agents`, `models`.
    # 3. Create an empty `__init__.py` file inside `data_collection` and `agents`.
    # 4. Save each code block above into its respective file (e.g., config.py, main.py).
    # 5. Install requirements: `pip install -r requirements.txt`
    # 6. Fill in your API keys in `config.py`.
    # 7. Run from the project's root directory: `python main.py`
    run()