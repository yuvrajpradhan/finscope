# agents/master_agent.py
import numpy as np
import pandas as pd
from collections import defaultdict
from agents.llm_agent import LLMAgent
from agents.lstm_agent import LSTMAgent
from agents.data_fetcher_agent import DataFetcherAgent


class MasterAgent:
    """
    Pipeline:
      1) Fetch stock + news
      2) Daily sentiment (optional)
      3) Attach sentiment to price data
      4) Train/load LSTM (internal preprocessing)
      5) Evaluate
      6) Predict next-day closing price
    """

    def __init__(self, seq_len=60, use_sentiment=False):
        self.use_sentiment = use_sentiment

        # Agents
        self.fetcher = DataFetcherAgent()
        self.llm = LLMAgent()
        self.lstm = LSTMAgent(
            sequence_length=seq_len,
            model_path="models/lstm_model.keras",
            scaler_path="models/lstm_scaler.pkl"
        )

    # ------------------------------------------------------------------
    # MAIN WORKFLOW
    # ------------------------------------------------------------------
    def run(self, stock_symbol: str, start_date: str, end_date: str):
        print(f"\nStarting FinScope workflow for {stock_symbol}...")

        self.lstm.model_path = f"models/{stock_symbol}_model.keras"
        self.lstm.scaler_path = f"models/{stock_symbol}_scaler.pkl"
        # ---------------------------------------------------------
        # 1) FETCH DATA
        # ---------------------------------------------------------
        fetched = self.fetcher.run(stock_symbol, start_date, end_date)
        stock_df = fetched.get("stock_data")
        # stock_df = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
        news_data = fetched.get("news_data")

        if stock_df is None or stock_df.empty:
            print("No stock data found. Exiting.")
            return None

        # ---------------------------------------------------------
        # 2) SENTIMENT (OPTIONAL)
        # ---------------------------------------------------------
        if self.use_sentiment:
            sentiment_map = self._compute_daily_sentiment(news_data)
            stock_df = self._attach_daily_sentiment(stock_df, sentiment_map)
        else:
            print("Skipping sentiment feature.")
            stock_df["Sentiment"] = 0.0   # keeps schema consistent

        print(stock_df)

        # ---------------------------------------------------------
        # 3) TRAIN OR LOAD MODEL
        # ---------------------------------------------------------
        if not self.lstm.load():
            print("Training fresh LSTM model...")
            self.lstm.train(stock_df)   # <--- your LSTM expects raw dataframe
        else:
            print("Loaded saved LSTM model.")

        # ---------------------------------------------------------
        # 4) EVALUATE
        # ---------------------------------------------------------
        print("\nEvaluating model...")
        metrics = self.lstm.evaluate(stock_df)

        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            if "MAPE" in k:
                print(f"{k}: {v:.2f}%")
            else:
                print(f"{k}: {v:.4f}")

        # ---------------------------------------------------------
        # 5) PREDICT NEXT-DAY CLOSE
        # ---------------------------------------------------------
        next_price = self.lstm.predict(stock_df)

        print(f"\nPredicted next Close for {stock_symbol}: {next_price:.2f}")

        return {
            "prediction": next_price,
            "metrics": metrics
        }

    # ------------------------------------------------------------------
    # SENTIMENT HELPERS
    # ------------------------------------------------------------------
    # def _compute_daily_sentiment(self, news_data):
    #     print("\nComputing daily sentiment...")
    #     result = {}

    #     if not news_data:
    #         return result

    #     df = pd.DataFrame(news_data)

    #     if "publishedAt" not in df.columns:
    #         return result

    #     df["date"] = pd.to_datetime(df["publishedAt"]).dt.date

    #     for date, group in df.groupby("date"):
    #         titles = group.get("title", []).fillna("").tolist()
    #         descs = group.get("description", []).fillna("").tolist()
    #         combined = " ".join(t for t in titles + descs if t)

    #         llm_output = self.llm.analyze_sentiment(combined)
    #         score = self._normalize_sentiment(llm_output)

    #         result[pd.Timestamp(date)] = score
    #         print(f"{date}: {score:.2f}")

    #     return result

    def _compute_daily_sentiment(self, news_data):
        if not news_data:
            return {}

        df = pd.DataFrame(news_data)

        if "publishedAt" not in df.columns:
            return {}

        df["date"] = pd.to_datetime(df["publishedAt"], errors='coerce').dt.strftime('%Y-%m-%d')
        df.dropna(subset=['date'], inplace=True)
        
        df['title'] = df['title'].fillna('')
        df['description'] = df['description'].fillna('')
        
        df['combined_text'] = df['title'] + " --- " + df['description']

        daily_combined_text = df.groupby("date")['combined_text'].apply(' --- '.join).to_dict()

        # print(daily_combined_text)

        if not daily_combined_text:
            return {}
            
        try:
            # Assuming self.llm.analyze_daily_batch_sentiment returns {date: score}
            llm_scores = self.llm.analyze_daily_batch_sentiment(daily_combined_text)
        except Exception:
            llm_scores = {}

        print(llm_scores)

        result = {
            pd.Timestamp(date): score 
            for date, score in llm_scores.items()
        }
        
        return result

    # def _normalize_sentiment(self, raw):
    #     """
    #     Converts the LLM's categorical text output ('Positive', 'Negative', 'Neutral') 
    #     into a standardized numerical score (-1.0 to +1.0).
    #     """
    #     if not raw:
    #         return 0.0

    #     # Ensure input is treated as a string and converted to lowercase
    #     label = str(raw).strip().lower()

    #     # Direct mapping based on your LLM's output
    #     if "positive" in label:
    #         return 1.0
    #     elif "negative" in label:
    #         return -1.0
    #     else: # Catches "Neutral" and any unexpected output
    #         return 0.0

    def _attach_daily_sentiment(self, df_prices, sentiment):
        df = df_prices.copy()
        df.index = pd.to_datetime(df.index).normalize()

        sentiment_series = pd.Series(sentiment)
        
        df["Sentiment"] = df.index.map(sentiment_series).fillna(0.0)

        return df