import pandas as pd
import numpy as np
from agents.llm_agent import LLMAgent
from agents.lstm_agent import LSTMAgent
from agents.data_preprocessor_agent import DataPreprocessorAgent
from agents.data_fetcher_agent import DataFetcherAgent


class MasterAgent:
    """
    The Master Agent orchestrates the entire FinScope workflow:
      1. Fetches stock + news data
      2. Analyzes daily sentiment using LLM
      3. Aligns daily sentiment with stock data
      4. Preprocesses the combined dataset
      5. Predicts next/current price using LSTM
    """

    def __init__(self):
        self.fetcher = DataFetcherAgent()
        self.preprocessor = DataPreprocessorAgent()
        self.llm = LLMAgent()
        self.lstm = LSTMAgent()

    def run(self, stock_symbol: str, start_date: str, end_date: str):
        print(f"\nStarting FinScope workflow for {stock_symbol}...")

        # Step 1: Fetch stock and news data
        fetched = self.fetcher.run(stock_symbol, start_date, end_date)
        stock_df = fetched.get("stock_data")
        news_data = fetched.get("news_data")

        if stock_df is None or stock_df.empty:
            print("No stock data available. Terminating pipeline.")
            return None

        # Step 2: Run LLM on news data → daily sentiment mapping
        sentiment_by_date = self._compute_daily_sentiment(news_data)

        # Step 3: Attach daily sentiment to price data
        stock_with_sentiment = self._attach_daily_sentiment(stock_df, sentiment_by_date)

        # Step 4: Preprocess data for LSTM input
        processed_data = self.preprocessor.preprocess(stock_with_sentiment)

        # Step 5: Predict next price using LSTM
        predicted_price = self.lstm.predict(processed_data)

        print(f"\nFinal Predicted Price for {stock_symbol}: {predicted_price:.2f}")
        return predicted_price

    def _compute_daily_sentiment(self, news_data):
        """
        Groups news articles by date, analyzes each day's combined text
        using LLM, and returns {date: sentiment_score}.
        """
        print("\nComputing daily sentiment from news data...")

        sentiment_by_date = {}
        if not news_data:
            print("No news data found for sentiment analysis.")
            return sentiment_by_date

        df_news = pd.DataFrame(news_data)
        if "publishedAt" not in df_news.columns:
            print("Missing 'publishedAt' field in news data.")
            return sentiment_by_date

        df_news["date"] = pd.to_datetime(df_news["publishedAt"]).dt.date

        for date, group in df_news.groupby("date"):
            # Combine titles + descriptions for the day
            texts = list(filter(None, group["title"].tolist() + group["description"].tolist()))
            combined_text = " ".join(texts)

            sentiment_result = self.llm.analyze_sentiment(combined_text)
            sentiment_score = self._extract_sentiment_score(sentiment_result)

            sentiment_by_date[pd.Timestamp(date)] = sentiment_score
            print(f"{date}: Sentiment Score = {sentiment_score:.2f}")

        return sentiment_by_date

    def _extract_sentiment_score(self, sentiment_result):
        """
        Normalize any LLM output to a float score in [-1, 1].
        """
        if sentiment_result is None:
            return 0.0

        score = 0.0
        label = ""

        if isinstance(sentiment_result, dict):
            score = sentiment_result.get("score") or sentiment_result.get("confidence") or 0.0
            label = str(sentiment_result.get("label") or sentiment_result.get("sentiment") or "").lower()
        else:
            label = str(sentiment_result).lower()

        if "pos" in label or "bull" in label:
            score = 0.8
        elif "neg" in label or "bear" in label:
            score = -0.8

        # Normalize if in [0,1]
        if 0.0 <= score <= 1.0:
            score = score * 2 - 1

        return float(np.clip(score, -1.0, 1.0))

    def _attach_daily_sentiment(self, df_prices, sentiment_by_date, sentiment_col="Sentiment"):
        """
        Adds a sentiment column to price data aligned by date.
        Missing dates get sentiment = 0.0
        """
        print("\nAttaching sentiment to stock price data...")

        df = df_prices.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df[sentiment_col] = [
            sentiment_by_date.get(pd.Timestamp(date), 0.0) for date in df.index
        ]

        print("Sentiment column attached successfully.")
        return df
