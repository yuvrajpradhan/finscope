import pandas as pd
import requests
import config
from datetime import datetime


class DataFetcherAgent:
    """
    Fetches:
      • Historical stock price data using Alpha Vantage (recommended)
      • Financial news using NewsAPI
    """

    def __init__(self):
        self.alpha_key = config.API_KEY_AV
        self.news_api_key = config.API_KEY_NEWS

        if not self.alpha_key:
            print("Warning: No Alpha Vantage API key found in config.py.")
        if not self.news_api_key:
            print("Warning: No News API key found in config.py.")

        # For News API
        self.news_session = requests.Session()
        self.news_session.headers.update({
            "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
        })

        print("DataFetcherAgent initialized successfully.")

    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str):
        print(f"\nFetching stock data for {ticker} ({start_date} -> {end_date}) using Alpha Vantage...")

        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY&symbol={ticker}"
            f"&outputsize=full&apikey={self.alpha_key}"
        )

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "Time Series (Daily)" not in data:
                print("AlphaVantage Error:", data)
                return None

            ts = data["Time Series (Daily)"]
            df = pd.DataFrame(ts).T
            df.index = pd.to_datetime(df.index)

            # Rename columns
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close"
                # "5. adjusted close": "Adj Close",
                # "6. volume": "Volume"
            })

            # df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
            df = df[["Open", "High", "Low", "Close"]]

            # Convert numeric values from strings to float
            df = df.astype(float)

            # Filter by date range
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            df = df.loc[mask]

            if df.empty:
                print(f"No stock data found for {ticker} in the given date range.")
                return None

            print("Stock data fetched successfully from Alpha Vantage.")
            return df.sort_index()

        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None

    def fetch_news_data(self, query: str, max_articles: int = 10):
        print(f"\nFetching news for '{query}'...")

        if not self.news_api_key:
            print("Missing News API key. Skipping news fetching.")
            return []

        search_query = f"{query} stock OR {query} finance OR {query} earnings OR {query} results"

        url = (
            "https://newsapi.org/v2/everything?"
            f"q={search_query}"
            f"&apiKey={self.news_api_key}"
            f"&language=en"
            f"&sortBy=publishedAt"
            f"&pageSize={max_articles}"
        )

        try:
            response = self.news_session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            print(f"Found {len(articles)} financial news articles.")

            return [
                {
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "url": a.get("url"),
                    "publishedAt": a.get("publishedAt"),
                    "source": a.get("source", {}).get("name"),
                }
                for a in articles
            ]

        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def run(self, ticker: str = None, start_date: str = None, end_date: str = None):
        ticker = ticker or config.TICKER
        start_date = start_date or config.START_DATE
        end_date = end_date or config.END_DATE

        stock_data = self.fetch_stock_data(ticker, start_date, end_date)

        if stock_data is not None:
            stock_data.to_csv("stock_data.csv")
            print("Stock data saved to stock_data.csv")

        news_data = self.fetch_news_data(ticker)

        return {
            "stock_data": stock_data,
            "news_data": news_data
        }
