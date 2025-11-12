import pandas as pd
import requests
import config
from datetime import datetime, timedelta


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

    def fetch_alpha_vantage_sentiment(self, ticker: str, start_date_str: str, end_date_str: str, limit: int = 200):
        """
        Fetches news and sentiment data for a given ticker over a specified date range 
        using the Alpha Vantage News & Sentiment endpoint.
        """
        print(f"\nFetching Alpha Vantage sentiment for '{ticker}' from {start_date_str} to {end_date_str}...")

        # 1. Check for API Key
        if not self.alpha_key: # Assuming you have an 'self.av_api_key' attribute
            print("Missing Alpha Vantage API key. Skipping sentiment fetching.")
            return []

        # 2. Format Dates for AV (YYYYMMDDTHHMMSS)
        # AV typically requires a precise timestamp for time_from and time_to
        end_date_obj = datetime.now()
        start_date_obj = end_date_obj - timedelta(days=30)
        
        start_date_str = start_date_obj.strftime('%Y%m%d')
        end_date_str = end_date_obj.strftime('%Y%m%d')
        
        time_from = f"{start_date_str}T0000"
        time_to = f"{end_date_str}T0000"
        # time_from = f"{start_date_str.replace('-', '')}T0000"
        # time_to = f"{end_date_str.replace('-', '')}T0000"

        # 3. Construct the AV URL
        url = "https://www.alphavantage.co/query"

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "topics": "financial_markets",
            "time_from": time_from,
            "time_to": time_to,
            "limit": limit,
            "apikey": self.alpha_key
        }

        try:
            response = self.news_session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            print(data)

            # Check for error messages returned by AV
            if 'Error Message' in data:
                print(f"Alpha Vantage API Error: {data['Error Message']}")
                return []
            
            # Articles are under the 'feed' key
            articles = data.get("feed", [])
            print(f"Found {len(articles)} relevant news articles in the range.")

            # 4. Extract Relevant Data, including Sentiment Score
            sentiment_data = []
            for a in articles:
                # Alpha Vantage returns sentiment scores per ticker mentioned in the article
                ticker_sentiment = a.get("ticker_sentiment", [])
                
                # Find the sentiment data specifically for the requested ticker
                target_sentiment = next((ts for ts in ticker_sentiment if ts.get("ticker") == ticker), None)

                if target_sentiment:
                    sentiment_data.append({
                        "title": a.get("title"),
                        "time_published": a.get("time_published"), # Timestamp in AV format (YYYYMMDDTHHMMSS)
                        "source": a.get("source"),
                        # Key Extraction: The score your LSTM needs
                        "sentiment_score": float(target_sentiment.get("ticker_sentiment_score", 0.0)),
                        "relevance_score": float(target_sentiment.get("ticker_sentiment_label", 0.0)),
                    })
            
            return sentiment_data

        except Exception as e:
            print(f"Error fetching Alpha Vantage news and sentiment: {e}")
            return []


    def run(self, ticker: str = None, start_date: str = None, end_date: str = None):
        ticker = ticker or config.TICKER
        start_date = start_date or config.START_DATE
        end_date = end_date or config.END_DATE

        # stock_data = self.fetch_stock_data(ticker, start_date, end_date)
        stock_data = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)

        if stock_data is not None:
            stock_data.to_csv("stock_data.csv")
            print("Stock data saved to stock_data.csv")

        news_data = self.fetch_news_data(ticker, max_articles=50)
        # print(news_data)

        # news_data = []

        # av_news_data = self.fetch_alpha_vantage_sentiment(ticker, start_date, end_date, limit=150)
        # print(av_news_data)


        return {
            "stock_data": stock_data,
            "news_data": news_data
        }
