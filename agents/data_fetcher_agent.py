import yfinance as yf
import pandas as pd
import requests
import config


class DataFetcherAgent:
    """
    The Data Fetcher Agent is responsible for retrieving quantitative (stock price)
    and qualitative (news, reports) financial data from APIs.
    """

    def __init__(self):
        self.news_api_key = config.API_KEY_NEWS
        if not self.news_api_key:
            print("Warning: No News API key found in config.py.")
        else:
            print("DataFetcherAgent initialized successfully with News API key.")

    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str):
        """
        Fetches historical stock data from Yahoo Finance.

        Args:
            ticker (str): The stock ticker symbol.
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: Historical stock price data.
        """
        print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date, impersonate=False, progress=False)
            if stock_data.empty:
                print(f"No stock data found for {ticker}. Check if the ticker is valid.")
                return None
            print("Stock data fetched successfully.")
            return stock_data
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None

    def fetch_news_data(self, query: str, max_articles: int = 10):
        """
        Fetches recent financial news related to a specific company or keyword.

        Args:
            query (str): Search keyword (e.g., company name or ticker).
            max_articles (int): Maximum number of articles to fetch.

        Returns:
            list[dict]: List of news articles with title, description, and URL.
        """
        print(f"Fetching financial news for '{query}'...")

        if not self.news_api_key:
            print("News API key missing in config.py. Skipping news fetching.")
            return []

        # Create a more specific financial query
        search_query = f"{query} stock OR {query} shares OR {query} finance OR {query} earnings"

        # Build API request
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={search_query}&apiKey={self.news_api_key}&language=en&sortBy=publishedAt&pageSize={max_articles}"
        )

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
            print(f"Found {len(articles)} financial news articles for '{query}'.")
            return [
                {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "publishedAt": article.get("publishedAt"),
                    "source": article.get("source", {}).get("name"),
                }
                for article in articles
            ]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news data: {e}")
            return []

    def run(self, ticker: str = None, start_date: str = None, end_date: str = None):
        """
        Fetches both stock and news data for a given ticker symbol.

        Args:
            ticker (str): Stock ticker symbol (defaults to config.TICKER)
            start_date (str): Start date for stock data (defaults to config.START_DATE)
            end_date (str): End date for stock data (defaults to config.END_DATE)

        Returns:
            dict: Combined dictionary with stock and news data.
        """
        ticker = ticker or config.TICKER
        start_date = start_date or config.START_DATE
        end_date = end_date or config.END_DATE

        stock_data = self.fetch_stock_data(ticker, start_date, end_date)
        news_data = self.fetch_news_data(ticker)

        return {
            "stock_data": stock_data,
            "news_data": news_data
        }
