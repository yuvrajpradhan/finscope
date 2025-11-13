import yfinance as yf
import pandas as pd
import requests
import config
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

API_KEY_NEWS = os.getenv("API_KEY_NEWS")

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        
    Returns:
        pandas.DataFrame: A DataFrame with historical stock data, or None if failed.
    """
    print(f"Fetching historical stock data for {ticker} from {start_date} to {end_date}...")
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data found for ticker {ticker}. It might be delisted or the ticker is incorrect.")
            return None
        print("Stock data fetched successfully.")
        return stock_data
    except Exception as e:
        print(f"An error occurred while fetching stock data: {e}")
        return None

def fetch_news_data(query):
    """
    Fetches news articles related to a query from a news API.
    
    NOTE: This is a generic template. You will need to adapt it to your specific
          news API provider (e.g., NewsAPI.org, Alpha Vantage, etc.).
    
    Args:
        query (str): The search term for news (e.g., the company name).
        
    Returns:
        list: A list of news article dictionaries, or an empty list if failed.
    """
    print(f"Fetching news for '{query}'...")
    if not config.API_KEY_NEWS or config.API_KEY_NEWS == "YOUR_NEWS_API_KEY":
        print("WARNING: News API key not found in config.py. Skipping news fetching.")
        return []
        
    # Example for NewsAPI.org - you may need to change the URL and params
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={config.API_KEY_NEWS}&language=en&sortBy=publishedAt&pageSize=5"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        articles = data.get("articles", [])
        print(f"Found {len(articles)} news articles.")
        return articles
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching news data: {e}")
        return []