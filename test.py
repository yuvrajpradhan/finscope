import yfinance as yf
data = yf.download("AAPL", start="2024-01-01", end="2024-11-01")
print(data.head())