import yfinance as yf
import pandas as pd

def fetch_price(ticker):
    df = yf.download(ticker, start="2018-01-01", end="2024-12-31")
    df.reset_index(inplace=True)
    df.to_csv(f"{ticker}_price.csv", index=False)
    print(f"{ticker} price data saved.")

if __name__ == "__main__":
    fetch_price("AAPL")
