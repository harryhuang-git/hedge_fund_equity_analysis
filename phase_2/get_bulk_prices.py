import yfinance as yf
import pandas as pd

tickers = open("tickers.txt").read().splitlines()

for ticker in tickers:
    print(f"Downloading {ticker}...")
    df = yf.Ticker(ticker).history(start="2023-01-01", end="2024-12-31", auto_adjust=False)
    if not df.empty:
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        df.index.name = "Date"  # Ensures index becomes a column
        df.to_csv(f"{ticker}_price.csv")
        print(f"✅ Saved: {ticker}_price.csv")
    else:
        print(f"⚠️ No data for {ticker}")
