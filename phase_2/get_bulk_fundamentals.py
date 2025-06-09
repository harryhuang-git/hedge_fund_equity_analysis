import requests
import pandas as pd
import time

API_KEY = 'DWaeGNvxcC0vB8jrqly5hUSqEwanoCdz'  # Replace with your actual API key
tickers = open("tickers.txt").read().splitlines()

def fetch_fundamentals(ticker):
    url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={API_KEY}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if data:
            df = pd.DataFrame(data)
            df.to_csv(f"{ticker}_fundamentals.csv")
            print(f"Saved: {ticker}_fundamentals.csv")
    time.sleep(1.5)  # To respect API rate limit

for t in tickers:
    fetch_fundamentals(t)
