import requests
import pandas as pd

API_KEY = "DWaeGNvxcC0vB8jrqly5hUSqEwanoCdz"  # Your actual FMP API key

def get_income_statement(ticker):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=5&apikey={API_KEY}"
    response = requests.get(url)
    df = pd.DataFrame(response.json())
    df.to_csv(f"{ticker}_income.csv", index=False)
    print(f"{ticker} income statement saved.")

def get_balance_sheet(ticker):
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?limit=5&apikey={API_KEY}"
    response = requests.get(url)
    df = pd.DataFrame(response.json())
    df.to_csv(f"{ticker}_balance.csv", index=False)
    print(f"{ticker} balance sheet saved.")

if __name__ == "__main__":
    ticker = "AAPL"
    get_income_statement(ticker)
    get_balance_sheet(ticker)
