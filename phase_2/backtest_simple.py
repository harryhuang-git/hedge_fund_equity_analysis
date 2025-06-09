import pandas as pd
import matplotlib.pyplot as plt

ranked = pd.read_csv("ranked_stocks.csv").head(5)
tickers = ranked['Ticker'].tolist()

portfolio = pd.DataFrame()
for t in tickers:
    try:
        df = pd.read_csv(f"{t}_price.csv")
        if 'Date' in df.columns and 'Adj Close' in df.columns:
            df = df[['Date', 'Adj Close']]
            df = df.rename(columns={"Adj Close": t})
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            portfolio = pd.concat([portfolio, df], axis=1)
        else:
            print(f"Skipping {t} — missing required columns.")
    except Exception as e:
        print(f"Error processing {t}: {e}")

portfolio = portfolio.dropna()
portfolio['Mean'] = portfolio.mean(axis=1)
portfolio['Mean'].plot(title="Backtest: Top 5 Stocks by Value Score")
plt.show()
