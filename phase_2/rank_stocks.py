import pandas as pd
import os

results = []
for file in os.listdir():
    if file.endswith("_fundamentals.csv"):
        df = pd.read_csv(file)
        ticker = file.split("_")[0]
        try:
            roe = df.loc[0, 'roeTTM']
            pe = df.loc[0, 'peRatioTTM']
            results.append((ticker, roe, pe))
        except:
            continue

ranked = pd.DataFrame(results, columns=["Ticker", "ROE", "PE"])
ranked["Value_Score"] = ranked["ROE"] / ranked["PE"]
ranked.sort_values("Value_Score", ascending=False, inplace=True)
print(ranked)
ranked.to_csv("ranked_stocks.csv", index=False)
