import pandas as pd
import matplotlib.pyplot as plt

ticker = "AAPL"
df = pd.read_csv(f"{ticker}_income.csv")
df = df.sort_values("date")  # oldest to newest

# Convert to numeric (safe against API weirdness)
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
df["netIncome"] = pd.to_numeric(df["netIncome"], errors="coerce")

# Drop rows with missing values
df = df.dropna(subset=["revenue", "netIncome"])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["revenue"] / 1e9, label="Revenue (Billion $)", marker='o')
plt.plot(df["date"], df["netIncome"] / 1e9, label="Net Income (Billion $)", marker='o')
plt.title(f"{ticker} Revenue & Net Income (Last 5 Years)")
plt.xlabel("Year")
plt.ylabel("Amount ($B)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{ticker.lower()}_summary_chart.png")
plt.show()
