import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("AAPL_price.csv")

# Convert Close to float (force numeric conversion)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# Drop any rows with missing or invalid Close values
df = df.dropna(subset=['Close'])

# Calculate moving averages
df['MA_20'] = df['Close'].rolling(20).mean()
df['MA_50'] = df['Close'].rolling(50).mean()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label="Close Price")
plt.plot(df['Date'], df['MA_20'], label="20-day MA")
plt.plot(df['Date'], df['MA_50'], label="50-day MA")

plt.title("AAPL Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("aapl_price_plot.png")
plt.show()
