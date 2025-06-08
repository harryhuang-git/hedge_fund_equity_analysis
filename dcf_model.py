import pandas as pd

def run_dcf(ticker):
    # Load income statement
    df = pd.read_csv(f"{ticker}_income.csv")
    
    # Clean and sort by date ascending
    df = df.sort_values("date")
    
    # Use only the last 5 years of revenue
    revenue = df["revenue"].astype(float).tail(5)
    
    # Estimate average revenue growth
    growth_rate = revenue.pct_change().mean()
    last_revenue = revenue.iloc[-1]

    # Project revenue for 5 years
    future_revenue = [last_revenue * ((1 + growth_rate) ** i) for i in range(1, 6)]

    # Estimate FCF (Free Cash Flow) using a fixed margin
    fcf_margin = 0.15  # 15% of revenue assumed as FCF
    future_fcf = [rev * fcf_margin for rev in future_revenue]

    # Discount FCF to present value
    discount_rate = 0.10
    npv_fcf = sum([fcf / ((1 + discount_rate) ** i) for i, fcf in enumerate(future_fcf, 1)])

    # Terminal value (Gordon growth)
    terminal_value = (future_fcf[-1] * (1 + 0.03)) / (discount_rate - 0.03)
    npv_terminal = terminal_value / ((1 + discount_rate) ** 5)

    total_value = npv_fcf + npv_terminal

    print(f"\n📊 Intrinsic Value of {ticker}: ${round(total_value / 1e9, 2)}B")

if __name__ == "__main__":
    run_dcf("AAPL")
