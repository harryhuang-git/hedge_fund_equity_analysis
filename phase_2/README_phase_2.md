# 📊 Hedge Fund Equity Ranking – Phase 2

This phase simulates how hedge funds evaluate, rank, and backtest multiple stocks based on value and quality metrics.  
We scale up from single-stock analysis to multi-stock screening and backtesting.

---

## 🔍 Features

- ✅ Downloads price data for 10+ stocks from `yfinance`
- ✅ Fetches fundamentals (ROE, P/E) from Financial Modeling Prep API
- ✅ Ranks stocks based on custom value-quality score (ROE / P/E)
- ✅ Simulates a simple long-only portfolio of top-ranked stocks
- ✅ Plots backtest results

---

## 🧰 Tools Used

- Python 3.9
- pandas
- matplotlib
- yfinance
- requests
- Financial Modeling Prep API

---

## 🚀 How to Run Locally

1. **Create a virtual environment and install requirements:**
   ```bash
   cd hedge_fund_equity_analysis
   python3 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Navigate to Phase 2 folder:**
   ```bash
   cd phase2
   ```
   
3. **Create a stock list:**
   ```bash
   echo -e "AAPL\nMSFT\nGOOG\nAMZN\nNVDA\nTSLA\nMETA\nPEP\nKO\nJPM" > tickers.txt
   ```
   
4. **Download price data:**
   ```bash
   python get_bulk_prices.py
   ```
   
5. **Download fundamental data (make sure your FMP API key is set in the script):**
   ```bash
   python get_bulk_fundamentals.py
   ```
   
6. **Rank stocks by ROE / P/E:**
   ```bash
   python rank_stocks.py
   ``` 
   
7. **Backtest top 5 ranked stocks:**
   ```bash
   python backtest_simple.py
   ``` 
 

📈 Sample Output

Portfolio Backtest (Top 5 by Value Score)
A simple line chart showing portfolio performance over time will appear after running the final script.


📁 Output Files
	•	*_price.csv: Daily price history for each stock
	•	*_fundamentals.csv: Fundamental metrics from FMP API
	•	ranked_stocks.csv: Sorted rankings with Value Scores
	•	Plot output (displayed via matplotlib)



🔑 Notes
	•	Remember to replace the API key in get_bulk_fundamentals.py
	•	Make sure all files are in phase2/ when running scripts
	•	Use Python 3.9+ to avoid compatibility issues
