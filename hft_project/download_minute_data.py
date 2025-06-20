import tushare as ts
import pandas as pd
import json
import os

TOKEN = '2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211'
START_DATE = '20240101'
END_DATE = '20240630'
CACHE_DIR = 'data/tushare_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Load your universe
with open('data/tushare/metadata.json', 'r') as f:
    meta = json.load(f)
stock_codes = meta['stock_codes']

pro = ts.pro_api(TOKEN)

for ts_code in stock_codes:
    print(f"Downloading {ts_code} minute data...")
    try:
        df = pro.stk_mins(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE, freq='1min')
        if df is not None and not df.empty:
            df.to_csv(f'{CACHE_DIR}/stk_mins_{ts_code}_{START_DATE}_{END_DATE}.csv', index=False)
    except Exception as e:
        print(f"Error downloading {ts_code}: {e}") 