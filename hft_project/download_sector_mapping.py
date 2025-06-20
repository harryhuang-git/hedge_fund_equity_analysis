import tushare as ts
import pandas as pd
import json

# === CONFIGURATION ===
TUSHARE_TOKEN = '2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211'  # User's original token
UNIVERSE_FILE = 'data/tushare/metadata.json'  # Path to your metadata.json
OUTPUT_FILE = 'data/stock_sector_mapping.csv'

# === INIT TUSHARE ===
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# === LOAD UNIVERSE ===
with open(UNIVERSE_FILE, 'r') as f:
    meta = json.load(f)
stock_codes = meta['stock_codes']

# === DOWNLOAD INDUSTRY DATA ===
print("Downloading industry info from Tushare...")
df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry,market,list_date')

# Filter to your universe
df = df[df['ts_code'].isin(stock_codes)]

# === SAVE TO CSV ===
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"Saved sector/industry mapping for {len(df)} stocks to {OUTPUT_FILE}") 