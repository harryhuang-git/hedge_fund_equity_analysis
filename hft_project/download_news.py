import tushare as ts
import os

TOKEN = '2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211'
START_DATE = '20240101'
END_DATE = '20240630'
CACHE_DIR = 'data/tushare_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

pro = ts.pro_api(TOKEN)

print("Downloading news...")
news = pro.news(start_date=START_DATE, end_date=END_DATE)
if news is not None and not news.empty:
    news.to_csv(f'{CACHE_DIR}/news_{START_DATE}_{END_DATE}.csv', index=False)

print("Downloading major news...")
major_news = pro.major_news(start_date=START_DATE, end_date=END_DATE)
if major_news is not None and not major_news.empty:
    major_news.to_csv(f'{CACHE_DIR}/major_news_{START_DATE}_{END_DATE}.csv', index=False)

print("Downloading hot topics...")
ths_hot = pro.ths_hot(date=END_DATE)
if ths_hot is not None and not ths_hot.empty:
    ths_hot.to_csv(f'{CACHE_DIR}/ths_hot_{END_DATE}.csv', index=False) 