import tushare as ts
import pandas as pd
import os
import time
from functools import lru_cache

class TushareDataFetcher:
    def __init__(self, token, cache_dir='data/tushare_cache', max_retries=5, backoff_factor=2):
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, name, **kwargs):
        key = '_'.join([str(v) for v in kwargs.values()])
        return os.path.join(self.cache_dir, f'{name}_{key}.csv')

    def _fetch_with_cache(self, name, fetch_func, force_refresh=False, **kwargs):
        path = self._cache_path(name, **kwargs)
        if os.path.exists(path) and not force_refresh:
            return pd.read_csv(path)
        for attempt in range(self.max_retries):
            try:
                df = fetch_func(**kwargs)
                if df is not None and not df.empty:
                    df.to_csv(path, index=False)
                    return df
            except Exception as e:
                print(f"Tushare API error on {name}: {e}. Retrying...")
                time.sleep(self.backoff_factor ** attempt)
        raise RuntimeError(f"Failed to fetch {name} after {self.max_retries} attempts.")

    def get_minute_data(self, ts_code, start_date, end_date, freq='1min', force_refresh=False):
        return self._fetch_with_cache(
            'stk_mins',
            lambda ts_code, start_date, end_date, freq: self.pro.stk_mins(ts_code=ts_code, start_date=start_date, end_date=end_date, freq=freq),
            ts_code=ts_code, start_date=start_date, end_date=end_date, freq=freq, force_refresh=force_refresh
        )

    def get_technical(self, ts_code, start_date, end_date, force_refresh=False):
        # Example: MACD, RSI, PSY
        macd = self._fetch_with_cache('macd', lambda ts_code, start_date, end_date: self.pro.macd(ts_code=ts_code, start_date=start_date, end_date=end_date), ts_code=ts_code, start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        rsi = self._fetch_with_cache('rsi', lambda ts_code, start_date, end_date: self.pro.rsi(ts_code=ts_code, start_date=start_date, end_date=end_date), ts_code=ts_code, start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        psy = self._fetch_with_cache('psy', lambda ts_code, start_date, end_date: self.pro.psy(ts_code=ts_code, start_date=start_date, end_date=end_date), ts_code=ts_code, start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        return {'macd': macd, 'rsi': rsi, 'psy': psy}

    def get_tick_data(self, ts_code, trade_date, force_refresh=False):
        return self._fetch_with_cache(
            'stk_ticks',
            lambda ts_code, trade_date: self.pro.stk_ticks(ts_code=ts_code, trade_date=trade_date),
            ts_code=ts_code, trade_date=trade_date, force_refresh=force_refresh
        )

    def get_news(self, start_date, end_date, force_refresh=False):
        news = self._fetch_with_cache('news', lambda start_date, end_date: self.pro.news(start_date=start_date, end_date=end_date), start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        major_news = self._fetch_with_cache('major_news', lambda start_date, end_date: self.pro.major_news(start_date=start_date, end_date=end_date), start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        return {'news': news, 'major_news': major_news}

    def get_qa(self, start_date, end_date, force_refresh=False):
        qa_sh = self._fetch_with_cache('irm_qa_sh', lambda start_date, end_date: self.pro.irm_qa_sh(start_date=start_date, end_date=end_date), start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        qa_sz = self._fetch_with_cache('irm_qa_sz', lambda start_date, end_date: self.pro.irm_qa_sz(start_date=start_date, end_date=end_date), start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        return {'qa_sh': qa_sh, 'qa_sz': qa_sz}

    def get_hot_topics(self, date, force_refresh=False):
        ths_hot = self._fetch_with_cache('ths_hot', lambda date: self.pro.ths_hot(date=date), date=date, force_refresh=force_refresh)
        dc_hot = self._fetch_with_cache('dc_hot', lambda date: self.pro.dc_hot(date=date), date=date, force_refresh=force_refresh)
        return {'ths_hot': ths_hot, 'dc_hot': dc_hot}

    def get_announcements(self, ts_code, start_date, end_date, force_refresh=False):
        return self._fetch_with_cache('anns_d', lambda ts_code, start_date, end_date: self.pro.anns_d(ts_code=ts_code, start_date=start_date, end_date=end_date), ts_code=ts_code, start_date=start_date, end_date=end_date, force_refresh=force_refresh)

    def get_sector_concept(self, ts_code, force_refresh=False):
        ths_concept = self._fetch_with_cache('ths_concept', lambda ts_code: self.pro.ths_member(ts_code=ts_code), ts_code=ts_code, force_refresh=force_refresh)
        ths_industry = self._fetch_with_cache('ths_industry', lambda ts_code: self.pro.ths_index(ts_code=ts_code), ts_code=ts_code, force_refresh=force_refresh)
        return {'ths_concept': ths_concept, 'ths_industry': ths_industry}

    def get_moneyflow(self, ts_code, start_date, end_date, force_refresh=False):
        hsgt = self._fetch_with_cache('moneyflow_hsgt', lambda ts_code, start_date, end_date: self.pro.moneyflow_hsgt(ts_code=ts_code, start_date=start_date, end_date=end_date), ts_code=ts_code, start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        margin = self._fetch_with_cache('margin_detail', lambda ts_code, start_date, end_date: self.pro.margin_detail(ts_code=ts_code, start_date=start_date, end_date=end_date), ts_code=ts_code, start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        return {'moneyflow_hsgt': hsgt, 'margin_detail': margin} 