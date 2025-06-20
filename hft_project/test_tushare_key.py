import tushare as ts

pro = ts.pro_api("2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211")
 
df = pro.index_weight(index_code='000300.SH', start_date='20240501', end_date='20240612')
print(df.head()) 