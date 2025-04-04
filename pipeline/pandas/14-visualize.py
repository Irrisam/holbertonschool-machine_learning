#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=['Weighted_Price'])

df = df.rename(columns={'Timestamp': 'Date'})

df['Date'] = pd.to_datetime(df['Date'], unit='s')

df = df.set_index('Date')


df['Close'] = df['Close'].fillna(method='ffill')


df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])


df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)


df_filtered = df[df.index.year >= 2017]

df_resampled = df_filtered.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})


plt.figure(figsize=(10,6))
df_resampled['Close'].plot(label='Close', color='blue')
plt.title('Close Price from 2017 onwards')
plt.xlabel('Date')
plt.ylabel('Close Price (Currency)')
plt.legend()
plt.grid(True)
plt.show()
