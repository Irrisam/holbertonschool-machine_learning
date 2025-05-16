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

for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])

df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

df = df['2017':]

daily_df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

fig, ax1 = plt.subplots(figsize=(15, 8))

ax1.plot(daily_df.index, daily_df['High'],
         label='High', color='green', alpha=0.7)
ax1.plot(daily_df.index, daily_df['Low'], label='Low', color='red', alpha=0.7)
ax1.plot(daily_df.index, daily_df['Open'],
         label='Open', color='blue', alpha=0.7)
ax1.plot(daily_df.index, daily_df['Close'],
         label='Close', color='orange', alpha=0.7)
ax1.set_ylabel('Price (USD)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.bar(daily_df.index, daily_df['Volume_(BTC)'],
        label='Volume (BTC)', color='purple', alpha=0.3, width=1)
ax2.set_ylabel('Volume (BTC)', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

plt.title('Bitcoin Price and Volume (2017-2019)', fontsize=14)
ax1.set_xlabel('Date')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()
