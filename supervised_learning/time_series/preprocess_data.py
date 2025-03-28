#!/usr/bin/env python3
"""
    module to preprocess crypto data
"""

import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot
import os

def plot_time_series(data):
    first_output = go.Scatter(
        x=data.index,
        y=data['Open'].astype(float),
        mode='lines',
        name='Open'
    )
    second_output = go.Scatter(
        x=data.index,
        y=data['High'].astype(float),
        mode='lines',
        name='High'
    )
    third_output = go.Scatter(
        x=data.index,
        y=data['Low'].astype(float),
        mode='lines',
        name='Low'
    )
    fourth_output = go.Scatter(
        x=data.index,
        y=data['Close'].astype(float),
        mode='lines',
        name='Close'
    )
    
    layout = dict(
        title='Historical Bitcoin Price',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=12, label='1y', step='month', stepmode='backward'),
                    dict(count=36, label='3y', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        ))
    
    dataplot = [first_output, second_output, third_output, fourth_output]
    data_shape = dict(data=dataplot, layout=layout)
    
    iplot(data_shape)

def preprocess_data(path_file1, path_file2):
    if not os.path.isfile(path_file1):
        raise FileNotFoundError(f"File {path_file1} doesn't exist.")
    if not os.path.isfile(path_file2):
        raise FileNotFoundError(f"File {path_file2} doesn't exist.")
    
    print(f"Load data from {path_file1} and {path_file2}")
    origin_dataframe = pd.read_csv(path_file1)
    worked_dataframe= pd.read_csv(path_file2)
    
    origin_dataframe = origin_dataframe.set_index(pd.to_datetime(origin_dataframe['Timestamp'], unit='s'))
    origin_dataframe = origin_dataframe.drop('Timestamp', axis=1)
    worked_dataframe= worked_dataframe.set_index(pd.to_datetime(df2['Timestamp'], unit='s'))
    worked_dataframe= worked_dataframe.drop('Timestamp', axis=1)
    
    del_col = ['Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
    origin_dataframe_clean = origin_dataframe.drop(columns=del_col)
    df2_clean = worked_dataframe.drop(columns=del_col)
    
    combined_df = origin_dataframe_clean.combine_first(df2_clean)
    resulting_dataframe = combined_df[combined_df.index >= pd.Timestamp(2017, 1, 1)]
    
    resulting_dataframe['Open'] = resulting_dataframe['Open'].fillna(method='ffill')
    resulting_dataframe['High'] = resulting_dataframe['High'].fillna(method='ffill')
    resulting_dataframe['Low'] = resulting_dataframe['Low'].fillna(method='ffill')
    resulting_dataframe['Close'] = resulting_dataframe['Close'].fillna(method='ffill')
    
    resulting_dataframe.to_csv('preprocess_data.csv', index=False)
    
    plot_time_series(resulting_dataframe)
    
    return resulting_dataframe

if __name__ == "__main__":
    preprocessed_data = preprocess_data("bitstamp.csv", "coinbase.csv")
