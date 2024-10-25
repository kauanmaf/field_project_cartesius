import numpy as numpy
import pandas as pd
from tradingUtils import *
import ta

def ADX(data, adx_period=14):
    adx = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Adj Close'], window=adx_period).adx()
    return pd.Series(adx, index=data.index)

def parabolic_sar(data, acceleration=0.02, max_acceleration=0.2):

    sar = ta.trend.PSARIndicator(high=data['High'], low=data['Low'], close=data['Adj Close'], step=acceleration, max_step=max_acceleration).psar()
    return sar

def on_balance_volume(data):
    close = data['Adj Close'].values
    volume = data['Volume'].values
    
    obv = [0]  # Initial OBV starts at 0
    
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + volume[i]) 
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - volume[i]) 
        else:
            obv.append(obv[-1]) 
    return pd.Series(obv, index=data.index)

def average_true_range(data, atr_period=14):
    
    atr = ta.volatility.AverageTrueRange(high = data['High'], low = data['Low'], close = data['Adj Close'], window=atr_period).average_true_range()
    print(atr)
    
    return pd.Series(atr, index=data.index)


def commodity_channel_index(data, cci_period=20):
    
    cci = ta.trend.CCIIndicator(high = data['High'], low = data['Low'], close = data['Adj Close'], window=cci_period).cci()
    
    return pd.Series(cci, index=data.index)

def bollinger_bands(data, bb_period=20, num_std=2):
    # Ensure 'Adj Close' column exists in the data
    if 'Adj Close' not in data.columns:
        raise ValueError("Data must contain 'Adj Close' column.")

    # Calculate the Bollinger Bands
    data['SMA'] = ta.trend.sma_indicator(close=data['Adj Close'], window=bb_period)
    data['Upper Band'] = data['SMA'] + (ta.volatility.bollinger_hband_indicator(data['Adj Close'], window=bb_period, window_dev=num_std))
    data['Lower Band'] = data['SMA'] - (ta.volatility.bollinger_lband_indicator(data['Adj Close'], window=bb_period, window_dev=num_std))

    # Return a DataFrame with SMA, Upper Band, and Lower Band
    return data[['SMA', 'Upper Band', 'Lower Band']]

# Create a copy of the data to work with
data = tsla_data.copy()
data["ADX"] = ADX(data)
data["Parabolic Sar"] = parabolic_sar(data)
# Add OBV
data['OBV'] = on_balance_volume(data)

# Add ATR
data['ATR'] = average_true_range(data)


# Add CCI
data['CCI'] = commodity_channel_index(data)

# Add Bollinger Bands (this adds multiple columns)
bollinger = bollinger_bands(data)
data['SMA'] = bollinger['SMA']
data['Upper Band'] = bollinger['Upper Band']
data['Lower Band'] = bollinger['Lower Band']

print(data['ADX'])
print(data['Parabolic Sar'])
print(data)
