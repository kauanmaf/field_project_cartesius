import numpy as numpy
import pandas as pd
from tradingUtils import *
import ta

def MACD(data, macd_fast=12, macd_slow=26, macd_signal=9):
   
    # Calculate the MACD, Signal Line, and MACD Histogram
    macd = ta.trend.MACD(
        close=data['Adj Close'],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal
    )
    
    # Create a DataFrame with the MACD line, Signal line, and MACD Histogram
    macd_df = pd.DataFrame({
        'MACD Line': macd.macd(),
        'Signal Line': macd.macd_signal(),
        'MACD Histogram': macd.macd_diff()
    }, index=data.index)
    
    return macd_df

def aroon_indicator(data, aroon_period=25):
    
    aroon = ta.trend.AroonIndicator(high=data['High'], low = data['Low'], window=aroon_period)
    
    return pd.DataFrame({
        'Aroon Up': aroon.aroon_up(),
        'Aroon Down': aroon.aroon_down(),
        'Aroon Oscillator': aroon.aroon_indicator()
    }, index=data.index)

def schaff_trend_cycle(data, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3):
    # Calcula o STC usando a classe STCIndicator
    stc = ta.trend.STCIndicator(
        close=data['Close'],
        window_slow=window_slow,
        window_fast=window_fast,
        cycle=cycle,
        smooth1=smooth1,
        smooth2=smooth2
    ).stc()
    
    # Retorna o STC como uma Série com o mesmo índice dos dados de entrada
    return pd.Series(stc, index=data.index, name='STC')

def ichimoku_cloud(data, tenkan=9, kijun=26, senkou_span_b=52):
    ichimoku = ta.trend.IchimokuIndicator(high=data['High'], low=data['Low'], window1=tenkan, window2=kijun, window3=senkou_span_b)
    
    return pd.DataFrame({
        'Tenkan-sen': ichimoku.ichimoku_conversion_line(),
        'Kijun-sen': ichimoku.ichimoku_base_line(),
        'Senkou Span A': ichimoku.ichimoku_a(),
        'Senkou Span B': ichimoku.ichimoku_b()
    }, index=data.index)

def kst_oscillator(data, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, signal=9):
    kst = ta.trend.KSTIndicator(close=data['Adj Close'], roc1=r1, roc2=r2, roc3=r3, roc4=r4, window1=n1, window2=n2, window3=n3, window4=n4, nsig=signal)
    
    return pd.DataFrame({
        'KST': kst.kst(),
        'KST Signal': kst.kst_sig(),
        'KST Diff': kst.kst_diff()
    }, index=data.index)

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

    # Calculate the Bollinger Bands
    data['EMA'] = ta.trend.ema_indicator(close=data['Adj Close'], window=bb_period)
    data['Upper Band'] = data['EMA'] + (ta.volatility.bollinger_hband_indicator(data['Adj Close'], window=bb_period, window_dev=num_std))
    data['Lower Band'] = data['EMA'] - (ta.volatility.bollinger_lband_indicator(data['Adj Close'], window=bb_period, window_dev=num_std))

    # Return a DataFrame with SMA, Upper Band, and Lower Band
    return data[['EMA', 'Upper Band', 'Lower Band']]


def agg_indicators(data):
    data["ADX"] = ADX(data)
    data["Parabolic Sar"] = parabolic_sar(data)
    data['OBV'] = on_balance_volume(data)
    data['ATR'] = average_true_range(data)
    data['CCI'] = commodity_channel_index(data)
    data[['EMA','Upper Band', 'Lower Band']] = bollinger_bands(data)
    data[['MACD Line', 'Signal Line', 'MACD Histogram']] = MACD(data)
    data[['Aroon Up', 'Aroon Down', 'Aroon Oscillator']] = aroon_indicator(data)
    data['STC'] = schaff_trend_cycle(data)
    data[['Tenkan-sen', 'Kijun-sen', 'Senkou Span A', 'Senkou Span B']] = ichimoku_cloud(data)
    data[['KST', 'KST Signal', 'KST Diff']] = kst_oscillator(data)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(data, bins=30, kde=True, figsize=(15, 20)):
    """
    Plots the distribution of each column in the DataFrame.

    Parameters:
    - data (pd.DataFrame): The DataFrame with the data to plot.
    - bins (int): Number of bins for the histogram.
    - kde (bool): If True, include KDE plot.
    - figsize (tuple): Size of the overall figure.

    Returns:
    - None: Displays the plots.
    """
    num_columns = len(data.columns)
    num_rows = (num_columns + 1) // 2  # Arrange plots in a grid with 2 columns
    
    fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, column in enumerate(data.columns):
        sns.histplot(data[column], bins=bins, kde=kde, ax=axes[i], color='skyblue')
        axes[i].set_title(f"Distribution of {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Frequency")
    
    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()

data = tsla_data.copy()
agg_indicators(data)
plot_distributions(data)
def adj_data(data):
    data = data.iloc[:, 6:]
    data.to_numpy()


