import numpy as numpy
import pandas as pd
from tradingUtils import *
import ta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


## Trend Indicators
def MACD(data, macd_fast=12, macd_slow=26, macd_signal=9):
   
    # Calculate the MACD, Signal Line, and MACD Histogram
    macd = ta.trend.MACD(
        close=data['Adj Close'],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal
    )
    
    return macd.macd(), macd.macd_signal(), macd.macd_diff()

def aroon_indicator(data, aroon_period=25):
    
    aroon = ta.trend.AroonIndicator(high=data['High'], low = data['Low'], window=aroon_period)
    
    return aroon.aroon_up(), aroon.aroon_down(), aroon.aroon_indicator()

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
    return stc

def ichimoku_cloud(data, tenkan=9, kijun=26, senkou_span_b=52):
    ichimoku = ta.trend.IchimokuIndicator(high=data['High'], low=data['Low'], window1=tenkan, window2=kijun, window3=senkou_span_b)
    
    return ichimoku.ichimoku_conversion_line(), ichimoku.ichimoku_base_line(), ichimoku.ichimoku_a(), ichimoku.ichimoku_b()

def kst_oscillator(data, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, signal=9):
    kst = ta.trend.KSTIndicator(close=data['Adj Close'], roc1=r1, roc2=r2, roc3=r3, roc4=r4, window1=n1, window2=n2, window3=n3, window4=n4, nsig=signal)
    
    return kst.kst(), kst.kst_sig(), kst.kst_diff()

def ADX(data, adx_period=14):
    adx = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Adj Close'], window=adx_period).adx()
    return adx

def parabolic_sar(data, acceleration=0.02, max_acceleration=0.2):

    sar = ta.trend.PSARIndicator(high=data['High'], low=data['Low'], close=data['Adj Close'], step=acceleration, max_step=max_acceleration).psar()
    return sar

def average_true_range(data, atr_period=14):
    
    atr = ta.volatility.AverageTrueRange(high = data['High'], low = data['Low'], close = data['Adj Close'], window=atr_period).average_true_range()
    
    return pd.Series(atr, index=data.index)

def commodity_channel_index(data, cci_period=20):
    
    cci = ta.trend.CCIIndicator(high = data['High'], low = data['Low'], close = data['Adj Close'], window=cci_period).cci()
    
    return pd.Series(cci, index=data.index)

## Volume

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
## Oscilators

def average_true_range(data, atr_period=14):
    
    atr = ta.volatility.AverageTrueRange(high = data['High'], low = data['Low'], close = data['Adj Close'], window=atr_period).average_true_range()
    
    return atr

def commodity_channel_index(data, cci_period=20):
    
    cci = ta.trend.CCIIndicator(high = data['High'], low = data['Low'], close = data['Adj Close'], window=cci_period).cci()
    
    return cci

def bollinger_bands(data, bb_period=20, num_std=2):

    # Calculate the Bollinger Bands
    ema = ta.trend.ema_indicator(close=data['Adj Close'], window=bb_period)
    upper_band = ema + (ta.volatility.bollinger_hband_indicator(data['Adj Close'], window=bb_period, window_dev=num_std))
    lower_band = ema - (ta.volatility.bollinger_lband_indicator(data['Adj Close'], window=bb_period, window_dev=num_std))

    # Return a DataFrame with SMA, Upper Band, and Lower Band
    return ema, upper_band, lower_band

def vortex(data, window = 14):
    vi_pos = ta.trend.VortexIndicator(data["High"], data["Low"], data["Adj Close"], window).vortex_indicator_pos()
    vi_neg = ta.trend.VortexIndicator(data["High"], data["Low"], data["Adj Close"], window).vortex_indicator_neg()
    return pd.Series(vi_pos, index = data.index), pd.Series(vi_neg, index = data.index)

def trix(data, window = 15):
    ti = ta.trend.trix(data["Adj Close"], window)
    return ti

def mass(data, window_fast = 9, window_slow = 25):
    mi = ta.trend.MassIndex(data["High"], data["Low"], window_fast, window_slow).mass_index()
    return mi

def detrended_price(data, window = 20):
    dpo = ta.trend.DPOIndicator(data["Adj Close"], window).dpo()
    return dpo

## Processing

def agg_indicators(data):
    adx = ADX(data)
    psar = parabolic_sar(data)
    obv = on_balance_volume(data)
    atr = average_true_range(data)
    cci = commodity_channel_index(data)
    ema, upper_band, lower_band = bollinger_bands(data)
    macd_line, signal_line, macd_histogram = MACD(data)
    aroon_up, aroon_down, aroon_oscilator = aroon_indicator(data)
    stc = schaff_trend_cycle(data)
    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b = ichimoku_cloud(data)
    kst, kst_signal, kst_diff = kst_oscillator(data)
    vi_pos, vi_neg = vortex(data)
    ti = trix(data)
    mi = mass(data)
    dpo = detrended_price(data)

    indicators_df = pd.DataFrame({"ADX": adx,
                                  "Parabolic SAR": psar,
                                  "OBV": obv,
                                  "ATR": atr,
                                  "CCI": cci,
                                  "EMA": ema,
                                  "Upper Band": upper_band,
                                  "Lower Band": lower_band,
                                  "MACD Line": macd_line,
                                  "Signal Line": signal_line,
                                  "MACD Histogram": macd_histogram,
                                  "Aroon Up": aroon_up,
                                  "Aroon Down": aroon_down,
                                  "Aroon Oscillator": aroon_oscilator,
                                  "STC": stc,
                                  "Tenkan-sen": tenkan_sen,
                                  "Kijun-sen": kijun_sen,
                                  "Senkou Span A": senkou_span_a,
                                  "Senkou Span B": senkou_span_b,
                                  "KST": kst,
                                  "KST Signal": kst_signal,
                                  "KST Diff": kst_diff,
                                  "Positive Vortex": vi_pos,
                                  "Negative Vortex": vi_neg,
                                  "Trix": ti,
                                  "Mass": mi,
                                  "DPO": dpo})
    
    return indicators_df

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
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 100))
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

def normalize_indicators(data):
    # Cria uma cópia para evitar mudanças no DataFrame original
    data_normalized = data.copy().astype(float)
    
    # Inicializa o MinMaxScaler
    scaler = StandardScaler()
    
    # Ajusta o scaler e transforma os dados
    data_normalized.iloc[:, :] = scaler.fit_transform(data)
    
    return data_normalized

data = tsla_data.copy()
data = agg_indicators(data)
normalized_data = normalize_indicators(data)
print(normalized_data)
# agg_indicators(data)
plot_distributions(normalized_data)
def adj_data(data):
    data = data.iloc[:, 6:]
    data.to_numpy()


