import pandas as pd
from tradingUtils import *

data = tsla_data

signal_policies = {}

def MACD_pattern_policy(data, short_period = 12, long_period = 26, signal_period = 9):
    # Começando nossa séries de sinal
    result = pd.Series(0, index=data.index)

    # média móvel exponencial
    short_ema = data['Adj Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Adj Close'].ewm(span=long_period, adjust=False).mean()
    
    # linha do macd
    macd_line = short_ema - long_ema
    
    # linha de sinal
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # diferença do macd com o sinal
    diff = macd_line - signal_line

    # Generating signals based on the MACD
    result[diff > 0] = 1 
    result[diff < 0] = -1 

    return result

def bollinger_bands_policy(data, period = 20, num_std = 2):
    # Initializing the result Series
    result = pd.Series(0, index=data.index)

    # Calculating the moving average and standard deviation
    mid_band = data['Adj Close'].rolling(window=period).mean()
    rolling_std = data['Adj Close'].rolling(window=period).std()

    # Calculating the upper and lower bands
    upper_band = mid_band + (num_std * rolling_std)
    lower_band = mid_band - (num_std * rolling_std)

    # Generating signals based on Bollinger Bands
    result[data['Adj Close'] < lower_band] = 1  # Buy signal
    result[data['Adj Close'] > upper_band] = -1  # Sell signal

    return result

def MA_crossover_policy(data):
    # Initializing the result Series
    result = pd.Series(0, index=data.index)

    # MA Crossover parameters
    short_period = 10
    long_period = 50

    # Calculating the short and long moving averages
    short_average = data['Adj Close'].rolling(window=short_period).mean()
    long_average = data['Adj Close'].rolling(window=long_period).mean()

    # Generating signals based on the crossover
    result[(short_average > long_average) & (short_average.shift(1) <= long_average.shift(1))] = 1

    result[(short_average < long_average) & (short_average.shift(1) >= long_average.shift(1))] = -1

    return result

def marubozu_pattern_policy(data):
    result = pd.Series(0, index=data.index)

    close = data['Adj Close'].values
    open_ = data['Open'].values
    high = data['High'].values
    low = data['Low'].values

    for i in range(len(data)):
        
        # Padrão bullish (Marubozu de alta)
        if (
            close[i] > open_[i] and 
            high[i] == close[i] and 
            low[i] == open_[i]
        ):
            result.iloc[i] = 1  # Buy signal

        # Padrão bearish (Marubozu de baixa)
        elif (
            close[i] < open_[i] and 
            high[i] == open_[i] and 
            low[i] == close[i]
        ):
            result.iloc[i] = -1  # Sell signal

    return result

# Política de três candles como mesmo sinal
def three_candles_policy(data, body):
    result = pd.Series(0, index=data.index)

    # Começamos da terceira linha
    for i in range(2, len(data)):
        close = data['Adj Close'].values
        open_ = data['Open'].values

        # Checamos as condições de bullish
        if (
            (close[i] - open_[i] > body) and
            (close[i - 1] - open_[i - 1] > body) and
            (close[i - 2] - open_[i - 2] > body) and
            close[i] > close[i - 1] and
            close[i - 1] > close[i - 2]
        ):
            result.iloc[i] = 1  # Buy signal
        
        # Checamos as condições de bearish
        elif (
            (close[i] - open_[i] > body) and
            (close[i - 1] - open_[i - 1] > body) and
            (close[i - 2] - open_[i - 2] > body) and
            close[i] < close[i - 1] and
            close[i - 1] < close[i - 2]
        ):
            result.iloc[i] = -1  # Sell signal
    
    return result

def tasuki_pattern_policy(data):
    result = pd.Series(0, index=data.index)

    # Começamos da terceira linah
    for i in range(2, len(data)):
        # Pegamos a coluna close e open  
        close = data['Adj Close'].values
        open_ = data['Open'].values

        # tasuki de alta
        if (
            (close[i - 2] > open_[i - 2]) and 
            (close[i - 1] > open_[i - 1]) and 
            (open_[i - 1] > close[i - 2]) and  # Checa o gap
            (open_[i] < close[i - 1]) and  # Abre no corpo do segudno
            (close[i] < open_[i]) and  # é de baixa
            (close[i] > close[i - 2])  
        ):
            result.iloc[i] = 1  

        # Tasuki de baixa
        elif (
            (close[i - 2] < open_[i - 2]) and  
            (close[i - 1] < open_[i - 1]) and  
            (open_[i - 1] < close[i - 2]) and 
            (open_[i] > close[i - 1]) and  
            (close[i] > open_[i]) and  
            (close[i] < close[i - 2]) 
        ):
            result.iloc[i] = -1  

    return result

def hikkake_pattern_policy(data):
    result = pd.Series(0, index=data.index)

    close = data['Adj Close'].values
    open_ = data['Open'].values
    high = data['High'].values
    low = data['Low'].values

    for i in range(4, len(data)):  # Start from the 5th data point
        # Bullish Hikkake pattern
        if (
            close[i] > high[i - 3] and
            close[i] > close[i - 4] and
            low[i - 1] < open_[i] and
            close[i - 1] < close[i] and
            high[i - 1] <= high[i - 3] and
            low[i - 2] < open_[i] and
            close[i - 2] < close[i] and
            high[i - 2] <= high[i - 3] and
            high[i - 3] < high[i - 4] and
            low[i - 3] > low[i - 4] and
            close[i - 4] > open_[i - 4]
        ):
            result.iloc[i] = 1  # Buy signal for the next day

        # Bearish Hikkake pattern
        elif (
            close[i] < low[i - 3] and
            close[i] < close[i - 4] and
            high[i - 1] > open_[i] and
            close[i - 1] > close[i] and
            low[i - 1] >= low[i - 3] and
            high[i - 2] > open_[i] and
            close[i - 2] > close[i] and
            low[i - 2] >= low[i - 3] and
            low[i - 3] > low[i - 4] and
            high[i - 3] < high[i - 4] and
            close[i - 4] < open_[i - 4]
        ):
            result.iloc[i] = -1  # Sell signal for the next day

    return result

# Executando as políticas
run_signal_policy(data, MA_crossover_policy, "ma_crossover")
run_signal_policy(data, MACD_pattern_policy, "macd_pattern")
run_signal_policy(data, bollinger_bands_policy, "bollinger_bands")
run_signal_policy(data, marubozu_pattern_policy, "marubozu_pattern")
run_signal_policy(data, three_candles_policy, "three_candles_policy", body=0.1)
run_signal_policy(data, tasuki_pattern_policy, "tasuki_pattern")
run_signal_policy(data, hikkake_pattern_policy, "hikkake_pattern")

classic_ml_data = data.iloc[:, 6:]

