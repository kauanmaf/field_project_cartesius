import numpy as np
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

def stochastic_rsi(data, rsi_period=14, stoch_period=14, smooth1=3, smooth2=3):

    rsi = ta.momentum.RSIIndicator(close=data['Adj Close'], window=rsi_period).rsi()
    stoch_rsi = (rsi - rsi.rolling(window=stoch_period).min()) / (rsi.rolling(window=stoch_period).max() - rsi.rolling(window=stoch_period).min())
    stoch_rsi_k = stoch_rsi.rolling(window=smooth1).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=smooth2).mean()

    return stoch_rsi_k, stoch_rsi_d

def stochastic_oscillator(data, stoch_period=14, smooth_k=3, smooth_d=3):
    stoch = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Adj Close'], window=stoch_period, smooth_window=smooth_k)
    return stoch.stoch(), stoch.stoch_signal()

def relative_strength(data, window = 14):
    rsi = ta.momentum.RSIIndicator(data["Adj Close"], window).rsi()
    return rsi

def awesome(data, window1 = 5, window2 = 34):
    ao = ta.momentum.AwesomeOscillatorIndicator(data["High"], data["Low"], window1, window2).awesome_oscillator()
    return ao

## Processing

def agg_indicators(
    data,
    adx_period=14,
    psar_acceleration=0.02,
    psar_max_acceleration=0.2,
    atr_period=14,
    cci_period=20,
    bb_period=20,
    bb_num_std=2,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    aroon_period=25,
    stc_window_slow=50,
    stc_window_fast=23,
    stc_cycle=10,
    stc_smooth1=3,
    stc_smooth2=3,
    ichimoku_tenkan=9,
    ichimoku_kijun=26,
    ichimoku_senkou_span_b=52,
    kst_r1=10, kst_r2=15, kst_r3=20, kst_r4=30,
    kst_n1=10, kst_n2=10, kst_n3=10, kst_n4=15,
    kst_signal=9,
    vortex_window=14,
    trix_window=15,
    mass_window_fast=9,
    mass_window_slow=25,
    dpo_window=20,
    stoch_rsi_period=14,
    stoch_period=14,
    stoch_smooth1=3,
    stoch_smooth2=3,
    sto_period=14,
    sto_smooth_k=3,
    sto_smooth_d=3,
    rsi_window=14,
    awesome_window1=5,
    awesome_window2=34
):
    adx = ADX(data, adx_period=adx_period)
    psar = parabolic_sar(data, acceleration=psar_acceleration, max_acceleration=psar_max_acceleration)
    obv = on_balance_volume(data)
    atr = average_true_range(data, atr_period=atr_period)
    cci = commodity_channel_index(data, cci_period=cci_period)
    ema, upper_band, lower_band = bollinger_bands(data, bb_period=bb_period, num_std=bb_num_std)
    macd_line, signal_line, macd_histogram = MACD(data, macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal)
    aroon_up, aroon_down, aroon_oscillator = aroon_indicator(data, aroon_period=aroon_period)
    stc = schaff_trend_cycle(data, window_slow=stc_window_slow, window_fast=stc_window_fast, cycle=stc_cycle, smooth1=stc_smooth1, smooth2=stc_smooth2)
    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b = ichimoku_cloud(data, tenkan=ichimoku_tenkan, kijun=ichimoku_kijun, senkou_span_b=ichimoku_senkou_span_b)
    kst, kst_signal_line, kst_diff = kst_oscillator(data, r1=kst_r1, r2=kst_r2, r3=kst_r3, r4=kst_r4, n1=kst_n1, n2=kst_n2, n3=kst_n3, n4=kst_n4, signal=kst_signal)
    vi_pos, vi_neg = vortex(data, window=vortex_window)
    ti = trix(data, window=trix_window)
    mi = mass(data, window_fast=mass_window_fast, window_slow=mass_window_slow)
    dpo = detrended_price(data, window=dpo_window)
    stoch_rsi_k, stoch_rsi_d = stochastic_rsi(data, rsi_period=stoch_rsi_period, stoch_period=stoch_period, smooth1=stoch_smooth1, smooth2=stoch_smooth2)
    sto_osc, sto_sig = stochastic_oscillator(data, stoch_period=sto_period, smooth_k=sto_smooth_k, smooth_d=sto_smooth_d)
    rsi = relative_strength(data, window=rsi_window)
    ao = awesome(data, window1=awesome_window1, window2=awesome_window2)

    indicators_df = pd.DataFrame({
        "ADX": adx,
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
        "Aroon Oscillator": aroon_oscillator,
        "STC": stc,
        "Tenkan-sen": tenkan_sen,
        "Kijun-sen": kijun_sen,
        "Senkou Span A": senkou_span_a,
        "Senkou Span B": senkou_span_b,
        "KST": kst,
        "KST Signal": kst_signal_line,
        "KST Diff": kst_diff,
        "Positive Vortex": vi_pos,
        "Negative Vortex": vi_neg,
        "Trix": ti,
        "Mass": mi,
        "DPO": dpo,
        "SRSI-k": stoch_rsi_k, 
        "SRSI-d": stoch_rsi_d,
        "Stochastic Oscillator": sto_osc, 
        "Stochastic Oscillator Signal": sto_sig,
        "RSI": rsi,
        "Awesome": ao
    })
    
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

def decorrelate(data_normalized, limite_correlacao = 0.8, show_graphs = False, k_best = 15):
    # Calcular a matriz de correlação
    correlation_matrix = data_normalized.corr()

    columns_to_drop = []
    # Calcule a matriz de correlação
    matriz_corr = correlation_matrix.abs()  # Pegue valores absolutos para facilitar a comparação
    matriz_corr_for = matriz_corr.copy()
    np.fill_diagonal(matriz_corr_for.values, 0)

    while True:
        remove = {"biggest_sum": 0, "column": None}
        for column in matriz_corr.columns:
            
            # Sum correlations for this column that are above the threshold
            correlation_sum = matriz_corr_for[column][matriz_corr_for[column] > limite_correlacao].sum()
            
            if correlation_sum > remove["biggest_sum"]:
                remove["biggest_sum"] = correlation_sum
                remove["column"] = column
        # If no columns exceed the threshold, break the loop
        if remove["column"] is None or matriz_corr.shape[0] < k_best:
            break
        # Add the column to the drop list and remove it from the correlation matrix
        columns_to_drop.append(remove["column"])
        matriz_corr.drop(columns=remove["column"], inplace=True)
        matriz_corr.drop(index=remove["column"], inplace=True)

    data_descor = data_normalized.drop(columns=columns_to_drop)

    corr_novo = data_descor.corr()
    # Plotar a matriz de correlação usando um heatmap
    if show_graphs:
        plt.figure(figsize=(10, 16))  # Aumenta a altura para acomodar os dois gráficos

        # Primeiro gráfico
        plt.subplot(2, 1, 1)  # 2 linhas, 1 coluna, 1ª posição
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 4})
        plt.title("Matriz de Correlação 1")

        # Segundo gráfico
        plt.subplot(2, 1, 2)  # 2 linhas, 1 coluna, 2ª posição
        sns.heatmap(corr_novo, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 4})
        plt.title("Matriz de Correlação 2")

        plt.tight_layout()  # Ajusta o espaçamento para evitar sobreposição
        plt.show()
    
    return data_descor
    

data = tsla_data.copy()

data = agg_indicators(data)
normalized_data = normalize_indicators(data)
new_indicators = decorrelate(data, show_graphs=False)
