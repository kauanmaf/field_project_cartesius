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
    **kwargs
):
    # Default parameter values
    defaults = {
        "adx_period": 14,
        "atr_period": 14,
        "cci_period": 20,
        "bb_period": 20,
        "bb_num_std": 2,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "aroon_period": 25,
        "stc_window_slow": 50,
        "stc_window_fast": 23,
        "stc_cycle": 10,
        "stc_smooth1": 3,
        "stc_smooth2": 3,
        "kst_r1": 10, "kst_r2": 15, "kst_r3": 20, "kst_r4": 30,
        "kst_n1": 10, "kst_n2": 10, "kst_n3": 10, "kst_n4": 15,
        "kst_signal": 9,
        "vortex_window": 14,
        "trix_window": 15,
        "mass_window_fast": 9,
        "mass_window_slow": 25,
        "dpo_window": 20,
        "stoch_rsi_period": 14,
        "stoch_period": 14,
        "stoch_smooth1": 3,
        "stoch_smooth2": 3,
        "sto_period": 14,
        "sto_smooth_k": 3,
        "sto_smooth_d": 3,
        "rsi_window": 14,
        "awesome_window1": 5,
        "awesome_window2": 34,
    }

    # Update defaults with any parameters provided via kwargs
    params = {**defaults, **kwargs}

    # Initialize an empty DataFrame to store the results
    indicators_df = pd.DataFrame()

    # Calculate indicators only for provided parameters
    if "adx_period" in params:
        indicators_df["adx"] = ADX(data, adx_period=params["adx_period"])
    if "atr_period" in params:
        indicators_df["atr"] = average_true_range(data, atr_period=params["atr_period"])
    if "cci_period" in params:
        indicators_df["cci"] = commodity_channel_index(data, cci_period=params["cci_period"])
    if "bb_period" in params and "bb_num_std" in params:
        ema, upper_band, lower_band = bollinger_bands(
            data, bb_period=params["bb_period"], num_std=params["bb_num_std"]
        )
        indicators_df["ema"] = ema
        indicators_df["upper band"] = upper_band
        indicators_df["lower band"] = lower_band
    if {"macd_fast", "macd_slow", "macd_signal"}.issubset(params):
        macd_line, signal_line, macd_histogram = MACD(
            data, macd_fast=params["macd_fast"], macd_slow=params["macd_slow"], macd_signal=params["macd_signal"]
        )
        indicators_df["macd line"] = macd_line
        indicators_df["signal line"] = signal_line
        indicators_df["macd histogram"] = macd_histogram
    if "aroon_period" in params:
        aroon_up, aroon_down, aroon_oscillator = aroon_indicator(
            data, aroon_period=params["aroon_period"]
        )
        indicators_df["aroon up"] = aroon_up
        indicators_df["aroon down"] = aroon_down
        indicators_df["aroon oscillator"] = aroon_oscillator
    if {"stc_window_slow", "stc_window_fast", "stc_cycle", "stc_smooth1", "stc_smooth2"}.issubset(params):
        indicators_df["stc"] = schaff_trend_cycle(
            data,
            window_slow=params["stc_window_slow"],
            window_fast=params["stc_window_fast"],
            cycle=params["stc_cycle"],
            smooth1=params["stc_smooth1"],
            smooth2=params["stc_smooth2"],
        )
    if {"kst_r1", "kst_r2", "kst_r3", "kst_r4", "kst_n1", "kst_n2", "kst_n3", "kst_n4", "kst_signal"}.issubset(params):
        kst, kst_signal_line, kst_diff = kst_oscillator(
            data,
            r1=params["kst_r1"], r2=params["kst_r2"], r3=params["kst_r3"], r4=params["kst_r4"],
            n1=params["kst_n1"], n2=params["kst_n2"], n3=params["kst_n3"], n4=params["kst_n4"],
            signal=params["kst_signal"],
        )
        indicators_df["kst"] = kst
        indicators_df["kst signal"] = kst_signal_line
        indicators_df["kst diff"] = kst_diff
    if "vortex_window" in params:
        vi_pos, vi_neg = vortex(data, window=params["vortex_window"])
        indicators_df["positive vortex"] = vi_pos
        indicators_df["negative vortex"] = vi_neg
    if "trix_window" in params:
        indicators_df["trix"] = trix(data, window=params["trix_window"])
    if {"mass_window_fast", "mass_window_slow"}.issubset(params):
        indicators_df["mass"] = mass(
            data, window_fast=params["mass_window_fast"], window_slow=params["mass_window_slow"]
        )
    if "dpo_window" in params:
        indicators_df["dpo"] = detrended_price(data, window=params["dpo_window"])
    if {"stoch_rsi_period", "stoch_period", "stoch_smooth1", "stoch_smooth2"}.issubset(params):
        stoch_rsi_k, stoch_rsi_d = stochastic_rsi(
            data,
            rsi_period=params["stoch_rsi_period"],
            stoch_period=params["stoch_period"],
            smooth1=params["stoch_smooth1"],
            smooth2=params["stoch_smooth2"],
        )
        indicators_df["srsi-k"] = stoch_rsi_k
        indicators_df["srsi-d"] = stoch_rsi_d
    if {"sto_period", "sto_smooth_k", "sto_smooth_d"}.issubset(params):
        sto_osc, sto_sig = stochastic_oscillator(
            data,
            stoch_period=params["sto_period"],
            smooth_k=params["sto_smooth_k"],
            smooth_d=params["sto_smooth_d"],
        )
        indicators_df["stochastic oscillator"] = sto_osc
        indicators_df["stochastic oscillator signal"] = sto_sig
    if "rsi_window" in params:
        indicators_df["rsi"] = relative_strength(data, window=params["rsi_window"])
    if {"awesome_window1", "awesome_window2"}.issubset(params):
        indicators_df["awesome"] = awesome(
            data, window1=params["awesome_window1"], window2=params["awesome_window2"]
        )

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
        if remove["column"] is None or matriz_corr.shape[0] <= k_best:
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
new_indicators = decorrelate(data)
