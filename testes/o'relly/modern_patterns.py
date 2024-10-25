import pandas as pd
import numpy as np
from tradingUtils import *

data = tsla_data

signal_policies = {}

def quintuplets(data, body):
    result = pd.Series(0, index = data.index)
    ups = 0
    downs = 0

    for date in data.index:
        variation = data.loc[date]["Adj Close"] - data.loc[date]["Open"]
        if variation > 0:
            downs = 0
            if variation < body:
                ups += 1
        elif variation < 0:
            ups = 0
            if variation > -body:
                downs += 1

        if ups >= 5:
            result[date] = 1
        elif downs >= 5:
            result[date] = -1

    return result

def atr(data, period):
    # Inicializa a coluna "True Range" com o tipo float
    data["True Range"] = 0.0
    previous_day = data.index[0]

    # Cálculo do True Range
    for date in data.index[1:]:
        current_data = data.loc[date]
        previous_data = data.loc[previous_day]
        tr = max(current_data["High"] - current_data["Low"],
                 current_data["High"] - previous_data["Adj Close"],
                 previous_data["Adj Close"] - current_data["Low"])
        data.at[date, "True Range"] = tr
        previous_day = date

    # Inicializa a coluna "Atr" com o tipo float
    data["Atr"] = 0.0

    # Calcula o ATR inicial
    data.at[data.index[period - 1], "Atr"] = np.mean(data["True Range"][:period])

    previous_day = data.index[period - 1]

    # Cálculo do ATR para os dias subsequentes
    for date in data.index[period:]:
        data.at[date, "Atr"] = (data.loc[previous_day]["Atr"] * (period - 1) + data.loc[date, "True Range"]) / period
        previous_day = date


def double_trouble(data, period):
    atr(data, period)

    result = pd.Series(0, index=data.index)
    previous_day = data.index[period - 1]

    for date in data.index[period:]:
        current_data = data.loc[date]
        previous_data = data.loc[previous_day]

        # Padrão altista (bullish)
        if (previous_data["Adj Close"] > previous_data["Open"] and
            current_data["Adj Close"] > current_data["Open"] and
            previous_data["Adj Close"] < current_data["Adj Close"] and
            current_data["High"] - current_data["Low"] >= 2 * previous_data["Atr"]):
            result.loc[date] = 1

        # Padrão baixista (bearish)
        elif (previous_data["Adj Close"] < previous_data["Open"] and
              current_data["Adj Close"] < current_data["Open"] and
              previous_data["Adj Close"] > current_data["Adj Close"] and
              current_data["High"] - current_data["Low"] >= 2 * previous_data["Atr"]):
            result.loc[date] = -1

        previous_day = date

    return result

# Executando as políticas
run_signal_policy(tsla_data, quintuplets, "quintuplets_pattern")
run_signal_policy(tsla_data, double_trouble, "double_trouble_pattern")

modern_ml_data = tsla_data.iloc[:, 6:]