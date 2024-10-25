import numpy as numpy
import pandas as pd
from tradingUtils import *
import ta
from read_data import *

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


def agg_indicators2(data):
    vi_pos, vi_neg = vortex(data)
    ti = trix(data)
    mi = mass(data)
    dpo = detrended_price(data)

    indicators_df = pd.DataFrame({"Positive Vortex": vi_pos,
                                  "Negative Vortex": vi_neg,
                                  "Trix": ti,
                                  "Mass": mi,
                                  "DPO": dpo})
    
    return indicators_df