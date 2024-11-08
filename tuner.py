import os
from indicadores import *
import models as mod
import optuna
import labeling as lb
from backtesting import Backtest, Strategy
import json

def agg_indicators_tune(
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


def objective(trial):
    # Sugerir valores para os hiperparâmetros dos indicadores
    adx_period = trial.suggest_int("adx_period", 10, 30)
    psar_acceleration = trial.suggest_float("psar_acceleration", 0.01, 0.2)
    psar_max_acceleration = trial.suggest_float("psar_max_acceleration", 0.1, 0.5)
    atr_period = trial.suggest_int("atr_period", 10, 30)
    cci_period = trial.suggest_int("cci_period", 10, 30)
    bb_period = trial.suggest_int("bb_period", 15, 30)
    bb_num_std = trial.suggest_float("bb_num_std", 1.5, 3.0)
    macd_fast = trial.suggest_int("macd_fast", 8, 15)
    macd_slow = trial.suggest_int("macd_slow", 20, 30)
    macd_signal = trial.suggest_int("macd_signal", 5, 10)
    aroon_period = trial.suggest_int("aroon_period", 10, 30)
    stc_window_slow = trial.suggest_int("stc_window_slow", 40, 60)
    stc_window_fast = trial.suggest_int("stc_window_fast", 15, 30)
    stc_cycle = trial.suggest_int("stc_cycle", 8, 15)
    stc_smooth1 = trial.suggest_int("stc_smooth1", 2, 5)
    stc_smooth2 = trial.suggest_int("stc_smooth2", 2, 5)
    ichimoku_tenkan = trial.suggest_int("ichimoku_tenkan", 7, 12)
    ichimoku_kijun = trial.suggest_int("ichimoku_kijun", 20, 30)
    ichimoku_senkou_span_b = trial.suggest_int("ichimoku_senkou_span_b", 45, 60)
    kst_r1 = trial.suggest_int("kst_r1", 5, 15)
    kst_r2 = trial.suggest_int("kst_r2", 10, 20)
    kst_r3 = trial.suggest_int("kst_r3", 15, 25)
    kst_r4 = trial.suggest_int("kst_r4", 25, 35)
    kst_n1 = trial.suggest_int("kst_n1", 5, 15)
    kst_n2 = trial.suggest_int("kst_n2", 5, 15)
    kst_n3 = trial.suggest_int("kst_n3", 5, 15)
    kst_n4 = trial.suggest_int("kst_n4", 10, 20)
    kst_signal = trial.suggest_int("kst_signal", 5, 10)
    vortex_window = trial.suggest_int("vortex_window", 10, 30)
    trix_window = trial.suggest_int("trix_window", 10, 30)
    mass_window_fast = trial.suggest_int("mass_window_fast", 5, 15)
    mass_window_slow = trial.suggest_int("mass_window_slow", 20, 30)
    dpo_window = trial.suggest_int("dpo_window", 15, 25)
    stoch_rsi_period = trial.suggest_int("stoch_rsi_period", 10, 20)
    stoch_period = trial.suggest_int("stoch_period", 10, 20)
    stoch_smooth1 = trial.suggest_int("stoch_smooth1", 2, 5)
    stoch_smooth2 = trial.suggest_int("stoch_smooth2", 2, 5)
    sto_period = trial.suggest_int("sto_period", 10, 20)
    sto_smooth_k = trial.suggest_int("sto_smooth_k", 2, 5)
    sto_smooth_d = trial.suggest_int("sto_smooth_d", 2, 5)
    rsi_window = trial.suggest_int("rsi_window", 10, 30)
    awesome_window1 = trial.suggest_int("awesome_window1", 2, 10)
    awesome_window2 = trial.suggest_int("awesome_window2", 20, 40)
    n_estimators = trial.suggest_int("n_estimators", 60, 140)

    # Calcula os indicadores com os valores sugeridos
    indicators = agg_indicators_tune(
        data,
        adx_period=adx_period,
        psar_acceleration=psar_acceleration,
        psar_max_acceleration=psar_max_acceleration,
        atr_period=atr_period,
        cci_period=cci_period,
        bb_period=bb_period,
        bb_num_std=bb_num_std,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        aroon_period=aroon_period,
        stc_window_slow=stc_window_slow,
        stc_window_fast=stc_window_fast,
        stc_cycle=stc_cycle,
        stc_smooth1=stc_smooth1,
        stc_smooth2=stc_smooth2,
        ichimoku_tenkan=ichimoku_tenkan,
        ichimoku_kijun=ichimoku_kijun,
        ichimoku_senkou_span_b=ichimoku_senkou_span_b,
        kst_r1=kst_r1,
        kst_r2=kst_r2,
        kst_r3=kst_r3,
        kst_r4=kst_r4,
        kst_n1=kst_n1,
        kst_n2=kst_n2,
        kst_n3=kst_n3,
        kst_n4=kst_n4,
        kst_signal=kst_signal,
        vortex_window=vortex_window,
        trix_window=trix_window,
        mass_window_fast=mass_window_fast,
        mass_window_slow=mass_window_slow,
        dpo_window=dpo_window,
        stoch_rsi_period=stoch_rsi_period,
        stoch_period=stoch_period,
        stoch_smooth1=stoch_smooth1,
        stoch_smooth2=stoch_smooth2,
        sto_period=sto_period,
        sto_smooth_k=sto_smooth_k,
        sto_smooth_d=sto_smooth_d,
        rsi_window=rsi_window,
        awesome_window1=awesome_window1,
        awesome_window2=awesome_window2
    )

    # Calculando os indicadores
    indicators = normalize_indicators(indicators)
    # Calculando o rótulo
    y = np.array(lb.labelData(data["Adj Close"], 0.1)).ravel()
    # Eliminando as linhas com NaN
    indicators["y"] = y
    indicators = indicators.dropna()

    indicators_train = indicators[indicators.index.year != 2023]
    indicators_backtest = indicators[indicators.index.year == 2023]

    # Convertendo para numpy arrays, caso ainda não estejam
    X = np.array(indicators_train)[:, :-1]
    y = np.array(indicators_train)[:, -1]
    # Treinando o modelo
    model = mod.random_forest(X, y, n_estimators = n_estimators)
    # Predizendo a política para aquele ano
    pred = model.predict(np.array(indicators_backtest)[:, :-1])

    olhc_backtest = data[data.index.year == 2023]
    # Criando uma série com a predição e o index do ano
    policy = pd.Series(pred, index = olhc_backtest.index)
    # Colocando a predição nesse dataframe
    olhc_backtest["Signal"] = 0
    olhc_backtest.loc[policy.index, "Signal"] = policy

    bt = Backtest(olhc_backtest, OurStrategy, cash=10000)
    stats = bt.run()
    score = stats["Equity Final [$]"]

    return score


# study = optuna.create_study(direction = "maximize")
# study.optimize(objective, n_trials = 5)

# # Escrevendo o arquivo
# stock_name = "Prio3"
# file_path = os.path.join("params", f"{stock_name}_best_params.json")
# with open(file_path, "w") as f:
#     json.dump(study.best_params, f)

# # carregando os melhores parametros do arquivo
# file_path = os.path.join("params", f"{stock_name}_best_params.json")
# with open(file_path, "r") as f:
#     best_params = json.load(f)

# ## Como utilizar eles?
# # resultado = backtesting_model(olhc, model, year = None, **best_params):

# print("Melhores parâmetros:", study.best_params)
# print("Melhor score:", study.best_value)