import os
from indicadores import *
from backtesting_process import *
import models as mod
import optuna
import labeling as lb
from backtesting import Backtest, Strategy
import json

# Função para criar a função objetivo
def create_objective(ohlc, year):
    # Cria a função objetivo
    def objective(trial):
        # Sugerindo valores para os hiperparâmetros dos indicadores
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

        # Calculando a política para o dado e ano especificados com os parâmetros a serem testados
        ohlc_backtest = backtesting_model(ohlc, year, n_estimators = n_estimators,
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
                                          awesome_window2=awesome_window2)

        bt = Backtest(ohlc_backtest, OurStrategy, cash=10000)
        stats = bt.run()
        score = stats["Equity Final [$]"]

        return score
    
    return objective

# Função que otimiza os hiperparâmetros
def run_optimization(ohlc, year, n_trials = 100):
    # Criando o estudo e realizando a otimização
    study = optuna.create_study(direction = "maximize")
    objective = create_objective(ohlc, year)
    study.optimize(objective, n_trials = n_trials)

    # Pegando o nome do dado usado
    data_name = [name for name, value in globals().items() if value is ohlc]
    
    # Escrevendo os hiperparâmetros no JSON adequado
    file_path = os.path.join("hyperparams", f"{data_name[0]}_{year}.json")
    with open(file_path, "w") as f:
        json.dump(study.best_params, f)

# # carregando os melhores parametros do arquivo
# file_path = os.path.join("params", f"{stock_name}_best_params.json")
# with open(file_path, "r") as f:
#     best_params = json.load(f)

# ## Como utilizar eles?
# # resultado = backtesting_model(olhc, model, year = None, **best_params):

# print("Melhores parâmetros:", study.best_params)
# print("Melhor score:", study.best_value)