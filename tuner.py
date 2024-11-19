import os
from indicadores import *
from backtesting_process import *
import optuna
import labeling as lb
from backtesting import Backtest, Strategy
import json

# Função para criar a função objetivo
def create_objective(ohlc, binarized, year, lista_colunas):
    # Cria a função objetivo    
    def objective(trial):
        # Dictionary mapping indicators to their associated hyperparameters
        hyperparams_mapping = {
            "adx": {"adx_period": (10, 30)},
            "atr": {"atr_period": (10, 30)},
            "cci": {"cci_period": (10, 30)},
            "bollinger": {"bb_period": (15, 30), "bb_num_std": (1.5, 3.0)},
            "macd": {"macd_fast": (8, 15), "macd_slow": (20, 30), "macd_signal": (5, 10)},
            "aroon": {"aroon_period": (10, 30)},
            "stc": {
                "stc_window_slow": (40, 60),
                "stc_window_fast": (15, 30),
                "stc_cycle": (8, 15),
                "stc_smooth1": (2, 5),
                "stc_smooth2": (2, 5),
            },
            "kst": {
                "kst_r1": (5, 15),
                "kst_r2": (10, 20),
                "kst_r3": (15, 25),
                "kst_r4": (25, 35),
                "kst_n1": (5, 15),
                "kst_n2": (5, 15),
                "kst_n3": (5, 15),
                "kst_n4": (10, 20),
                "kst_signal": (5, 10),
            },
            "vortex": {"vortex_window": (10, 30)},
            "trix": {"trix_window": (10, 30)},
            "mass": {"mass_window_fast": (5, 15), "mass_window_slow": (20, 30)},
            "dpo": {"dpo_window": (15, 25)},
            "stoch_rsi": {"stoch_rsi_period": (10, 20)},
            "stoch": {
                "stoch_period": (10, 20),
                "stoch_smooth1": (2, 5),
                "stoch_smooth2": (2, 5),
            },
            "sto": {
                "sto_period": (10, 20),
                "sto_smooth_k": (2, 5),
                "sto_smooth_d": (2, 5),
            },
            "rsi": {"rsi_window": (10, 30)},
            "awesome": {"awesome_window1": (2, 10), "awesome_window2": (20, 40)},
        }

        # Default parameters to always include
        params = {}
        params["n_estimators"] = trial.suggest_int("n_estimators", 60, 140)

        # Loop through hyperparameters and filter by lista_colunas
        for indicador, hyperparams in hyperparams_mapping.items():
            if indicador in lista_colunas:
                for param, bounds in hyperparams.items():
                    if isinstance(bounds[0], int):
                        params[param] = trial.suggest_int(param, bounds[0], bounds[1])
                    elif isinstance(bounds[0], float):  # Float range
                        params[param] = trial.suggest_float(param, bounds[0], bounds[1])
  
        # Calculating the policy dynamically
        ohlc_backtest = backtesting_model(
            ohlc,
            binarized,
            year,
            **params,
        )

        bt = Backtest(ohlc_backtest[0], OurStrategy, cash=10000)
        stats = bt.run()
        score = stats["Equity Final [$]"]

        return score
    
    return objective

# Função que otimiza os hiperparâmetros
def run_optimization(ohlc, binarized, year, ticker, lista_colunas: list, n_trials = 100):
    # Criando o estudo e realizando a otimização
    study = optuna.create_study(direction = "maximize")
    objective = create_objective(ohlc, binarized, year, lista_colunas)
    study.optimize(objective, n_trials = n_trials)
    
    # Escrevendo os hiperparâmetros no JSON adequado
    if binarized:
        file_path = os.path.join("hyperparams", f"{ticker}_b_{year}_{len(lista_colunas)}.json")
    else:
        file_path = os.path.join("hyperparams",f"{ticker}_{year}_{len(lista_colunas)}.json")

    with open(file_path, "w") as f:
        json.dump(study.best_params, f)