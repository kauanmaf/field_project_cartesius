from backtesting_process import *
from tuner import *
import sys
import os
import json

# List of datasets
DATA_LIST = [prio_data, tsla_data, viva_data, azul_data]
DATA_NAMES = ["prio_data", "tsla_data", "viva_data", "azul_data"]  # Names for each dataset in the list

# Year settings
YEAR_BACKTEST = 2024
YEAR_VAL = 2023
TUNE = False  # Variable to activate or deactivate tuning

for data, data_name in zip(DATA_LIST, DATA_NAMES):
    print(f"\nProcessing dataset: {data_name}")

    # Check for tuning
    if TUNE:
        if not YEAR_VAL or not YEAR_BACKTEST or YEAR_VAL == YEAR_BACKTEST:
            print("Erro de chamada: os anos de validação e backtest devem estar determinados e devem ser diferentes.")
            sys.exit()
        run_optimization(data, YEAR_VAL, n_trials=10)

    # Hyperparameters file path
    hyperparams_path = os.path.join("hyperparams", f"{data_name}_{YEAR_VAL}.json")

    # Load hyperparameters if the file exists
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, "r") as f:
            best_params = json.load(f)
        policy = backtesting_model(data, YEAR_BACKTEST, **best_params)
    else:
        policy = backtesting_model(data, YEAR_BACKTEST)  # Use default arguments if no hyperparameters file

    # Run backtest
    bt = Backtest(policy, OurStrategy, cash=10000)
    stats = bt.run()

    # Display results for the current dataset
    print(f"\nResults for {data_name}:")
    print(stats)
    # Chamando `bt.plot()` dentro da figura criada
    bt.plot(title_data=data_name)  # Gera o gráfico


