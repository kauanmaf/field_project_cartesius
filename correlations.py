import pandas as pd
import json
import os
from backtesting_process import *
from tuner import *
import sys
import glob

# Leitura dos arquivos de volatilidade
with open("data/new_stocks/volatility.json", "r") as f:
    volatility = json.load(f)

# Converte os dados de volatilidade para um DataFrame
volatility_df = pd.DataFrame(volatility)

# Lista de arquivos CSV
files = glob.glob("data/new_stocks/*.csv")

# Lista para armazenar os resultados finais
final_results = []

for element in files:
    # Pega o ticker
    ticker = element.split("\\")[-1][:-4]
    DATA = read_and_set_index(element)
    
    # Ano no qual será feito o backtest
    YEAR_BACKTEST = 2024
    # Ano a ser usado como validação da tunagem de hiperparâmetros
    YEAR_VAL = 2023
    # Variável para ativar ou desativar a tunagem
    TUNE = False

    # Se a opção de tunagem for ativada, otimiza os hiperparâmetros
    if TUNE:
        if not YEAR_VAL or not YEAR_BACKTEST or YEAR_VAL == YEAR_BACKTEST:
            print("Erro de chamada: os anos de validação e backtest devem estar determinados e devem ser diferentes.")
            sys.exit()
        run_optimization(DATA, YEAR_VAL, n_trials=50)

    # Pegando o caminho do arquivo com os hiperparâmetros
    data_name = [name for name, value in globals().items() if value is DATA]
    hyperparams_path = os.path.join("hyperparams", f"{data_name[0]}_{str(YEAR_VAL)}.json")

    # Se o arquivo de hiperparâmetros existir...
    if os.path.exists(hyperparams_path):
        # Pega os hiperparâmetros
        with open(hyperparams_path, "r") as f:
            best_params = json.load(f)

        # Calcula a política com esses parâmetros
        policy, accuracy = backtesting_model(DATA, YEAR_BACKTEST, **best_params)

    # Se não, utiliza os argumentos padrão
    else:
        try:
            policy, accuracy = backtesting_model(DATA, YEAR_BACKTEST)
        except:
            pass

    # Calculando o backtest
    bt = Backtest(policy, OurStrategy, cash=10000)
    stats = bt.run()

    # Obtém o retorno percentual e a taxa de vitória
    percent_return = stats["Return [%]"]
    win_rate = stats["Win Rate [%]"]

    vol = volatility_df[volatility_df['Stock'] == ticker]['Volatility'].values[0]

    # Adiciona os resultados ao DataFrame final
    final_results.append({
        "Stock": ticker,
        "Volatility": vol,
        "Percent Return": percent_return,
        "Win Rate": win_rate,
        "accuracy test": accuracy
    })
    print(f"{ticker} processado")

# Converte a lista de resultados para um DataFrame
final_df = pd.DataFrame(final_results)

# Exibe os resultados finais
print(final_df)
final_df.to_csv("results.csv")
