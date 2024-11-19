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
files = glob.glob("data/*.csv")

# Lista de resultados finais
final_results = []

# Range de correlação
len_indic = [13,20]  # Intervalo de tamanhos para k_best

for element in files:
    # Pega o ticker
    TICKER = element.split("\\")[-1][:-4]
    DATA = read_and_set_index(element)
    
    # Ano no qual será feito o backtest
    YEAR_BACKTEST = 2024
    # Ano a ser usado como validação da tunagem de hiperparâmetros
    YEAR_VAL = 2023
    # Variável para ativar ou desativar a tunagem
    TUNE = True
    # Variável para ativar ou desativar a binarização dos dados
    BINARIZED = True
    # Variável para testar várias quantidades de colunas
    TEST_COLUMS = True

    for n_colunas in len_indic:
        # Obtém as n_colunas melhores *features*
        lista_colunas = get_columns_name(DATA, k_best=n_colunas)
        # Se a opção de tunagem for ativada, otimiza os hiperparâmetros
        if TUNE:
            if not YEAR_VAL or not YEAR_BACKTEST or YEAR_VAL == YEAR_BACKTEST:
                print("Erro de chamada: os anos de validação e backtest devem estar determinados e devem ser diferentes.")
                sys.exit()

            # Executa a otimização com essas colunas
            run_optimization(DATA, BINARIZED, YEAR_VAL, TICKER, n_trials=5, lista_colunas=lista_colunas)

        # Caminho do arquivo com os hiperparâmetros
        if BINARIZED:
            hyperparams_path = os.path.join("hyperparams", f"{TICKER}_b_{str(YEAR_VAL)}_{n_colunas}.json")
        else:
            hyperparams_path = os.path.join("hyperparams", f"{TICKER}_{str(YEAR_VAL)}_{n_colunas}.json")

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

        vol = volatility_df[volatility_df['Stock'] == TICKER]['Volatility'].values[0]
        teste = {
            "Stock": TICKER,
            "Features": n_colunas,
            "Volatility": vol,
            "Percent Return": percent_return,
            "Win Rate": win_rate,
            "accuracy test": accuracy
        }
        print(teste)
        # Adiciona os resultados ao DataFrame final
        final_results.append(teste)
        print(f"{TICKER} processado com {n_colunas} indicadores")

# Converte a lista de resultados para um DataFrame
final_df = pd.DataFrame(final_results)

# Exibe os resultados finais
print(final_df)
final_df.to_csv("results.csv")
