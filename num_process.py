import pandas as pd
import json
import os
from backtesting_process import *
from tuner import *
import sys
import glob

# Função principal
def main_process(files: list, len_indic: np.array, year_backtest: int, year_val: int, tune: bool, binarized: bool, test_columns: bool, volatility_df: pd.DataFrame):
    # Lista de resultados finais
    final_results = []

    for element in files:
        json_ticker = {}
        # Pega o ticker
        TICKER = element.split("\\")[-1][:-4]

        DATA = read_and_set_index(element)

        if not year_val or not year_backtest or year_val == year_backtest:
            print("Erro de chamada: os anos de validação e backtest devem ser diferentes e determinados.")
            sys.exit()

        for n_colunas in len_indic:
            best_params = None
            # Se a opção de tunagem for ativada, otimiza os hiperparâmetros
            if tune:
                study_params, study_value = run_optimization(DATA, binarized, year_val, TICKER, n_trials=100, k_best=n_colunas)
                json_ticker[n_colunas] = {"params": study_params, "value": study_value}
                best_params = study_params
            else:
                # Caminho do arquivo com os hiperparâmetros
                hyperparams_path = os.path.join(
                    "hyperparams",
                    f"{TICKER}_{'b_' if binarized else ''}{year_val}.json"
                )

                # Se o arquivo de hiperparâmetros existir, carrega os parâmetros
                if os.path.exists(hyperparams_path):
                    with open(hyperparams_path, "r") as f:
                        dict_best_params = json.load(f)
                    if n_colunas in dict_best_params.keys():
                        best_params = dict_best_params[n_colunas]["params"]

            if best_params is not None:
                policy, accuracy, dict_total, _ = backtesting_model(DATA, year_backtest, **best_params)
            else:
                try:
                    policy, accuracy, dict_total, _ = backtesting_model(DATA, year_backtest)
                except Exception as e:
                    print(f"Erro ao executar o backtesting para {TICKER} com {n_colunas} indicadores: {e}")
                    continue

            # Calculando o backtest
            bt = Backtest(policy, OurStrategy, cash=10000)
            stats = bt.run()

            # Obtém o retorno percentual e a taxa de vitória
            percent_return = stats["Return [%]"]
            win_rate = stats["Win Rate [%]"]

            # Obtém a volatilidade da ação
            try:
                vol = volatility_df[volatility_df['Stock'] == TICKER]['Volatility'].values[0]
            except KeyError:
                vol = None  # Caso não exista informação de volatilidade

            teste = {
                "Stock": TICKER,
                "Features": n_colunas,
                "Volatility": vol,
                "Percent Return": percent_return,
                "Win Rate": win_rate,
                "Accuracy Test": accuracy,
                "Total Dict": dict_total,
                "Selected Features": list(best_params.keys()) if 'best_params' in locals() else []
            }
            # Adiciona os resultados ao DataFrame final
            final_results.append(teste)
            print(f"{TICKER} processado com {n_colunas} indicadores")

        # Ordenando os elementos do dicionário com base no valor obtido por aquela combinação
        json_ticker = dict(sorted(json_ticker.items(), key=lambda x: x[1]["value"], reverse=True))

        if not test_columns:
            break  # Sai do loop de LEN_INDIC caso TEST_COLUMNS esteja desativado
    
        file_path =  os.path.join("hyperparams", f"{TICKER}_{'b_' if binarized else ''}{year_val}.json")
        with open(file_path, "w") as f:
            json.dump(json_ticker, f)

    # Converte a lista de resultados para um DataFrame
    final_df = pd.DataFrame(final_results)

    # Exibe e salva os resultados finais
    print(final_df)
    final_df.to_csv("results.csv", index=False)
