import pandas as pd
import json
import os
from backtesting_process import *
from tuner import *
import sys
import glob

# Função principal para processar dados, realizar tunagem de hiperparâmetros, executar backtesting e salvar resultados
def main_process(files: list, len_indic: np.array, year_backtest: int, year_val: int, tune: bool, binarized: bool, test_columns: bool, volatility_df: pd.DataFrame):
    """
    Processa uma lista de arquivos contendo dados financeiros, executa tunagem de hiperparâmetros (se ativada), 
    realiza backtesting e gera resultados consolidados.

    Parâmetros:
    - files (list): Lista de caminhos para arquivos contendo dados históricos das ações.
    - len_indic (np.array): Array contendo os números de indicadores a serem testados, Caso queira apenas um, de um array com um número.
    - year_backtest (int): Ano para executar o backtest.
    - year_val (int): Ano para validação durante a tunagem de hiperparâmetros.
    - tune (bool): Define se os hiperparâmetros devem ser otimizados antes do backtest.
    - binarized (bool): Indica se os dados foram binarizados antes do processamento.
    - test_columns (bool): Se verdadeiro, testa combinações diferentes de indicadores; caso contrário, usa apenas o primeiro.
    - volatility_df (pd.DataFrame): DataFrame contendo informações de volatilidade das ações.

    Retorna:
    - Salva os resultados em um arquivo CSV chamado 'results.csv'.
    - Para cada ativo, salva os hiperparâmetros otimizados (se `tune=True`) em um arquivo JSON na pasta 'hyperparams'.

    Fluxo Geral:
    1. Valida os anos de backtest e validação.
    2. Para cada ativo:
        - Lê os dados do arquivo.
        - Executa tunagem de hiperparâmetros (se ativada) ou carrega hiperparâmetros de arquivos existentes.
        - Realiza backtesting com os melhores parâmetros ou configuração padrão.
        - Armazena os resultados no DataFrame final.
    3. Salva os resultados finais em 'results.csv'.

    Observações:
    - O backtesting é realizado utilizando a biblioteca `Backtest`.
    - A volatilidade do ativo é adicionada ao resultado final se disponível no `volatility_df`.

    Exceções:
    - Se os anos de backtest e validação forem iguais ou não definidos, a função encerra com `sys.exit()`.
    - Captura erros específicos no backtesting e os imprime, mas continua com o próximo ativo.
    """
    # Inicializa uma lista para armazenar os resultados finais
    final_results = []

    # Itera sobre cada arquivo na lista de arquivos
    for element in files:
        json_ticker = {}
        # Extrai o ticker (símbolo da ação) do caminho do arquivo
        TICKER = element.split("\\")[-1][:-4]

        # Lê os dados do arquivo e define o índice como a data
        DATA = read_and_set_index(element)

        # Verifica se os anos de validação e backtest são válidos e diferentes
        if not year_val or not year_backtest or year_val == year_backtest:
            print("Erro de chamada: os anos de validação e backtest devem ser diferentes e determinados.")
            sys.exit()  # Encerra o programa se os anos forem inválidos

        # Itera sobre o número de indicadores especificados em len_indic
        for n_colunas in len_indic:
            best_params = None

            # Se a tunagem estiver ativada, otimiza os hiperparâmetros
            if tune:
                study_params, study_value = run_optimization(DATA, binarized, year_val, TICKER, n_trials=100, k_best=n_colunas)
                # Salva os parâmetros otimizados e o valor de desempenho no dicionário
                json_ticker[n_colunas] = {"params": study_params, "value": study_value}
                best_params = study_params
            else:
                # Define o caminho para o arquivo de hiperparâmetros
                hyperparams_path = os.path.join(
                    "hyperparams",
                    f"{TICKER}_{'b_' if binarized else ''}{year_val}.json"
                )

                # Se o arquivo de hiperparâmetros existir, carrega os parâmetros
                if os.path.exists(hyperparams_path):
                    with open(hyperparams_path, "r") as f:
                        dict_best_params = json.load(f)
                    # Se test_columns não estiver ativo, usa o primeiro conjunto de parâmetros
                    if not test_columns:
                        best_params = dict_best_params[dict_best_params.keys()[0]]
                    # Caso contrário, verifica se n_colunas está nos parâmetros
                    elif n_colunas in dict_best_params.keys():
                        best_params = dict_best_params[n_colunas]["params"]

            # Executa o modelo de backtesting com os melhores parâmetros ou parâmetros padrão
            if best_params is not None:
                policy, accuracy, dict_total, _ = backtesting_model(DATA, year_backtest, **best_params)
            else:
                try:
                    policy, accuracy, dict_total, _ = backtesting_model(DATA, year_backtest)
                except Exception as e:
                    # Captura erros durante o backtesting
                    print(f"Erro ao executar o backtesting para {TICKER} com {n_colunas} indicadores: {e}")
                    continue

            # Configura e executa o backtesting com a biblioteca Backtest
            bt = Backtest(policy, OurStrategy, cash=10000)
            stats = bt.run()

            # Extrai o retorno percentual e a taxa de vitória
            percent_return = stats["Return [%]"]
            win_rate = stats["Win Rate [%]"]

            # Obtém a volatilidade da ação no DataFrame fornecido
            try:
                vol = volatility_df[volatility_df['Stock'] == TICKER]['Volatility'].values[0]
            except KeyError:
                vol = None  # Caso não exista informação de volatilidade

            # Cria um dicionário com os resultados obtidos
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
            # Adiciona os resultados à lista final
            final_results.append(teste)
            print(f"{TICKER} processado com {n_colunas} indicadores")

            # Sai do loop de len_indic se test_columns estiver desativado
            if not test_columns:
                break

        # Ordena os elementos do dicionário com base no valor de desempenho
        if tune:
            json_ticker = dict(sorted(json_ticker.items(), key=lambda x: x[1]["value"], reverse=True))
            # Salva os parâmetros otimizados em um arquivo JSON
            file_path = os.path.join("hyperparams", f"{TICKER}_{'b_' if binarized else ''}{year_val}.json")
            with open(file_path, "w") as f:
                json.dump(json_ticker, f)

    # Converte a lista de resultados para um DataFrame
    final_df = pd.DataFrame(final_results)

    # Exibe e salva os resultados finais em um arquivo CSV
    print(final_df)
    final_df.to_csv("results.csv", index=False)
