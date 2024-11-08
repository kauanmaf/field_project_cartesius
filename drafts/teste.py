from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tradingUtils import *
from indicadores import *
import labeling as lb
from backtesting import Backtest
from backtesting.lib import crossover

DATA = prio_data
YEAR = None

# Função a ser executada: faz o backtesting para um dado modelo e ano
# Se "year" for passado, ele faz o backtest nesse ano
# Se não, ele faz nos últimos 365 dias de dado
def backtesting_model(olhc, model, year = None, **kwargs):
    # Calculando os indicadores
    indicators = agg_indicators(olhc)
    indicators = normalize_indicators(indicators)
    # Calculando o rótulo
    y = np.array(lb.labelData(olhc["Adj Close"].to_numpy())).ravel()
    # Eliminando as linhas com NaN

    indicators["y"] = y
    indicators = indicators.dropna()
    # Separando os dados de treino e backtest com base no ano selecionado
    # Se tiver um ano específico, separa ele para backtest
    if year:
        indicators_train = indicators[indicators.index.year != year]
        indicators_backtest = indicators[indicators.index.year == year]
    # Se não, pega os últimos 365 dias
    else:
        last_day = indicators.index[-1]
        indicators_train = indicators[indicators.index <= last_day - pd.DateOffset(years = 1)]
        indicators_backtest = indicators[indicators.index > last_day - pd.DateOffset(years = 1)]
    # Convertendo para numpy arrays, caso ainda não estejam
    X = np.array(indicators_train)[:, :-1]
    y = np.array(indicators_train)[:, -1]
    # Treinando o modelo
    model = model(X, y, **kwargs)
    # Predizendo a política para aquele ano
    pred = model.predict(np.array(indicators_backtest)[:, :-1])
    # Salvando a predição em um dataframe adequado (próximas 4 linhas)
    # Pegando os dados originais do período de backtest
    if year:
        olhc_backtest = olhc[olhc.index.year == year]
    else:
        last_day = olhc.index[-1]
        olhc_backtest = olhc[olhc.index > last_day - pd.DateOffset(years = 1)]
    # Criando uma série com a predição e o index do ano
    policy = pd.Series(pred, index = olhc_backtest.index)
    # Colocando a predição nesse dataframe
    olhc_backtest["Signal"] = 0
    olhc_backtest.loc[policy.index, "Signal"] = policy

    return olhc_backtest

def random_forest(data, y, n_estimators=100, max_depth=None, random_state=42):
    """
    Treina um modelo de Random Forest e retorna as previsões e o relatório de classificação.

    Parâmetros:
    - data: DataFrame com as features e a variável alvo.
    - target_column: Nome da coluna alvo no DataFrame.
    - n_estimators: Número de árvores na floresta.
    - max_depth: Profundidade máxima das árvores (None para ilimitado).
    - random_state: Semente para a geração de números aleatórios.

    Retorna:
    - y_pred_rf: Previsões das classes no conjunto de teste.
    - report: Relatório de classificação.
    """
    # Divide os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    # Definindo o modelo Random Forest
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                 random_state=random_state)

    # Treina o modelo Random Forest
    rf.fit(X_train, y_train)

    # Faz previsões de classe
    y_pred_rf = rf.predict(X_test)

    # Exibe o relatório de classificação para o Random Forest
    report = classification_report(y_test, y_pred_rf)
    print(report)

    return rf

def mlp(data, y, hidden_layers=(100, 100, 100), activation='logistic', 
        solver='adam', max_iter=500, random_state=42):
    """
    Treina um modelo de rede neural MLP e retorna as previsões e o relatório de classificação.

    Parâmetros:
    - data: DataFrame contendo os dados de entrada.
    - hidden_layers: Tupla com o tamanho das camadas ocultas.
    - activation: Função de ativação a ser usada.
    - solver: Algoritmo de otimização a ser usado.
    - max_iter: Número máximo de iterações.
    - random_state: Semente para a geração de números aleatórios.

    Retorna:
    - y_pred_mlp: Previsões das classes no conjunto de teste.
    - report: Relatório de classificação.
    """
    # Divide os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    # Definindo a rede neural com múltiplas camadas
    mlp = MLPClassifier(hidden_layer_sizes = hidden_layers,
                        activation = activation,
                        solver = solver,
                        max_iter = max_iter,
                        random_state = random_state)

    # Treina a rede neural
    mlp.fit(X_train, y_train)

    # Faz previsões de classe
    y_pred_mlp = mlp.predict(X_test)

    # Exibe o relatório de classificação para o MLP
    report = classification_report(y_test, y_pred_mlp)
    print(report)

    return mlp

def gradient_boosting(data, y, random_state=42):
    """
    Treina um modelo de Gradient Boosting e retorna as previsões e o relatório de classificação.

    Parâmetros:
    - data: DataFrame com as features e a variável alvo.
    - target_column: Nome da coluna alvo no DataFrame.
    - random_state: Semente para a geração de números aleatórios.

    Retorna:
    - y_pred_gb: Previsões das classes no conjunto de teste.
    - report: Relatório de classificação.
    """
    # Divide os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    # Definindo o modelo Gradient Boosting
    gb_model = GradientBoostingClassifier(random_state=random_state)

    # Treina o modelo Gradient Boosting
    gb_model.fit(X_train, y_train)

    # Faz previsões de classe
    y_pred_gb = gb_model.predict(X_test)

    # Exibe o relatório de classificação para o Gradient Boosting
    report = classification_report(y_test, y_pred_gb)
    print(report)

    return gb_model

# Testando o modelo
dados_rf = backtesting_model(DATA, random_forest, year = YEAR)

bt = Backtest(dados_rf, OurStrategy, cash=10000)
stats = bt.run()

# # Exibindo o resultado
bt.plot()
print(stats)