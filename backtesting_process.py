from indicadores import *
from labeling import *
from models import *

# Função para calcular os indicadores do dado
def create_indicators(ohlc, **kwargs):
    # Calcula e agrega todos os indicadores
    indicators = agg_indicators(ohlc, **kwargs)
    # Normaliza eles
    indicators = normalize_indicators(indicators)
    # Retirando indicadores correlacionados
    # indicators = decorrelate(indicators)

    return indicators

# Função para separar os dados de treino e backtest
def train_backtest_split(indicators, year = None):
    # Se tiver um ano específico, separa ele para backtest
    if year:
        indicators_train = indicators[indicators.index.year != year]
        indicators_backtest = indicators[indicators.index.year == year]
    # Se não, pega os últimos 365 dias
    else:
        last_day = indicators.index[-1]
        indicators_train = indicators[indicators.index <= last_day - pd.DateOffset(years = 1)]
        indicators_backtest = indicators[indicators.index > last_day - pd.DateOffset(years = 1)]

    return indicators_train, indicators_backtest

# Função para criar um dataframe com os dados de backtest e a política
def adjust_policy_data(ohlc, year, policy):
    # Pegando os dados originais do período de backtest
    if year:
        ohlc_backtest = ohlc[ohlc.index.year == year].copy()
    else:
        last_day = ohlc.index[-1]
        ohlc_backtest = ohlc[ohlc.index > last_day - pd.DateOffset(years = 1)].copy()
    # Criando uma série com a predição e o index do ano
    policy = pd.Series(policy, index = ohlc_backtest.index)
    # Colocando a predição nesse dataframe
    ohlc_backtest["Signal"] = 0
    ohlc_backtest.loc[policy.index, "Signal"] = policy

    return ohlc_backtest

# Função principal que toma os dados e prediz a política para um ano específico
def backtesting_model(ohlc, year = None, n_estimators = 100, **kwargs):
    # Calculando os indicadores
    indicators = create_indicators(ohlc, **kwargs)
    # Calculando o rótulo
    y = np.array(labelData(ohlc["Adj Close"].to_numpy())).ravel()
    # Eliminando as linhas com NaN
    indicators["y"] = y
    indicators = indicators.dropna()

    # Separando os dados em treino e backtest
    indicators_train, indicators_backtest = train_backtest_split(indicators, year)

    # Convertendo para numpy arrays, caso ainda não estejam
    X = np.array(indicators_train)[:, :-1]
    y = np.array(indicators_train)[:, -1]

    # Separando os rótulos em vertentes de compra e de venda
    y_buy = y.copy()
    y_buy[y_buy == -1] = 0
    y_sell = y.copy()
    y_sell[y_sell == 1] = 0

    # Treinando os modelos
    model_buy = random_forest(X, y_buy, n_estimators = n_estimators)
    model_sell = random_forest(X, y_sell, n_estimators = n_estimators)

    # Predizendo a política para o ano de backtest
    policy_buy = model_buy.predict(np.array(indicators_backtest)[:, :-1])
    policy_sell = model_sell.predict(np.array(indicators_backtest)[:, :-1])

    policy = policy_buy + policy_sell

    total_accuracy = np.sum(policy == indicators_backtest["y"])/np.size(policy)
    print("Acurácia total: ", total_accuracy)

    # Juntando a política com os dados originais
    ohlc_backtest = adjust_policy_data(ohlc, year, policy)

    return ohlc_backtest