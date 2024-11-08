from indicadores import *
from labeling import *

# Função para calcular os indicadores do dado
def create_indicators(ohlc):
    # Calcula e agrega todos os indicadores
    indicators = agg_indicators(ohlc)
    # Normaliza eles
    indicators = normalize_indicators(indicators)

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


def backtesting_model(ohlc, model, year = None, **kwargs):
    # Calcula os indicadores
    indicators = create_indicators(ohlc)

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
    
    # Treinando o modelo
    model = model(X, y, **kwargs)
    # Predizendo a política para aquele ano
    pred = model.predict(np.array(indicators_backtest)[:, :-1])
    # Salvando a predição em um dataframe adequado (próximas 4 linhas)
    # Pegando os dados originais do período de backtest
    if year:
        ohlc_backtest = ohlc[ohlc.index.year == year]
    else:
        last_day = ohlc.index[-1]
        ohlc_backtest = ohlc[ohlc.index > last_day - pd.DateOffset(years = 1)]
    # Criando uma série com a predição e o index do ano
    policy = pd.Series(pred, index = ohlc_backtest.index)
    # Colocando a predição nesse dataframe
    ohlc_backtest["Signal"] = 0
    ohlc_backtest.loc[policy.index, "Signal"] = policy

    return ohlc_backtest