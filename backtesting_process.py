"""
Módulo o qual serviu para criar juntar todos os indicadores que rivemos, bem como criar os sinais.
"""
from indicators import *
from labeling import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel


# Função para calcular os indicadores do dado
def create_indicators(ohlc, **kwargs):
    # Calcula e agrega todos os indicadores
    indicators = agg_indicators(ohlc, **kwargs)
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
def backtesting_model(ohlc, binarized, year=None, n_estimators=100, n_features=None, show_report = False, **kwargs):
    # Calculando os indicadores
    indicators = create_indicators(ohlc, **kwargs)
    y = np.array(labelData(ohlc["Adj Close"].to_numpy())).ravel()
    indicators["y"] = y
    indicators = indicators.dropna()

    # Separando os dados em treino e backtest
    indicators_train, indicators_backtest = train_backtest_split(indicators, year)

    # Convertendo para numpy arrays
    X = np.array(indicators_train)[:, :-1]
    y = np.array(indicators_train)[:, -1]

    selected_columns = indicators_train.columns[:-1]  # Nomes das colunas originais
    if n_features:
        # Seleção de features
        base_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        base_model.fit(X, y)
        selector = SelectFromModel(base_model, max_features=n_features, threshold=-np.inf)
        X = selector.transform(X)
        X_backtest = selector.transform(np.array(indicators_backtest)[:, :-1])
        selected_columns = np.array(indicators_train.columns[:-1])[selector.get_support()]  # Atualiza com as selecionadas
    else:
        X_backtest = np.array(indicators_backtest)[:, :-1]  # Make sure this is used

    # Se for ativada a binarização...
    if binarized:
        # Separa os rótulos em vertentes de compra e de venda
        y_buy = y.copy()
        y_buy[y_buy == -1] = 0
        y_sell = y.copy()
        y_sell[y_sell == 1] = 0

        # Treina os modelos
        model_buy = RandomForestClassifier(n_estimators = n_estimators, random_state = 42)
        model_buy.fit(X, y_buy)
        model_sell = RandomForestClassifier(n_estimators = n_estimators, random_state = 42)
        model_sell.fit(X, y_sell)

        # Prediz a política para o ano de backtest
        policy_buy = model_buy.predict(X_backtest)  # Use X_backtest
        policy_sell = model_sell.predict(X_backtest)  # Use X_backtest

        policy = policy_buy + policy_sell

    else:
        # Treina o modelo
        model = RandomForestClassifier(n_estimators = n_estimators, random_state = 42)
        model.fit(X, y)

        # Prediz a política para aquele ano
        policy = model.predict(X_backtest)  # Use X_backtest

    # Exibindo os resultados do modelo
    if show_report == True:
        print(classification_report(np.array(indicators_backtest)[:, -1], policy))
    report = classification_report(np.array(indicators_backtest)[:, -1], policy, output_dict=True)
    accuracy = report["accuracy"]

    # Juntando a política com os dados originais
    ohlc_backtest = adjust_policy_data(ohlc, year, policy)

    return ohlc_backtest, accuracy, report ,selected_columns
