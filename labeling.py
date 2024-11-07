import numpy as np
from tradingUtils import *

def log_returns(prices):
    """
    Calcula o retorno de uma sequência de preços.

    Parameters
    ----------
    prices : numpy array
        Série de preços.

    Returns
    -------
    numpy.ndarray
        Array de retornos logaritmicos.

    Raises
    ------
    ValueError
        Se o input tiver menos que dois preços.

    Examples
    --------
    >>> prices = [100, 105, 102, 108]
    >>> calculate_log_returns(prices)
    array([ 0.04879016, -0.02898754,  0.05658353])

    Notes
    -----
    Fórmula para calculo do retorno:

        r_t = ln(P_t / P_{t-1})

    where:
        r_t : Retorno logaritmico no tempo t
        P_t : Preço da ação no tempo t
        P_{t-1} : Preço da ação no tempo t-1
        ln : Logaritmo natural
    """
    # Confere se tem pelo menos dois preços.
    if prices.size < 2:
        raise ValueError("At least two price points are required to calculate returns.")

    # Cálcula retorno logarítmico.
    log_returns = np.log(prices[1:] / prices[:-1])

    return log_returns

def fixed_time_horizon_labeling(returns, h, tau):
    """
    Cálculo os rótulos para uma série de retornos utilizando o "Fixed Time Horizon Method"

    Parameters:
    ----------
    returns : numpy array
        Sequência de retornos.
    h : int
        O horizonte de tempo, em dias, para calcular o retorno.
    tau : float
        O limiar para classificar o retorno como positivo, negativo ou neutro.

    Returns:
    -------
    labels : numpy array
        Um array de labels com os valores:
            - 1 se o retorno nos próximos `h` dias é maior que`tau`.
            - 0 se o valor absoluto do retorno nos próximos `h` dias é menor ou igual a `tau`.
            - -1 se o retorno nos próximos `h` dias é menor que`tau`.
        O comprimento de `labels` será `len(returns) - h` para considerar o intervalo dos primeiros `h` dias.
    """    
    labels = np.zeros(len(returns) - h)
    
    
    # Limiares
    # Se tivermos um array de limiares, usamos o limiar definido com ewm
    try:
        if isinstance(tau, type(np.array([1]))) or isinstance(tau, type(pd.Series([1]))):
            for i in range(len(returns) - h):
                # Calcula o retorno cumulativo dos próximos h dias.
                cumulative_return = np.sum(returns[i:i + h])
                
                # Define o rótulo de acordo com o limiar.
                if cumulative_return > tau[i]:
                    labels[i] = 1
                elif abs(cumulative_return) <= tau[i]:
                    labels[i] = 0
                else:
                    labels[i] = -1
        # Senão, usamos o limiar fixo durante toda a rotulagem.
        else:
            for i in range(len(returns) - h):
                # Calcula o retorno cumulativo dos próximos h dias.
                cumulative_return = np.sum(returns[i:i + h])
                
                # Define o rótulo de acordo com o limiar.
                if cumulative_return > tau:
                    labels[i] = 1
                elif abs(cumulative_return) <= tau:
                    labels[i] = 0
                else:
                    labels[i] = -1
    except Exception as e:
        print("Tau precisa ser um inteiro, um array ou uma Series")
        print(f"Mensagem de erro: {e}")
        
    return labels

# Função só pra ter uma ideia do calculao pra volatilidade diária.
def getDailyVol(close,span0=5):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0 - 1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    
    return df0

prices = tsla_data["Adj Close"].to_numpy()

# # Calculando retorno logaritmico na série de preços.
returns = log_returns(prices)

# Limiar para definir rótulo.
tau = np.std(returns)

# Intervalo de dias a se considerar
h = 5

# Calcula desvio padrão móvel ponderado exponencialmente (EWMSD).
returns_series = pd.Series(returns)
dynamic_tau = returns_series.ewm(span = h).std()

labels = fixed_time_horizon_labeling(returns, h, tau)
labels_dynamic = fixed_time_horizon_labeling(returns, h, dynamic_tau)


def labelData(data, span=5):
    # Calculando retorno logaritmico da série de preços.
    returns = log_returns(data)

    # Calcula desvio padrão móvel ponderado exponencialmente (EWMSD).
    returns_series = pd.Series(returns)
    dynamic_tau = returns_series.ewm(span = span).std()
    labels_dynamic = fixed_time_horizon_labeling(returns, span, dynamic_tau)
    labels_dynamic = np.concatenate((np.zeros(6), labels_dynamic))

    return labels_dynamic
    
# Selecionando apenas preços Adj_Close
tsla_prices = tsla_data["Adj Close"].to_numpy()

label_tsla_data = labelData(tsla_prices)

if __name__ == "__main__":
    def iguais(num):
        print(num * "=")

    def iguais_emvolta(num, txt):
        print(num * "=" + txt + num * "=")
    
    # iguais(30)
    # print("prices shape:")
    # print(prices.shape)
    
    iguais(30)
    print("returns shape:")
    print(returns.shape)
    
    iguais(30)
    print("dynamic_tau shape:")
    print(dynamic_tau.shape)

    # iguais(30)
    # print("labels com limiar igual ao desvio padrao")
    # print(labels)
    # print(labels.shape)
    # print("labels com limiar dinamico com ewmsd")
    # print(labels_dynamic)
    # print(labels_dynamic.shape)

    iguais(30)
    # print("contagem de labels")
    # iguais_emvolta(10, "Labels com tau fixo (desvio padrão)")
    # unique_values, count = np.unique(labels, return_counts = True)
    # print("unique_values")
    # print(unique_values)
    # print("count")
    # print(count)

    iguais_emvolta(10, "Labels com tau dinamico")
    unique_values, count = np.unique(labels_dynamic, return_counts = True)
    print("unique_values")
    print(unique_values)
    print("count")
    print(count)

    print("label_tsla_data")
    print(label_tsla_data)
    print("label_tsla_data.shape")
    print(label_tsla_data.shape)