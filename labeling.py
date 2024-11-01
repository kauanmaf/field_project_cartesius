import numpy as np
from tradingUtils import *

def labelData(data, max_variation):
    label = np.zeros((data.shape[0], 1))
    
    for row in range(data.shape[0] - 1):
        try:
            
            close_current = data.iloc[row]["Adj Close"]
            close_next = data.iloc[row + 1]["Adj Close"]

            # Check for upward variation
            if close_next > close_current and (close_next - close_current) * 100 / close_current >= max_variation:
                label[row] = 1
            # Check for downward variation
            elif close_next < close_current and (close_current - close_next) * 100 / close_current >= max_variation:
                label[row] = -1
        
        except:
            pass

    return label

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
    print("tipo do tau:")
    print(type(tau))
    # Se tivermos um array de limiares, usamos o limiar definido com ewm
    
    try:
        if isinstance(np.array, tau) or isinstance(pd.Series, tau):
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
    except:
        print("ERRO: tau precisa ser um inteiro, um array ou uma Series")
        
    return labels

# Função só pra ter uma ideia do calculao pra volatilidade diária.
def getDailyVol(close,span0=5):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0 - 1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    
    return df0

# Selecionando apenas preços Adj_Close
prices = tsla_data["Adj Close"].to_numpy()

# Calculando retorno logaritmico na série de preços.
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

label_tsla_data = labelData(tsla_data, 0.1)

if __name__ == "__main__":
    def iguais():
        print(50 * "=")
    
    iguais()
    print("prices shape:")
    print(prices.shape)
    
    iguais()
    print("returns shape:")
    print(returns.shape)
    
    iguais()
    print("returns_series shape:")
    print(returns_series.shape)

    iguais()
    print("labels com limiar igual ao desvio padrao")
    print(labels)
    print(labels.shape)
    print("labels com limiar dinamico com ewmsd")
    print(labels_dynamic)
    print(labels_dynamic.shape)

