import numpy as np
from tradingUtils import *

# def labelData(data, max_variation):
#     label = np.zeros((data.shape[0], 1))
    
#     for row in range(data.shape[0] - 1):
#         try:
            
#             close_current = data.iloc[row]["Adj Close"]
#             close_next = data.iloc[row + 1]["Adj Close"]

#             # Check for upward variation
#             if close_next > close_current and (close_next - close_current) * 100 / close_current >= max_variation:
#                 label[row] = 1
#             # Check for downward variation
#             elif close_next < close_current and (close_current - close_next) * 100 / close_current >= max_variation:
#                 label[row] = -1
           
#         except:
#             pass
#     label = pd.DataFrame(label, index=data.index, columns=['Label'])

#     return label

# Calcula retorno logaritmico.
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
        raise ValueError("Pelo menos dois preços são necessários para o cálculo de retorno.")

    # Cálcula retorno logarítmico.
    return np.log(prices[1:] / prices[:-1])

# Método do horizonte fixado para retornos com horizonte h e limiar(es) tau
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
    labels = np.zeros(len(returns) + 1)
    
    # Limiares
    # Se tivermos um array de limiares, usamos o limiar definido com ewm.
    try:
        if isinstance(tau, type(np.array([1]))) or isinstance(tau, type(pd.Series([1]))):
            for i in range(h, len(returns)):
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
            for i in range(h, len(returns)):
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

# Rotulagem usando o método do horizonte fixado com horizonte `h` e span `span` para
# cálculo dos limiares dinâmicos.
def labelDataFixedHorizon(data, h=3, span=5):

    # Calculando retorno logaritmico da série de preços.
    returns = log_returns(data)

    # Calcula desvio padrão móvel ponderado exponencialmente (EWMSD).
    returns_series = pd.Series(returns)
    dynamic_tau = returns_series.ewm(span = span).std()

    labels_dynamic = fixed_time_horizon_labeling(returns, h, dynamic_tau)

    return labels_dynamic

# Método de barreira tripla (stop-loss, profit-taking e limite de tempo) para rotulagem
# de dados.
def triple_barrier_labeling(prices, h, span, pt_factor=1, sl_factor=1):
    """
    Método de barreira tripla para rotulagem de dados financeiros.
    
    Parameters
    ----------
    prices : numpy array
        Série de preços.
    h : int
        Horizonte de tempo (em número de barras) até a barreira vertical.
    span : int
        Span para o cálculo do EWMSD para limiares dinâmicos.
    pt_factor : float
        Fator multiplicador para definir a barreira superior (profit-taking).
    sl_factor : float
        Fator multiplicador para definir a barreira inferior (stop-loss).
    
    Returns
    -------
    labels : numpy array
        Array de rótulos com 1 (profit-taking), -1 (stop-loss) ou 0 (barreira de tempo).
    """
    
    # Calcula os retornos logarítmicos.
    returns = log_returns(prices)
    returns_series = pd.Series(returns)

    # Calcula desvio padrão móvel ponderado exponencialmente (EWMSD).
    ewmstd = returns_series.ewm(span=span).std()

    labels = np.zeros(len(returns) + 1)
    
    for i in range(h, len(returns)):
        # Define os limites das barreiras superior e inferior para o período utilizando o limiar dinâmico.
        upper_barrier = pt_factor * ewmstd[i]
        lower_barrier = -sl_factor * ewmstd[i]
        
        # Calcula o retorno cumulativo ao longo dos próximos `h` dias.
        cumulative_return = np.cumsum(returns[i:i + h])
        
        # Verifica qual barreira é tocada primeiro
        for j, cum_ret in enumerate(cumulative_return):
            if cum_ret >= upper_barrier:  # Barreira superior (profit-taking)
                labels[i] = 1
                break
            elif cum_ret <= lower_barrier:  # Barreira inferior (stop-loss)
                labels[i] = -1
                break
        else:
            # Se nenhuma barreira superior/inferior for tocada, verifica a barreira de tempo
            labels[i] = 0
    
    return labels

# Rotulagem usando o método de barreira tripla.
def labelDataTripleBarrier(data, h=5, span=5, pt_factor=1, sl_factor=1):
    """
    Função principal para rotular a série de preços usando o método de barreira tripla.
    
    Parameters
    ----------
    data : numpy array
        Série de preços.
    h : int
        Horizonte de tempo (em número de barras) até a barreira vertical.
    span : int
        Span para cálculo do EWMSD.
    pt_factor : float
        Fator multiplicador para a barreira de lucro (superior).
    sl_factor : float
        Fator multiplicador para a barreira de perda (inferior).
    
    Returns
    -------
    numpy array
        Rótulos da série de preços.
    """
    return triple_barrier_labeling(data, h, span, pt_factor, sl_factor)

# Função que implementa ambas as funções de rotulagem, para facilitar.
def labelData(data: np.array, h: int=3 , span: int=5, triple_barrier: bool=True, pt_factor: float=1, sl_factor: float=1):
    if triple_barrier:
        return labelDataTripleBarrier(data, h, span, pt_factor, sl_factor)
    else:
        return labelDataFixedHorizon(data, h, span)

# Função só pra ter uma ideia do calculao pra volatilidade diária.
def getDailyVol(close,span0=5):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0 - 1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    
    return df0

prices = tsla_data["Adj Close"].to_numpy()
label_tsla_data = labelData(prices)

if __name__ == "__main__":
    def iguais(num):
        print(num * "=")

    def iguais_emvolta(num, txt):
        print(num * "=" + txt + num * "=")
    
    # Selecionando apenas preços Adj_Close
    iguais(30)
    iguais_emvolta(10, "Preços")
    prices = tsla_data["Adj Close"].to_numpy()
    print("prices")
    print(prices)
    print("prices.shape")
    print(prices.shape)

    bool_list = [True, False]
    for value in bool_list:
        labels = labelData(prices, triple_barrier=value)

        iguais(30)
        if value:
            print("Utilizando barreira tripla")
        else:
            print("Utilizando horizonte fixado")
        print("contagem de labels")
        print(labels.shape)
        iguais_emvolta(10, "Contagem dos valores de label")
        unique_values, count = np.unique(labels, return_counts = True)
        print("unique_values")
        print(unique_values)
        print("count")
        print(count)