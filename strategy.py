import pandas as pd
import numpy as np

# Classe que servirá como classe abstrata para as estratégias desenvolvidas.
class Strategy():
    def __init__(self) -> None:
        pass

    def run(self):
        pass

    # Função para calcular a média móvel exponencial dos dados
    def moving_average(self, data: pd.Series, size: int) -> pd.Series:
        average = data.ewm(span = size, adjust = False).mean()
        return average


class MA_crossover(Strategy):
    def __init__(self, data: pd.Series, short_period: int, long_period: int) -> None:
        super().__init__()
        self.data = data
        self.short_period = short_period
        self.long_period = long_period
    
    def run(self):
        long_average = self.moving_average(self.data, self.long_period)
        short_average = self.moving_average(self.data, self.short_period)
        
        # Fazendo a interseção dos valores das médias de acordo com as datas
        long_average_aligned, short_average_aligned = long_average.align(short_average, join = "inner")
        # Calculando a diferença entre as médias
        diff = short_average_aligned - long_average_aligned
        # Calculando a variação diária da média curta
        short_average_var = short_average_aligned.diff()
        
        # Se a média curta está maior que a longa e está subindo, compra
        diff[(diff * short_average_var > 0) & (diff > 0)] = 1
        # Se a média curta está menor que a longa e está descendo, vende
        diff[(diff * short_average_var > 0) & (diff < 0)] = -1
        # Se a média curta está menor que a longa e está subindo ou maior e descendo, fica neutro
        diff[diff * short_average_var < 0] = 0

        return diff

class BollingerBands(Strategy):
    # Setando o padrão com o tamanho sendo 20 e o desvio padrão sendo 2
    def __init__(self, data: pd.Series, period: int = 20, num_std: float = 2.0):
        super().__init__()
        self.data = data
        self.period = period
        self.num_std = num_std

    def run(self):
        # Calculando a média móvel e o desvio padrão
        mid_band = self.moving_average(self.data, self.period)
        rolling_std = self.data.rolling(window=self.period).std()
        
        # Calculando as duas linhas 
        upper_band = mid_band + (self.num_std * rolling_std)
        lower_band = mid_band - (self.num_std * rolling_std)
        
        # Inicializando os sinais como 0
        signal = pd.Series(0, index=self.data.index)
        
        # Gerando os sinais de compra e venda
        signal[self.data < lower_band] = 1  
        signal[self.data > upper_band] = -1 
        
        return signal


class MACD(Strategy):
    # Inicializamos nossa estratégia com os padrões para o macd, com o short sendo 12, o longo sendo 26 e a média móvel sendo 9
    def __init__(self, data: pd.Series, short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> None:
        super().__init__()
        self.data = data
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period

    def run(self):
        # Calculando a média as duas médias móveis
        short_ema = self.moving_average(self.data, self.short_period)
        long_ema = self.moving_average(self.data, self.long_period)
        
        # A linha macd é a diminuição de uma pela outra
        macd_line = short_ema - long_ema
        
        # A linha de sinal é 
        signal_line = self.moving_average(macd_line, self.signal_period)
        
        # Calculate the difference between the MACD line and the signal line
        diff = macd_line - signal_line
        
        # Generate buy, sell, and hold signals
        diff[diff > 0] = 1  # Buy signal when MACD is above the signal line
        diff[diff < 0] = -1  # Sell signal when MACD is below the signal line
        diff[diff == 0] = 0  # Neutral signal when MACD equals the signal line
        
        return diff
