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