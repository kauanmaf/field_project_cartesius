import pandas as pd
import numpy as np

# Classe que servirá como classe abstrata para as estratégias desenvolvidas.
class Strategy():
    def __init__(self) -> None:
        pass

    def run(self):
        pass

    def moving_average(self, data_column, date_column, size):
        # Número de dados
        n_of_data = data_column.size
        # Array para conter a média móvel
        moving_average = np.zeros((n_of_data - size + 1))

        # Primeira média móvel calculada como aritmética
        previous_mme = np.mean(data_column[0:size - 1])
        # Salvando a primeira média
        moving_average[0] = previous_mme

        # Multiplicador da fórmula
        alpha = 2/(size + 1)

        # Para cada média móvel...
        for i in range(1, n_of_data - size + 1):
            # Calcula a nova média
            new_mme = (data_column[size - 1 + i] - previous_mme) * alpha + previous_mme
            # Salva no array
            moving_average[i] = new_mme
            # Atualiza a anterior
            previous_mme = new_mme

        # Adiciona a data à série
        dated_average = pd.Series(moving_average, date_column[size - 1:])
        
        return dated_average

class MA_crossover(Strategy):
    def __init__(self, data, short_period, long_period) -> None:
        super().__init__()
        self.data = data
        self.short_period = short_period
        self.long_period = long_period
    
    def run(self):
        long_average = self.moving_average(self.data["Adj Close"], self.long_period)
        short_average = self.moving_average(self.data["Adj Close"], self.short_period)
        
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

    
