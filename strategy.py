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
        
        # Matriz com as duas médias lado a lado
        aux_matrix = np.hstack((long_average.values.reshape(-1, 1), short_average.values[short_average.size - long_average.size:].reshape(-1, 1)))
        
        # Vetor que analisa quando a média curta é maior que a longa
        bool_vector = aux_matrix[:, 1] > aux_matrix[:, 0]
        
        # Analisando quando a média curta cruza a longa
        bool_matrix = np.zeros((bool_vector.size + 1, 2))
        bool_matrix[1:, 0] = bool_vector
        bool_matrix[:-1, 1] = bool_vector
        evaluation = bool_matrix[1:-1, 1] - bool_matrix[1:-1, 0]
        evaluation = evaluation.astype(str)

        # TODO: Mudar o 0 de keep para zerar a posição, vender voce fica negativo
        # Rotulando quando comprar, vender e manter
        evaluation[evaluation == "0.0"] = "keep"
        evaluation[evaluation == "1.0"] = "buy"
        evaluation[evaluation == "-1.0"] = "sell"

        # Adicionando a data à série
        dated_evaluations = pd.Series(evaluation, long_average.index[1:])

        return dated_evaluations
        self.moving_average()

    
