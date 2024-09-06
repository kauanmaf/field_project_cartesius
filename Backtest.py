import pandas as pd

class Backtest():
    def __init__(self, stock: pd.Series, function: function, cash: int, time_start: str, time_stop: str):
        self.stock = stock
        self.function = function
        self.cash = cash
        self.time_start = time_start
        self.time_stop = time_stop
    
    def stats(self):
        # Filtrando os dados para o período selecionado
        filtered_data = self.stock[(self.stock.index >= pd.to_datetime(self.time_start)) & (self.stock.index <= pd.to_datetime(self.time_stop))]
        
