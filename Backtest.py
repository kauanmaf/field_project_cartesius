import pandas as pd

class Backtest():
    def __init__(self, stock: pd.DataFrame, function: function, cash: int):
        self.stock = stock
        self.function = function
        self.cash = cash
    
    def stats(self):
        """
        TODO: adicionar filtro de data
        """
        
        
