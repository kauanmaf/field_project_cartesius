from backtesting import Backtest, Strategy
import pandas as pd
from bokeh.models import DatetimeTickFormatter



tsla_data = pd.read_csv("data/TSLA.csv")
tsla_data["Date"] = pd.to_datetime(tsla_data["Date"])
tsla_data.set_index("Date", inplace=True)
# tsla_data = pd.Series(tsla_data["Adj Close"].values, index = tsla_data["Date"])

class MA_crossover_bt(Strategy):
    short_period = 10
    long_period = 50

    def init(self):
        self.short_period = short_period
        self.long_period = long_period
        # Inicializando as médias móveis: curta e longa
        self.short_ma = self.I(self.moving_average, self.data.Close, self.short_period)
        self.long_ma = self.I(self.moving_average, self.data.Close, self.long_period)
    
    def moving_average(self, data, size):
        """Função para calcular a média móvel exponencial dos dados"""
        return pd.Series(data).ewm(span=size, adjust=False).mean()

    def next(self):
        # Certifique-se de que temos dados suficientes para realizar a operação
        if len(self.data) < max(self.short_period, self.long_period):
            return
        
        # Calculando a diferença e a variação da média curta
        diff = self.short_ma[-1] - self.long_ma[-1]
        short_average_var = self.short_ma[-1] - self.short_ma[-2] if len(self.short_ma) > 1 else 0

        # Condições de compra e venda adaptadas da sua estratégia:
        # Compra se a média curta está maior que a longa e subindo
        if diff > 0 and short_average_var > 0 and not self.position:
            self.buy()
        
        # Venda se a média curta está menor que a longa e caindo
        elif diff < 0 and short_average_var < 0 and self.position:
            self.sell()

# Parâmetros da estratégia
short_period = 10
long_period = 50

# Executando o backtest
bt = Backtest(tsla_data, MA_crossover_bt, cash=10_000, commission=0)

# Executa o backtest
stats = bt.run()

# Exibe os resultados
print(stats)
plot = bt.plot()