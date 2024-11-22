import backtrader as bt
import pandas as pd

# Load the TSLA data
tsla_data = pd.read_csv("data/TSLA.csv")
tsla_data["Date"] = pd.to_datetime(tsla_data["Date"])
tsla_data = tsla_data[(tsla_data["Date"] > pd.to_datetime("2022-01-01")) & (tsla_data["Date"] < pd.to_datetime("2023-01-01"))]
tsla_data.set_index("Date", inplace=True)

class MA_Crossover(bt.Strategy):
    # Define the parameters for the strategy
    params = (('short_period', 10), ('long_period', 50),)
    
    def __init__(self):
        # Initialize moving averages
        self.short_ma = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.short_period)
        self.long_ma = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.long_period)
    
    def next(self):
        # Calculating the difference between short and long moving averages
        diff = self.short_ma[0] - self.long_ma[0]
        short_average_var = self.short_ma[0] - self.short_ma[-1] if len(self.data) > 1 else 0

        # Buy if short MA is above long MA and increasing
        if diff > 0 and short_average_var > 0 and not self.position:
            self.buy()

        # Sell if short MA is below long MA and decreasing
        elif diff < 0 and short_average_var < 0 and self.position:
            self.sell()

# Prepare the data for Backtrader
class PandasData(bt.feeds.PandasData):
    # Use adjusted close data
    lines = ('adj_close',)
    params = (('adj_close', 5),)

# Instantiate Cerebro engine
cerebro = bt.Cerebro()

# Add strategy
cerebro.addstrategy(MA_Crossover)

# Feed data to Cerebro
data = PandasData(dataname=tsla_data)
cerebro.adddata(data)

# Set initial capital
cerebro.broker.setcash(10000)

# Set commission
cerebro.broker.setcommission(commission=0)

# Run the backtest
results = cerebro.run()

# Print final portfolio value
print(f"Final Portfolio Value: {cerebro.broker.getvalue()}")

cerebro.plot()