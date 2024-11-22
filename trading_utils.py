import pandas as pd
from backtesting import Strategy, Backtest

def read_and_set_index(file_csv):
    data = pd.read_csv(file_csv)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace = True)
    return data

class OurStrategy(Strategy):
    def init(self):
        self.initial_equity = None 
        self.stop_loss_triggered = False 
        self.current_signal = None  

    def next(self):
        try:
            if self.stop_loss_triggered and self.data.Signal[-1] == self.current_signal:
                return  

            if self.data.Signal[-1] == 1 and not self.position.is_long:
                self.position.close()
                self.buy()
                self.initial_equity = self.equity
                self.stop_loss_triggered = False  
                self.current_signal = 1
            
            elif self.data.Signal[-1] == -1 and not self.position.is_short:
                self.position.close()  
                self.sell()
                self.initial_equity = self.equity  
                self.stop_loss_triggered = False  
                self.current_signal = -1

            if self.initial_equity is not None and (self.equity / self.initial_equity - 1) <= -0.05:
                self.position.close()
                self.stop_loss_triggered = True 
                self.initial_equity = None 
            
            elif self.data.Signal[-1] == 0:
                self.position.close()
                self.initial_equity = None
                self.stop_loss_triggered = False

        except Exception as e:
            print(f"Error: {e}")