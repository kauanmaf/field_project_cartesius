import pandas as pd
from backtesting import Strategy, Backtest

# Carregando os dados e ajustando o datetime
tsla_data = pd.read_csv("data/TSLA.csv")
tsla_data["Date"] = pd.to_datetime(tsla_data["Date"])
tsla_data.set_index("Date", inplace = True)

azul_data = pd.read_csv("data/AZUL4.SA.csv")
azul_data["Date"] = pd.to_datetime(azul_data["Date"])
azul_data.set_index("Date", inplace = True)

prio_data = pd.read_csv("data/PRIO3.SA.csv")
prio_data["Date"] = pd.to_datetime(prio_data["Date"])
prio_data.set_index("Date", inplace = True)

viva_data = pd.read_csv("data/VIVA3.SA.csv")
viva_data["Date"] = pd.to_datetime(viva_data["Date"])
viva_data.set_index("Date", inplace = True)


# Comum a todos, define os pontos de saída
def exit_points(data, policy, max_window, win_rate):
    current_state = 0
    days_on_state = 0
    win_goal = 0
    start_value = 0
    for date, action in zip(policy.index, policy.values):
        if action != 0:
            days_on_state = 0
            start_value = data.loc[date]["Adj Close"]
            win_goal = win_rate*(data.loc[date]["Adj Close"] - data.loc[date]["Open"])
            if action == 1:
                current_state = 1
            else:
                current_state = -1
            continue
        
        current_diff = data.loc[date]["Adj Close"] - start_value
        days_on_state += 1

        if days_on_state <= max_window and abs(current_diff) < win_goal:
            if current_state == 1:
                policy[date] = 1
            elif current_state == -1:
                policy[date] = -1

# class OurStrategy(Strategy):
#     def init(self):
#         pass

#     def next(self):
#         try:
#             if self.data.Signal[-1] == 1:
#                 if not self.position.is_long:
#                     self.position.close()  # Close any existing short position
#                     self.buy()

#             # Sell signal
#             elif self.data.Signal[-1] == -1:
#                 if not self.position.is_short:
#                     self.position.close()  # Close any existing long position
#                     self.sell()

#             # Close all positions if Signal is 0
#             elif self.data.Signal[-1] == 0:
#                 self.position.close()
#         except:
#             pass
class OurStrategy(Strategy):
    def init(self):
        super().init()
        # Initialize strategy parameters
        self.entry_price = None
        self.last_sell_date = None
        self.wait_days = 5
        self.stop_loss_threshold = 0.95
        
        
        # Register Signal as a custom indicator
        self.signal = self.I(lambda: self.data.Signal)

    def next(self):
        # Get current price
        current_price = self.data.Close[-1]
        current_signal = self.signal[-1]
        
        # Calculate wait period
        if self.last_sell_date is not None:
            days_since_sell = (self.data.index[-1] - self.last_sell_date).days
            wait_over = days_since_sell >= self.wait_days
        else:
            wait_over = True

        # Buy signal conditions
        if current_signal == 1 and wait_over and not self.position:
            self.buy()
            self.entry_price = current_price

        # Sell signal conditions
        elif current_signal == -1 and self.position:
            self.position.close()
            self.last_sell_date = self.data.index[-1]

        # Stop-loss check for long positions
        elif self.position and self.entry_price is not None:
            if current_price <= self.entry_price * self.stop_loss_threshold:
                self.position.close()
                self.last_sell_date = self.data.index[-1]

        # Close positions on neutral signal
        elif current_signal == 0 and self.position:
            self.position.close()


def run_signal_policy(tsla_data, policy_function, policy_name, body=None, exec_back = True):
    # Gerando os sinais que dão match com a política
    if body is not None:
        policy = policy_function(tsla_data, body)
    else:
        policy = policy_function(tsla_data)

    # Definindo os pontos de saída
    exit_points(tsla_data, policy, 5, 1)

    # Adicionando uma coluna sinal
    # Ensure 'Signal' column is float before assignment


    tsla_data["Signal"] = 0
    tsla_data['Signal'] = tsla_data['Signal'].astype(float)
    tsla_data.loc[policy.index, "Signal"] = policy.astype(float)


    # Rodando o backtest
    if exec_back:
        bt = Backtest(tsla_data, OurStrategy, cash=10000)
        stats = bt.run()

    # Renomeando a coluna de sinal
    tsla_data.rename(columns={"Signal": f"Signal_{policy_name}"}, inplace=True)