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
        self.initial_equity = None  # Stores initial equity when opening a position
        self.stop_loss_triggered = False  # Tracks if stop-loss has been triggered for current position
        self.current_signal = None  # Stores the last signal to avoid repeated sells

    def next(self):
        print(self.data.index[-1], self.stop_loss_triggered, self.data.Signal[-1])
        try:
            # Avoid repeated sell signals if stop-loss was triggered and signal remains the same
            if self.stop_loss_triggered and self.data.Signal[-1] == self.current_signal:
                return  

            # Buy Signal
            if self.data.Signal[-1] == 1 and not self.position.is_long:
                self.position.close()  # Close any short position
                self.buy()
                self.initial_equity = self.equity  # Save initial equity for stop-loss
                self.stop_loss_triggered = False  # Reset stop-loss on new position
                self.current_signal = 1

            # Sell Signal
            elif self.data.Signal[-1] == -1 and not self.position.is_short:
                self.position.close()  # Close any long position
                self.sell()
                self.initial_equity = self.equity  # Save initial equity for stop-loss
                self.stop_loss_triggered = False  # Reset stop-loss on new position
                self.current_signal = -1

            # Stop-loss Check (5% drop from initial equity)
            if self.initial_equity is not None and (self.equity / self.initial_equity - 1) <= -0.05:
                self.position.close()  # Close position if loss exceeds 5%
                self.stop_loss_triggered = True  # Mark stop-loss as triggered
                self.initial_equity = None  # Reset initial equity tracking

            # Neutral Signal: Close any position
            elif self.data.Signal[-1] == 0:
                self.position.close()
                self.initial_equity = None  # Reset initial equity on closing position
                self.stop_loss_triggered = False  # Reset stop-loss trigger for new trades

        except Exception as e:
            print(f"Error: {e}")



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