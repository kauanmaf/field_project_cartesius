import pandas as pd
from backtesting import Strategy

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

class OurStrategy(Strategy):
    def init(self):
        # Inicializar qualquer coisa necessária, se houver
        pass

    def next(self):
        # Se sinal for 1, compra apenas se não há posição aberta
        if self.data.Signal[-1] == 1 and not self.position.is_long:
            self.buy()
        elif self.data.Signal[-1]== 0 and (self.position.is_short or self.position.is_long):
            self.position.close()
        # Se sinal for -1, vende apenas se há uma posição aberta (long)
        elif self.data.Signal[-1] == -1 and self.position.is_short:
            self.sell()