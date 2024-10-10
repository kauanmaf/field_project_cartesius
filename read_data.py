import numpy as np
import pandas as pd

# Carregando os dados e ajustando o datetime
tsla_data = pd.read_csv("data/TSLA.csv")
tsla_data["Date"] = pd.to_datetime(tsla_data["Date"])
tsla_data = pd.Series(tsla_data["Adj Close"].values, index = tsla_data["Date"])

azul_data = pd.read_csv("data/AZUL4.SA.csv")
azul_data["Date"] = pd.to_datetime(azul_data["Date"])
azul_data = pd.Series(azul_data["Adj Close"].values, index = azul_data["Date"])

prio_data = pd.read_csv("data/PRIO3.SA.csv")
prio_data["Date"] = pd.to_datetime(prio_data["Date"])
prio_data = pd.Series(prio_data["Adj Close"].values, index = prio_data["Date"])

viva_data = pd.read_csv("data/VIVA3.SA.csv")
viva_data["Date"] = pd.to_datetime(viva_data["Date"])
viva_data = pd.Series(viva_data["Adj Close"].values, index = viva_data["Date"])