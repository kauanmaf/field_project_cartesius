from backtesting_process import *
from tuner import *
import sys

# Dado
DATA = prio_data
# Ano no qual será feito o backtest
YEAR_BACKTEST = 2024
# Ano a ser usado como validação da tunagem de hiperparâmetros
YEAR_VAL = 2023
# Variável para ativar ou desativar a tunagem
TUNE = True

# Se a opção de tunagem for ativada, otimiza os hiperparâmetros
if TUNE:
    if not YEAR_VAL or YEAR_VAL == YEAR_BACKTEST:
        print("Erro de chamada: o ano de validação deve estar determinado e deve ser diferente do ano de backtest.")
        sys.exit()
    run_optimization(DATA, YEAR_VAL, n_trials = 5)

# Calculando a política para o dado e ano especificado
policy = backtesting_model(DATA, YEAR_BACKTEST)

# Calculando o backtest
bt = Backtest(policy, OurStrategy, cash=10000)
stats = bt.run()

# Exibindo os resultados
bt.plot()
print(stats)