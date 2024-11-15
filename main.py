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
# Variável para ativar ou desativar a binarização dos dados
BINARIZED = True

# Se a opção de tunagem for ativada, otimiza os hiperparâmetros
if TUNE:
    if not YEAR_VAL or not YEAR_BACKTEST or YEAR_VAL == YEAR_BACKTEST:
        print("Erro de chamada: os anos de validação e backtest devem estar determinados e devem ser diferentes.")
        sys.exit()
    run_optimization(DATA, BINARIZED, YEAR_VAL, n_trials = 100)

# Pegando o caminho do arquivo com os hiperparâmetros
data_name = [name for name, value in globals().items() if value is DATA]
if BINARIZED:   
    hyperparams_path = os.path.join("hyperparams", f"{data_name[0]}_b_{str(YEAR_VAL)}.json")
else:
    hyperparams_path = os.path.join("hyperparams", f"{data_name[0]}_{str(YEAR_VAL)}.json")

# Se o arquivo existir...
if os.path.exists(hyperparams_path):
    # Pega os hiperparâmetros
    with open(hyperparams_path, "r") as f:
        best_params = json.load(f)

    # Calcula a política com esses parâmetros
    policy = backtesting_model(DATA, BINARIZED, YEAR_BACKTEST, **best_params)

# Se não, utiliza os argumentos padrão
else:
    policy = backtesting_model(DATA, BINARIZED, YEAR_BACKTEST)

# Calculando o backtest
bt = Backtest(policy, OurStrategy, cash=10000)
stats = bt.run()

# Exibindo os resultados
bt.plot()
print(stats)