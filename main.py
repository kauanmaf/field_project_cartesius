from backtesting_process import *
from tuner import *
from main_process import *
import glob
import sys

# Leitura dos arquivos de volatilidade
with open("data/new_stocks/volatility.json", "r") as f:
    volatility = json.load(f)

# Converte os dados de volatilidade para um DataFrame
volatility_df = pd.DataFrame(volatility)

FILES = glob.glob("data/*.csv")

LEN_INDIC = np.arange(3,28)  # Intervalo de tamanhos para k_best
# Ano no qual será feito o backtest
YEAR_BACKTEST = 2024
# Ano a ser usado como validação da tunagem de hiperparâmetros
YEAR_VAL = 2023
# Variável para ativar ou desativar a tunagem
TUNE = True
# Variável para ativar ou desativar a binarização dos dados
BINARIZED = False
# Variável para testar várias quantidades de colunas
TEST_COLUMS = True

main_process(FILES, LEN_INDIC, YEAR_BACKTEST, YEAR_VAL, TUNE, BINARIZED, TEST_COLUMS, volatility_df=volatility_df)