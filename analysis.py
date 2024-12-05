from labeling import *
from trading_utils import *
from backtesting_process import *
import matplotlib.pyplot as plt

def test_accuracy(data):
    data_copy = data.copy()

    y = np.array(labelData(data_copy["Adj Close"].to_numpy())).ravel()

    equities = {}

    for accuracy in range(0, 101, 10):
        y_wrong = y.copy()
        error = 100 - accuracy

        num_to_change = int(len(y_wrong) * error/100)
        indices_to_change = np.random.choice(len(y_wrong), num_to_change, replace = False)

        for index in indices_to_change:
            current_value = y_wrong[index]
            new_value = np.random.choice([v for v in [-1, 0, 1] if v != current_value])
            y_wrong[index] = new_value

        policy = pd.Series(y_wrong, index = data_copy.index)
        data_copy["Signal"] = 0
        data_copy.loc[policy.index, "Signal"] = policy

        bt = Backtest(data_copy, OurStrategy, cash=10000)
        stats = bt.run()

        equities[accuracy] = stats["Equity Final [$]"]

    return equities


def test_train_time(data, value):
    values = {}
    min_year = data.index.year.min()
    start_years = range(min_year, 2024)

    for start_year in start_years:
        filtered_data = data[data.index.year >= start_year]
        policy = backtesting_model(filtered_data, True, 2024)[0]

        bt = Backtest(policy, OurStrategy, cash=10000)
        stats = bt.run()
        equity = stats[value]

        values[start_year] = equity

    return values


def plot_graphs_style(df, x_col, y_col, title, normal_value, 
                      font=None, 
                      specific_offsets=None, 
                      palette="#1c285c", 
                      show_legend=True):
    # Adicionando a reta de regressão
    x = df[x_col]
    y = df[y_col]
    coef = np.polyfit(x, y, 1)  # Coeficientes da regressão linear
    poly1d_fn = np.poly1d(coef)  # Função para a linha de regressão

    # Cálculo do R²
    y_pred = poly1d_fn(x)
    r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    # Configurações do gráfico
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 9))

    # Criar o scatterplot
    sns.scatterplot(data=df, x=x_col, y=y_col,s= 100, color=palette, legend=show_legend)

    # Adicionando o nome das ações
    texts = []
    for i in range(len(df)):
        ticker = df["Stock"].iloc[i][:-3] if df["Stock"].iloc[i][-3:] == ".SA" else df["Stock"].iloc[i]
        offset = specific_offsets.get(ticker, normal_value)
        texts.append(plt.text(df[x_col].iloc[i], 
                              df[y_col].iloc[i] + offset, 
                              ticker, 
                              fontsize=13, 
                              ha='center', 
                              va='center', 
                              font=font))

    # Adicionando a reta de regressão
    plt.plot(x, poly1d_fn(x), color="#e42e2e", label=rf"$R^2$ = {r_squared:.2f}")

    # Remover as bordas do gráfico (spines)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Ajustes no gráfico
    plt.title(title, fontsize=14, font=font, color="black")
    plt.xlabel(x_col, font=font, color="black",fontsize=18)
    plt.ylabel(y_col, font=font, color="black",fontsize=18)
    
    if show_legend:
        plt.legend(frameon=False)
        
    plt.grid(False)
    plt.tight_layout()

    # Exibir o gráfico
    plt.show()