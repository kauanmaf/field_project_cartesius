from labeling import *
from trading_utils import *
from backtesting_process import *
import matplotlib.pyplot as plt
import ast

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

        equities[accuracy] = max(0.001, stats["Equity Final [$]"])

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

positions = {(0, 0): "AZUL4.SA",
             (1, 0): "PRIO3.SA",
             (0, 1): "TSLA",
             (1, 1): "VIVA3.SA"}

def r_squared(poly1d_fn, x, y):
    # Calculando o R²
    y_pred = poly1d_fn(x)  # Valores previstos pela regressão
    ss_res = np.sum((y - y_pred) ** 2)  # Soma dos quadrados dos resíduos
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Soma dos quadrados totais
    r_squared = 1 - (ss_res / ss_tot)  # Cálculo do R²

    return r_squared


def linear_regression(x, y):
    coef = np.polyfit(x, y, 1)  # Coeficientes da regressão linear
    poly1d_fn = np.poly1d(coef)

    return poly1d_fn


def plot_graph(data_x, data_y, xmin, xmax, ymin, ymax, axes, row, column, graph, title, label = None, regression = False, color = "#1c285c", font = None):
    if graph == "bar":
        axes[row, column].bar(data_x["data"], data_y["data"], label = label, color = color)
    elif graph == "scatter":
        axes[row, column].scatter(data_x["data"], data_y["data"], label = label,  color = color)
    else:
        axes[row, column].plot(data_x["data"], data_y["data"], label = label,  color = color)
        
    if regression:
        poly1d_fn = linear_regression(data_x["data"], data_y["data"])
        r2 = round(r_squared(poly1d_fn, data_x["data"], data_y["data"]), 2)
        axes[row, column].plot(data_x["data"], poly1d_fn(data_x["data"]), color='red', label=rf"$R^2$ = {r2}")
        axes[row, column].legend()

    axes[row, column].set_title(title, font = font, color = "black")
    axes[row, column].set_xlim(xmin, xmax)
    axes[row, column].set_ylim(ymin, ymax)


def plot_graphs(data, x, y, graph, both = False, regression = False, font = None):
    fig, axes = plt.subplots(2, 2)
    axes.flatten()

    data_x = pd.DataFrame()
    data_y = pd.DataFrame()

    data_x["Stock"] = data["Stock"]
    data_y["Stock"] = data["Stock"]

    data_x["binarized"] = data["binarized"]
    data_y["binarized"] = data["binarized"]

    if "/" in x:
        trade, stat = x.split("/")
        data_x["data"] = data["Total Dict"].apply(lambda x: x[trade][stat])
    else:
        data_x["data"] = data[x]

    if "/" in y:
        trade, stat = y.split("/")
        data_y["data"] = data["Total Dict"].apply(lambda x: x[trade][stat])
    else:
        data_y["data"] = data[y]

    xmin, xmax = np.min(data_x["data"]), np.max(data_x["data"])
    ymin, ymax = np.min(data_y["data"]), np.max(data_y["data"])

    for position, title in positions.items():
        if both:
            plot_graph(data_x[(data_x["Stock"] == title) & (data_x["binarized"] == 0)], 
                       data_y[(data_y["Stock"] == title) & (data_y["binarized"] == 0)], 
                       xmin, xmax, ymin, ymax, axes, position[0], position[1], graph, title, label = "NB", regression = regression, color = "#FF8C00", font = font)
            plot_graph(data_x[(data_x["Stock"] == title) & (data_x["binarized"] == 1)], 
                       data_y[(data_y["Stock"] == title) & (data_y["binarized"] == 1)], 
                       xmin, xmax, ymin, ymax, axes, position[0], position[1], graph, title, label = "B", regression = regression, font = font)
            axes[position[0], position[1]].legend()
        else:
            plot_graph(data_x[data_x["Stock"] == title], data_y[data_y["Stock"] == title], xmin, xmax, ymin, ymax, axes, position[0], position[1], graph, title, regression = regression,font = font)

    axes[0, 0].set_ylabel(y, font = font,color = "black")
    axes[1, 0].set_ylabel(y, font = font,color = "black")
    axes[1, 0].set_xlabel(x, font = font,color = "black")
    axes[1, 1].set_xlabel(x, font = font,color = "black")

    fig.suptitle(f"{x} x {y}", font = font)

    plt.tight_layout()
    plt.show()

# Plot de frequência de indicadores
def plot_frequencia_indicarores():
    # Carregando dataset
    df_resultado_junto = pd.read_csv("resultado_junto.csv")

    # Transformando strings em listas
    df_resultado_junto["Selected Features"] = df_resultado_junto["Selected Features"].apply(ast.literal_eval)
    # Expandindo dataset colocando os indicadores em múltiplas linhas
    df_resultado_junto = df_resultado_junto.explode("Selected Features")
    # Retirando n_estimators
    df_resultado_junto = df_resultado_junto[df_resultado_junto['Selected Features'] != "n_estimators"]

    # Dicionários
    dicionario_valores = {
        "awesome_window": 2,
        "kst": 9,
        "stc": 4,
        "mass_window": 2,
    }
    dicionario = {
        "awesome_window1": "awesome_window",
        "awesome_window2": "awesome_window",
        "kst_n1": "kst",
        "kst_n2": "kst",
        "kst_n3": "kst",
        "kst_n4": "kst",
        "kst_r1": "kst",
        "kst_r2": "kst",
        "kst_r3": "kst",
        "kst_r4": "kst",
        "kst_signal": "kst",
        "stc_window_fast": "stc",
        "stc_window_slow": "stc",
        "stc_smooth1": "stc",
        "stc_smooth2": "stc",
        "stc_cycle": "stc",
        "mass_window_fast": "mass_window",
        "mass_window_slow": "mass_window",
    }

    df_resultado_junto['Selected Features'] = df_resultado_junto['Selected Features'].replace(dicionario)

    # Contando indicadores utilizados
    contagem_indicadores = df_resultado_junto["Selected Features"].value_counts()

    # Ajustando contagens para contar cada indicador uma única vez
    contagens_ajustadas = contagem_indicadores / contagem_indicadores.index.map(dicionario_valores).fillna(1) 
    # Ordenando contagens
    contagens_ajustadas_ordenadas = contagens_ajustadas.sort_values(ascending=False)

    # Criando gráfico de frequências de indicaodres utilizados
    plt.figure(figsize=(12, 9))
    sns.barplot(x=contagens_ajustadas_ordenadas.index, y=contagens_ajustadas_ordenadas.values, color="#1c285c")
    plt.title("Frequências de melhores indicadores selecionados para todas as empresas")
    plt.xlabel("Indicador")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45, ha='right')
    plt.show()
