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