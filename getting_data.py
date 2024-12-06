import pandas as pd
import numpy as np
import yfinance as yf
import investpy as inv
import os
import json

# 
br = inv.get_stocks(country="brazil")

# Reset the index to make Date a column
def transform_yf_data(data_stock):
    data_stock.reset_index(inplace=True)  
    data_stock['Date'] = data_stock['Date'].dt.strftime('%Y-%m-%d')
    data_stock.columns = data_stock.columns.droplevel(1)
    data_stock.columns.name = None 
    data_stock.set_index("Date", inplace = True)
    return data_stock


def fetch_stocks_above_volume(data, volume_threshold=10_000_000):
    # Get all the stocks
    tickers = data['symbol'] + ".SA"
    qualified_stocks = {}

    for ticker in tickers:
        try:
            data_stock = yf.download(ticker, start="2010-01-01", end="2024-11-15")
            data_stock = transform_yf_data(data_stock)
            # Filter for 2024 data
            data_stock_2024 = data_stock[data_stock.index > "2023-12-31"].copy()
            if data_stock_2024.empty:
                print(f"No 2024 data available for {ticker}")
                continue

            # Calculate average volume
            avg_volume = int(data_stock_2024['Volume'].mean())
            if avg_volume >= volume_threshold:
                print(f"{ticker} qualifies with avg volume: {avg_volume}")
                qualified_stocks[ticker] = data_stock
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")

    return qualified_stocks


def calculate_volatility(ticker, data):
    # Filter for 2024 data
    data_stock = data[data.index > "2023-12-31"].copy()
    size = len(data_stock)
    try:
        # Calculate log returns
        data_stock['Log Return'] = np.log(data_stock['Adj Close'] / data_stock['Adj Close'].shift(1))

        # Calculate annualized volatility
        volatility = np.std(data_stock['Log Return'].dropna()) * np.sqrt(size)
        return volatility
    except Exception as e:
        print(f"Failed to calculate volatility for {ticker}: {e}")
        return None

# Main workflow
volume_threshold = 10_000_000
stocks = fetch_stocks_above_volume(br["symbol"], volume_threshold)

results = []

for ticker, stock in stocks.items():
    vol = calculate_volatility(ticker, stock)
    if vol is not None:
        results.append({"Stock": ticker, "Volatility": vol})
        file_path = os.path.join("data/new_stocks/", f"{ticker}.csv")
        stock.to_csv(file_path)

file_path = "data/new_stocks/volatility.json"
with open(file_path, "w") as f:
    json.dump(results, f)
# Convert to DataFrame for better representation
df_results = pd.DataFrame(results)
print(df_results)