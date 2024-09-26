import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSM", "AVGO", "TSLA", "TCEHY", "ORCL", "ASML", "SSNLF", "NFLX", "SAP", "AMD", "CRM", "BABA", "ADBE", "CSCO", "QCOM", "TXN", "NOW", "PDD", "INTU", "AMAT", "UBER", "SLA", "SU.PA", "BKNG", "ANET", "MU", "SONY", "ADI", "ADP", "PANW", "LRCX", "KLAC", "MELI", "SHOP", "FI", "INTC", "DELL", "EQIX", "PLTR", "ABNB", "PYPL", "SNPS", "SPOT"]
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# Feature Engineering: Moving average, daily return, and target shift
for ticker in tickers:
    try:
        # Access the 'Adj Close' price
        if ('Adj Close', ticker) in data.columns:
            # Calculate a 50-day moving average for each ticker's 'Adj Close' price
            data[('50_MA', ticker)] = data[('Adj Close', ticker)].rolling(window=50).mean()

            # Calculate the daily return for each ticker
            data[('Daily_Return', ticker)] = data[('Adj Close', ticker)].pct_change()

            # Shift the 'Adj Close' price to create the target (next day's close)
            data[('Target', ticker)] = data[('Adj Close', ticker)].shift(-1)
        else:
            print(f"Ticker '{ticker}' not found in the dataset.")
    except KeyError as e:
        print(f"Error processing {ticker}: {e}")

# Drop any remaining NaN values after feature engineering
data.dropna(inplace=True)

# Split the data into train/test sets (80% train, 20% test)
split_ratio = 0.8
train_size = int(len(data) * split_ratio)
train_data = data[:train_size]
test_data = data[train_size:]

# Output first few rows of the final preprocessed dataset
print(data.head())

# Save the preprocessed data for later use
data.to_csv("preprocessed_stock_data.csv")