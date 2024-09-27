import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSM", "AVGO", "TSLA", "TCEHY", "ORCL", "ASML", "SSNLF", "NFLX", "SAP", "AMD", "CRM", "BABA", "ADBE", "CSCO", "QCOM", "TXN", "NOW", "PDD", "INTU", "AMAT", "UBER", "SLA", "SU.PA", "BKNG", "ANET", "MU", "SONY", "ADI", "ADP", "PANW", "LRCX", "KLAC", "MELI", "SHOP", "FI", "INTC", "DELL", "EQIX", "PLTR", "ABNB", "PYPL", "SNPS", "SPOT"]
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

long_format_data = pd.DataFrame()

for ticker in tickers:
    try:
        if ('Adj Close', ticker) in data.columns:
            ticker_df = pd.DataFrame({
                'Date': data.index,
                'Company': ticker,
                'Adj_Close': data[('Adj Close', ticker)],
                'Close': data[('Close', ticker)],
                'Open': data[('Open', ticker)],
                'High': data[('High', ticker)],
                'Low': data[('Low', ticker)],
                'Volume': data[('Volume', ticker)]
            })
            ticker_df['50_MA'] = ticker_df['Adj_Close'].rolling(window=50).mean()  # 50-day moving average
            ticker_df['Daily_Return'] = ticker_df['Adj_Close'].pct_change()  # Daily return
            ticker_df['Target'] = ticker_df['Adj_Close'].shift(-1)  # Next day's price as target

            long_format_data = pd.concat([long_format_data,ticker_df],ignore_index=True)
        else:
            print(f"Ticker '{ticker}' not found in the dataset.")
    except KeyError as e:
        print(f"Error processing {ticker}: {e}")

# Drop any remaining NaN values after feature engineering
long_format_data.dropna(inplace=True)

# Optional: Sort by date for time-series modeling
long_format_data.sort_values(by='Date', inplace=True)

# Output first few rows of the final preprocessed dataset
print(long_format_data.head())

# Save the long format data for later use
long_format_data.to_csv("preprocessed_long_format_stock_data.csv", index=False)

# Split the data into train/test sets (80% train, 20% test)
split_ratio = 0.8
train_size = int(len(long_format_data) * split_ratio)
train_data = long_format_data[:train_size]
test_data = long_format_data[train_size:]