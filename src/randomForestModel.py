from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSM", "AVGO", "TSLA", "TCEHY", "ORCL", "ASML", "SSNLF", "NFLX", "SAP", "AMD", "CRM", "BABA", "ADBE", "CSCO", "QCOM", "TXN", "NOW", "PDD", "INTU", "AMAT", "UBER", "SLA", "SU.PA", "BKNG", "ANET", "MU", "SONY", "ADI", "ADP", "PANW", "LRCX", "KLAC", "MELI", "SHOP", "FI", "INTC", "DELL", "EQIX", "PLTR", "ABNB", "PYPL", "SNPS", "SPOT"]
data = pd.read_csv('preprocessed_long_format_stock_data.csv')

data.drop(columns=["Adj_Close","Close","Open","High","Low"],inplace=True)

lag_days = 1
lag_weeks = 7

data['Volume_Lag_1_day'] = data.groupby('Company')['Volume'].shift(lag_days)
data['Daily_Return_Lag_1_day'] = data.groupby('Company')['Daily_Return'].shift(lag_days)

data['Volume_Lag_1_week'] = data.groupby('Company')['Volume'].shift(lag_weeks)
data['Daily_Return_Lag_1_week'] = data.groupby('Company')['Daily_Return'].shift(lag_weeks)

# After creating lag features, drop rows with NaN values (as they represent the first rows without lag data)
data.dropna(inplace=True)

data.set_index(['Date', 'Company'], inplace=True)

print(data.xs('AMD',level="Company"))
