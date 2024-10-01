from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSM", "AVGO", "TSLA", "TCEHY", "ORCL", "ASML", "SSNLF", "NFLX", "SAP", "AMD", "CRM", "BABA", "ADBE", "CSCO", "QCOM", "TXN", "NOW", "PDD", "INTU", "AMAT", "UBER", "SLA", "SU.PA", "BKNG", "ANET", "MU", "SONY", "ADI", "ADP", "PANW", "LRCX", "KLAC", "MELI", "SHOP", "FI", "INTC", "DELL", "EQIX", "PLTR", "ABNB", "PYPL", "SNPS", "SPOT"]
stock = input("Enter a stock: ")

indicator = False
while  indicator == False:
    for tick in tickers:
        if stock == tick:
            

data = pd.read_csv('preprocessed_long_format_stock_data.csv')

data.drop(columns=["Adj_Close","Close","Open","High","Low"],inplace=True)

lag_days = 1
lag_weeks = 7

data['Volume_Lag_1_day'] = data.groupby('Company')['Volume'].shift(lag_days)
data['Daily_Return_Lag_1_day'] = data.groupby('Company')['Daily_Return'].shift(lag_days)

data['Volume_Lag_1_week'] = data.groupby('Company')['Volume'].shift(lag_weeks)
data['Daily_Return_Lag_1_week'] = data.groupby('Company')['Daily_Return'].shift(lag_weeks)

data.dropna(inplace=True)

data.set_index(['Date', 'Company'], inplace=True)

X = data.drop(columns=['Target'])
y = data['Target']

train_size = int(0.8 * len(data))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
