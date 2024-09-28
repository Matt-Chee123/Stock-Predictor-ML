import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSM", "AVGO", "TSLA", "TCEHY", "ORCL", "ASML", "SSNLF", "NFLX", "SAP", "AMD", "CRM", "BABA", "ADBE", "CSCO", "QCOM", "TXN", "NOW", "PDD", "INTU", "AMAT", "UBER", "SLA", "SU.PA", "BKNG", "ANET", "MU", "SONY", "ADI", "ADP", "PANW", "LRCX", "KLAC", "MELI", "SHOP", "FI", "INTC", "DELL", "EQIX", "PLTR", "ABNB", "PYPL", "SNPS", "SPOT"]
data = pd.read_csv('preprocessed_long_format_stock_data.csv')
data.set_index(['Date', 'Company'], inplace=True)

performance_results = {}
for ticker in tickers:
    if ticker != 'SSNLF':
        ticker_data = data.xs(ticker,level="Company")
        X = ticker_data[['50_MA', 'Daily_Return', 'Volume']]
        Y = ticker_data['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate performance metric (Mean Squared Error)
        mse = mean_squared_error(y_test, y_pred)
        performance_results[ticker] = mse

mse_df = pd.DataFrame(list(performance_results.items()), columns=['Company', 'MSE'])

print(mse_df)

# Create a bar plot
sns.barplot(x='Company', y='MSE', data=mse_df)

# Customize the plot
plt.title('Mean Squared Error (MSE) of Linear Regression Models for Tech Companies')
plt.xlabel('Company Ticker')
plt.ylabel('Mean Squared Error')
plt.xticks(rotation=45)  # Rotate the x labels for better readability
plt.tight_layout()  # Adjust the layout to fit labels

# Show the plot
plt.show()