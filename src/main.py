import yfinance as yf
data = yf.download('AAPL', start='2015-01-01', end='2023-01-01')
print(data.head())
