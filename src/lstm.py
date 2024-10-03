import yfinance as yf
import pandas as pd

ticker = "MSFT"
data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

format_data = pd.DataFrame()
print(data)