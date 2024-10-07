import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

ticker = "MSFT"
stock_data = yf.download(ticker, start="2020-01-01", end="2023-01-01", interval="1d")

data = stock_data['Close'].values.reshape(-1, 1)