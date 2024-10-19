import yfinance as yf
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

ticker = "AAPL"  # Example: Apple stock
start_date = "2010-01-01"
end_date = "2023-10-01"

data = yf.download(ticker, start=start_date, end=end_date)
missing_values = data.isnull().sum()
print(missing_values)