import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

def create_sequences(data, time_steps=60):
    sequences = []
    labels = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
        labels.append(data[i + time_steps])
    return np.array(sequences), np.array(labels)

ticker = "MSFT"
stock_data = yf.download(ticker, start="2020-01-01", end="2023-01-01", interval="1d")

data = stock_data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

time_step = 60
X, y = create_sequences(scaled_data, time_step)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential()

model.add(SimpleRNN(units=50, return_sequences=False,input_shape=(time_step,1)))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=20, batch_size=32)

predicted_prices = model.predict(X_test)

# Inverse scaling the predicted prices to original scale
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the results

plt.plot(stock_data.index[split+time_step:], data[split+time_step:], color='blue', label='Actual Stock Price')
plt.plot(stock_data.index[split+time_step:], predicted_prices, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()