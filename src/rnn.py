import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_sequences(data, time_steps=60):
    sequences = []
    labels = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i:i + time_steps])
        labels.append(data[i + time_steps, 0])
    return np.array(sequences), np.array(labels)

ticker = "MSFT"
stock_data = yf.download(ticker, start="2016-01-01", end="2024-01-01", interval="1d")

close_data = stock_data[['Close']].values
volume_data = stock_data[['Volume']].values

# Initialize separate scalers
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_volume = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the Close price and Volume data
scaled_close = scaler_close.fit_transform(close_data)
scaled_volume = scaler_volume.fit_transform(volume_data)

# Combine the scaled features into a single dataset
scaled_data = np.hstack((scaled_close, scaled_volume))

time_step = 60
X, y = create_sequences(scaled_data, time_step)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

model = Sequential()

model.add(SimpleRNN(units=100, return_sequences=False, input_shape=(time_step, 2)))
model.add(Dropout(0.2))
model.add(Dense(units=25, activation='relu'))  # First additional Dense layer
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32)

predicted_prices = model.predict(X_test)

predicted_prices_scaled = np.concatenate([predicted_prices, np.zeros((predicted_prices.shape[0], 1))], axis=1)
predicted_prices = scaler_close.inverse_transform(predicted_prices_scaled)[:, 0]  # Only take the Close price

y_test_scaled = np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1))], axis=1)
actual_prices = scaler_close.inverse_transform(y_test_scaled)[:, 0]

mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)
print("Mean Squared Error: ", mse)
print("Mean Absolute Error: ", mae)

plt.plot(stock_data.index[split+time_step:], close_data[split+time_step:, 0], color='blue', label='Actual Stock Price')
plt.plot(stock_data.index[split+time_step:], predicted_prices, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction with Volume Feature')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
