import yfinance as yf
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

data = yf.download("MSFT", start="2010-01-01", end="2023-01-01")

for lag in [1, 2, 3]:  # You can add more lag values
    data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
    data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

data = data.dropna()

data = data[['Close', 'Volume','Adj Close', 'Close_Lag_1', 'Volume_Lag_1',
             'Close_Lag_2', 'Volume_Lag_2',
             'Close_Lag_3', 'Volume_Lag_3']]

close_columns = ['Close', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3','Adj Close']
volume_columns = ['Volume', 'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3']

train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

close_scaler = MinMaxScaler()
volume_scaler = MinMaxScaler()

scaled_close_train = close_scaler.fit_transform(train_data[close_columns])
scaled_volume_train = volume_scaler.fit_transform(train_data[volume_columns])

scaled_close_test = close_scaler.transform(test_data[close_columns])
scaled_volume_test = volume_scaler.transform(test_data[volume_columns])

scaled_close_train_df = pd.DataFrame(scaled_close_train, columns=close_columns)
scaled_volume_train_df = pd.DataFrame(scaled_volume_train, columns=volume_columns)

scaled_close_test_df = pd.DataFrame(scaled_close_test, columns=close_columns)
scaled_volume_test_df = pd.DataFrame(scaled_volume_test, columns=volume_columns)

scaled_train_data = pd.concat([scaled_close_train_df, scaled_volume_train_df], axis=1)
scaled_test_data = pd.concat([scaled_close_test_df, scaled_volume_test_df], axis=1)

def create_sequences(data, time_step=60):
    X = []
    y = []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

train_data_np = scaled_train_data.values
test_data_np = scaled_test_data.values

time_step = 60

X_train, y_train = create_sequences(train_data_np, time_step)
X_test, y_test = create_sequences(test_data_np, time_step)

tf.random.set_seed(40)

model = Sequential()

model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(units=50))
model.add(Dropout(0.3))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=20, batch_size=64)

predictions = model.predict(X_test)

predicted_prices = close_scaler.inverse_transform(
    np.concatenate([predictions, np.zeros((predictions.shape[0], len(close_columns) - 1))], axis=1)
)[:, 0]

actual_prices = close_scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(close_columns) - 1))], axis=1)
)[:, 0]

mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

plt.figure(figsize=(10, 6))
plt.plot(actual_prices, color='blue', label='Actual Stock Price')
plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
