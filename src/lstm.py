import yfinance as yf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import timedelta


data = yf.download("MSFT", start="2010-01-01", end="2023-01-01")

data = data[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(data)

def create_sequences(data, time_step=60):
    X = []
    y = []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60  # Look back 60 days
X, y = create_sequences(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential()

#layer 2
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))

#layer 1
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# dense layer
model.add(Dense(units=25))
model.add(Dropout(0.2))

#output layer
model.add(Dense(units=1))  # Predicting a single future stock price

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=10, batch_size=64)

test_data = scaled_data[-time_step:]
test_data = np.array(test_data).reshape(1, -1, 1)


predicted_price = model.predict(test_data)

predicted_price_original_scale = scaler.inverse_transform(predicted_price)

last_date = data.index[-1]

next_date = last_date + timedelta(days=1)

print(f"Predicted future stock price for {next_date.date()}: {predicted_price_original_scale[0][0]}")