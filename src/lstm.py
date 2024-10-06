import yfinance as yf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Download MSFT stock data
data = yf.download("MSFT", start="2010-01-01", end="2023-01-01")

# Create lag features for 'Close' and 'Volume'
for lag in [1, 2, 3]:  # You can add more lag values
    data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
    data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)

# Drop rows with NaN values due to lagging
data = data.dropna()

# Select relevant features
data = data[['Close', 'Volume', 'Close_Lag_1', 'Volume_Lag_1',
             'Close_Lag_2', 'Volume_Lag_2',
             'Close_Lag_3', 'Volume_Lag_3']]

# Initialize a single scaler for all features
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the entire dataset
scaled_data = scaler.fit_transform(data)

# Split data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Function to create sequences of data
def create_sequences(data, time_step=60):
    X = []
    y = []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, :])  # The last `time_step` values
        y.append(data[i, 0])  # The target is the 'Close' price (first column)
    return np.array(X), np.array(y)

# Set the time step (sequence length)
time_step = 60

# Create training and testing sequences
X_train, y_train = create_sequences(train_data, time_step)
X_test, y_test = create_sequences(test_data, time_step)

# Build the LSTM model
model = Sequential()

# First LSTM layer with Dropout
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))

# Second LSTM layer with Dropout
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))

# Dense layers
model.add(Dense(units=50))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions and actual values
# Use the scaler to inverse transform only the first column (Close prices)
predicted_prices = scaler.inverse_transform(np.concatenate([predictions, np.zeros((predictions.shape[0], scaled_data.shape[1]-1))], axis=1))[:, 0]
actual_prices = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1]-1))], axis=1))[:, 0]

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, color='blue', label='Actual Stock Price')
plt.plot(predicted_prices, color='red', label='Predicted Stock Price')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
