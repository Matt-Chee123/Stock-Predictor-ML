import yfinance as yf
import pandas as pd
import tensorflow as tf
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

close_columns = ['Close', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']
volume_columns = ['Volume', 'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Lag_3']

close_data = data[close_columns]
volume_data = data[volume_columns]

# Create two separate scalers
close_scaler = MinMaxScaler()
volume_scaler = MinMaxScaler()

# Scale the data separately
scaled_close = close_scaler.fit_transform(close_data)
scaled_volume = volume_scaler.fit_transform(volume_data)

# Convert the scaled data back into DataFrames
scaled_close_df = pd.DataFrame(scaled_close, columns=close_columns)
scaled_volume_df = pd.DataFrame(scaled_volume, columns=volume_columns)

# Concatenate the two DataFrames back together
scaled_data = pd.concat([scaled_close_df, scaled_volume_df], axis=1)

# Now split the data into train and test sets
train_size = int(len(scaled_data) * 0.8)

train_data = scaled_data.iloc[:train_size]  # Use iloc to slice the DataFrame
test_data = scaled_data.iloc[train_size:]
print(train_data)
# Function to create sequences of data
def create_sequences(data, time_step=60):
    X = []
    y = []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, :])  # The last `time_step` values
        y.append(data[i, 0])  # The target is the 'Close' price (first column)
    return np.array(X), np.array(y)

# Convert DataFrames to NumPy arrays
train_data_np = train_data.values
test_data_np = test_data.values

# Set the time step (sequence length)
time_step = 60

# Create training and testing sequences
X_train, y_train = create_sequences(train_data_np, time_step)
X_test, y_test = create_sequences(test_data_np, time_step)
tf.random.set_seed(40)
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
predicted_prices = close_scaler.inverse_transform(
    np.concatenate([predictions, np.zeros((predictions.shape[0], len(close_columns) - 1))], axis=1)
)[:, 0]

# Do the same for the actual y_test values
actual_prices = close_scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], len(close_columns) - 1))], axis=1)
)[:, 0]
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
