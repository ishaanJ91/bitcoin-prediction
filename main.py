import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# CoinRanking API details
api_url = "https://api.coinranking.com/v2"
headers = {
    'x-access-token': 'coinranking92a1fe6775ffcff15a80311d0fd8d2c2d1da46d6caf4cd04',
}

# Clean data by dropping NaN values
def clean_data(df):
    return df.dropna()

# Fetch historical data
def fetch_crypto_history(coin_id, time_period):
    endpoint = f"/coin/{coin_id}/history?timePeriod={time_period}"
    url = api_url + endpoint
    response = requests.get(url, headers=headers)
    data = response.json()

    prices = [item['price'] for item in data['data']['history']]
    timestamps = [item['timestamp'] for item in data['data']['history']]

    dates = pd.to_datetime(timestamps, unit='s')
    df = pd.DataFrame({"Date": dates, "Close": prices})
    df.set_index("Date", inplace=True)
    df = df.astype(float)
    
    return clean_data(df)

# Generate sample weights that emphasize recent data
def generate_sample_weights(data_length, emphasis_factor=1.5):
    return np.linspace(1, emphasis_factor, data_length)

# Prepare data for training
def prepare_data(data, scaler, prediction_days=60):
    scaled_data = scaler.transform(data[['Close']])

    x_data, y_data = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_data.append(scaled_data[x - prediction_days:x])
        y_data.append(scaled_data[x, 0])

    x_data_np = np.array(x_data).reshape(len(x_data), prediction_days, 1)
    return x_data_np, np.array(y_data)

# Custom loss function with sample weights
def weighted_mse(y_true, y_pred, sample_weights):
    sample_weights = K.constant(sample_weights)
    mse = K.square(y_true - y_pred)
    return K.mean(mse * sample_weights, axis=-1)

# Coin ID for Bitcoin
coin_id = 'Qwsogvtv82FCd'
time_period = '5y'

# Fetch historical price data
data = fetch_crypto_history(coin_id, time_period)

# Split the data into training (60%) and testing (40%)
split_ratio = 0.6
split_index = int(len(data) * split_ratio)

train_data = data[:split_index]
test_data = data[split_index:]

# Scaling only the 'Close' price
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data[['Close']])

# Prepare training and testing data
x_train, y_train = prepare_data(train_data, scaler)
x_test, y_test = prepare_data(test_data, scaler)

# Generate sample weights for training data
sample_weights = generate_sample_weights(len(y_train))

# CNN + LSTM Model
model_cnn_lstm = Sequential()
model_cnn_lstm.add(LSTM(units=50, return_sequences=False, activation='relu', input_shape=(x_train.shape[1], 1), kernel_regularizer=l2(0.001)))
model_cnn_lstm.add(Dropout(0.3))
model_cnn_lstm.add(BatchNormalization())
model_cnn_lstm.add(Dense(units=50, kernel_regularizer=l2(0.001)))
model_cnn_lstm.add(Dropout(0.3))
model_cnn_lstm.add(Dense(units=10, activation='relu', kernel_regularizer=l2(0.001)))
model_cnn_lstm.add(Dropout(0.3))
model_cnn_lstm.add(Dense(units=1))

# Compile and train the model with dynamic learning rate
learning_rate = 0.01
optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
model_cnn_lstm.compile(optimizer=optimizer, loss=lambda y_true, y_pred: weighted_mse(y_true, y_pred, sample_weights))
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_cnn_lstm.fit(x_train, y_train, epochs=150, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Random Forest Model
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
x_train_flattened = x_train.reshape(x_train.shape[0], -1)
rf.fit(x_train_flattened, y_train)

# Predict on the test set
x_test_flattened = x_test.reshape(x_test.shape[0], -1)
y_pred_rf = rf.predict(x_test_flattened)
y_pred_cnn_lstm = model_cnn_lstm.predict(x_test)

# Calculate metrics for CNN + LSTM
mae_test_cnn_lstm = mean_absolute_error(y_test, y_pred_cnn_lstm)
mse_test_cnn_lstm = mean_squared_error(y_test, y_pred_cnn_lstm)
mape_test_cnn_lstm = mean_absolute_percentage_error(y_test, y_pred_cnn_lstm)

print(f"Test set Mean Absolute Error (MAE) - CNN + LSTM: {mae_test_cnn_lstm}")
print(f"Test set Mean Squared Error (MSE) - CNN + LSTM: {mse_test_cnn_lstm}")
print(f"Test set Mean Absolute Percentage Error (MAPE) - CNN + LSTM: {mape_test_cnn_lstm}")
accuracy = 100 - mape_test_cnn_lstm
print(f"Accuracy of the CNN + LSTM model: {accuracy}%")

# Plot the real and predicted data for both models
plt.figure(figsize=(10, 5))
plt.plot(y_test, color='blue', label='Real Prices')
plt.plot(y_pred_rf, color='red', label='RF Predictions')
plt.plot(y_pred_cnn_lstm, color='green', label='CNN + LSTM Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title("Bitcoin Price Prediction (Random Forest and CNN + LSTM)")
plt.legend()
plt.savefig("rf_cnn_lstm_price_prediction.png")
plt.show()