# Training file for Stock Price Prediction using LSTM with tuning
# A. Jayavanth

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt

# --------------- SETTINGS ----------------
stock = 'WMT'  # Default stock symbol
start = '2012-01-01'
end = '2022-12-31'
seq_len = 120  # number of past days to look at
model_path = "best_stock_model.keras"

# --------------- GET STOCK DATA ---------------
data = yf.download(stock, start, end)
if data.empty:
    print("No data found, check stock symbol or internet.")
    exit()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

train_size = int(len(scaled_data) * 0.80)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - seq_len:]

# create sequences function
def create_sequences(dataset, seq_length=seq_len):
    X, y = [], []
    for i in range(seq_length, len(dataset)):
        X.append(dataset[i - seq_length:i])
        y.append(dataset[i, 3])  # Close price index
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# --------------- MODEL BUILD FUNCTION ---------------
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('lstm_units1', min_value=50, max_value=200, step=50),
                   return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('lstm_units2', min_value=32, max_value=128, step=32),
                   return_sequences=False))
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [0.001, 0.0005])),
        loss='mean_squared_error'
    )
    return model

# --------------- HYPERPARAMETER TUNING ---------------
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=3,  # small for quick testing
    executions_per_trial=1,
    directory='tuner_results',
    project_name='stock_lstm'
)

stop_early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(X_train, y_train, epochs=10, batch_size=32,
             validation_data=(X_test, y_test), callbacks=[stop_early])

# get best model
best_model = tuner.get_best_models(num_models=1)[0]

# save the model
best_model.save(model_path)
print(f"Best model saved at {model_path}")
