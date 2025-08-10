# Stock Price Prediction using LSTM
# Name: A. Jayavanth

import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import streamlit as st
import plotly.graph_objects as go

# SETTINGS
model_path = "best_stock_model.keras"
seq_len = 120

# UI setup
st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4CAF50;'>📈 Stock Price Prediction using LSTM</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#A9A9A9;'>By A. Jayavanth</h4>", unsafe_allow_html=True)
# Input
stock = st.text_input('Enter Stock Symbol:', 'WMT')
start = '2012-01-01'
end = '2022-12-31'

# Load data
data = yf.download(stock, start, end)
if data.empty:
    st.error("🚫 No data fetched.")
    st.stop()

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

# Prepare test data
train_size = int(len(scaled_data) * 0.80)
test_data = scaled_data[train_size - seq_len:]

def create_sequences(dataset, seq_length=seq_len):
    X, y = [], []
    for i in range(seq_length, len(dataset)):
        X.append(dataset[i - seq_length:i])
        y.append(dataset[i, 3])
    return np.array(X), np.array(y)

X_test, y_test = create_sequences(test_data)

# Load model
model = load_model(model_path)

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.concatenate(
    (np.zeros((predictions.shape[0], 3)), predictions, np.zeros((predictions.shape[0], 1))), axis=1))[:, 3]
y_test_actual = scaler.inverse_transform(np.concatenate(
    (np.zeros((y_test.shape[0], 3)), y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 1))), axis=1))[:, 3]

# Metrics
mape = mean_absolute_percentage_error(y_test_actual, predictions) * 100
accuracy = 100 - mape
r_squared = r2_score(y_test_actual, predictions)

col1, col2, col3 = st.columns(3)
col1.metric("📊 Accuracy", f"{accuracy:.2f}%")
col2.metric("📉 MAPE", f"{mape:.2f}%")
col3.metric("📈 R² Score", f"{r_squared:.4f}")

# Plotly chart
fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test_actual, mode='lines', name='Actual Price', line=dict(color='green')))
fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted Price', line=dict(color='red')))
fig.update_layout(title='Stock Price Prediction', xaxis_title='Time', yaxis_title='Price', template='plotly_dark')
st.plotly_chart(fig, use_container_width=True)
