import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

st.title('STOCK PRICE PREDICTION')
# Function to download and prepare data for a list of tickers

ticker = st.selectbox('Select a stock ticker:', ['RS', 'AMZN', 'TSLA', 'HDB', 'AMD', 'NVDA'])


data = yf.download(ticker, start='2010-01-01', end=datetime.now())
data.reset_index(inplace=True)

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Test data for prediction
test_data = scaled_data[-60:]
x_test = []
for x in range(30, len(test_data)):
    x_test.append(test_data[x-30:x])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


model = load_model('/workspace/stock_price_prediction/universal_lstm_model.h5')

# Predict
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot actual vs predicted
st.subheader(f'{ticker} Stock Price Prediction')
fig, ax = plt.subplots()
ax.plot(data['Date'][-30:], data['Close'][-30:], color='blue', label='Actual Price')
ax.plot(data['Date'][-30:], predictions, color='red', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Predict tomorrow's price
last_30_days = scaled_data[-30:]
last_30_days = np.reshape(last_30_days, (1, last_30_days.shape[0], 1))
predicted_price = model.predict(last_30_days)
predicted_price = scaler.inverse_transform(predicted_price)

st.subheader(f'Tomorrow\'s predicted price for {ticker}: ${predicted_price[0][0]:.2f}')
