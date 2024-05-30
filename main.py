import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

st.title('STOCK PRICE PREDICTION')

# Function to download and prepare data for a list of tickers
ticker = st.text_input('Select a stock ticker to predict its price:')

if ticker:
    try:
        data = yf.download(ticker, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        if data.empty:
            st.error(f"No data found for ticker {ticker}. Please enter a valid ticker.")
        else:
            data.reset_index(inplace=True)

            # Ensure date column is in datetime format
            data['Date'] = pd.to_datetime(data['Date'])

            # Debugging: Check the date range
            st.write(f"Data range: {data['Date'].min()} to {data['Date'].max()}")

            # Preprocessing
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            # Test data for prediction from 2023 to today
            prediction_start_date = datetime(2023, 1, 1)
            if not any(data['Date'] >= prediction_start_date):
                st.error(f"No data found from {prediction_start_date.strftime('%Y-%m-%d')} onwards for ticker {ticker}.")
            else:
                prediction_start_index = data[data['Date'] >= prediction_start_date].index[0]
                test_data = scaled_data[prediction_start_index - 30:]

                x_test = []
                for x in range(30, len(test_data)):
                    x_test.append(test_data[x-30:x])

                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                model = load_model('/workspace/stock_price_prediction/universal_lstm_model.h5')

                # Predict
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)

                # Align predictions with actual data
                prediction_dates = data['Date'][prediction_start_index:]

                # Filter actual data from the prediction start date
                actual_data_from_2023 = data[data['Date'] >= prediction_start_date]

                # Plot actual vs predicted
                st.subheader(f'{ticker} Stock Price Prediction')
                fig, ax = plt.subplots()
                ax.plot(actual_data_from_2023['Date'], actual_data_from_2023['Close'], color='blue', label='Actual Price')
                ax.plot(prediction_dates, predictions, color='red', label='Predicted Price')
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
    except Exception as e:
        st.error(f"An error occurred: {e}")
