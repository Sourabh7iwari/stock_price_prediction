import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# Set the title of the Streamlit app
st.title('STOCK PRICE PREDICTION')

# Input for the user to enter the stock ticker symbom
ticker = st.text_input('Select a stock ticker to predict its price:')

# Only proceed if the user has entered a ticker symbol
if ticker:
    try:
        # Download stock data from Yahoo Finance starting from 2010 to the current data
        data = yf.download(ticker, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        
        # check if data was retrived successfully
        if data.empty:
            st.error(f"No data found for ticker {ticker}. Please enter a valid ticker.")
        else:
            # Reset index to ensure 'Date' becomes a column
            data.reset_index(inplace=True)

            # Ensure the 'Date' Column is in datetime formate for correct handling
            data['Date'] = pd.to_datetime(data['Date'])

            # Debugging: Display the range of dates in the data
            st.write(f"Data range: {data['Date'].min()} to {data['Date'].max()}")

            # Normalize the 'Close' prices to the range [0,1] for the LSTM model
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            # Define the start date for predictions
            prediction_start_date = datetime(2023, 1, 1)

            # Check if there are data points available from the start date onwards
            if not any(data['Date'] >= prediction_start_date):
                st.error(f"No data found from {prediction_start_date.strftime('%Y-%m-%d')} onwards for ticker {ticker}.")
            else:
                # Get the index of the first date from 2023-01-01
                prediction_start_index = data[data['Date'] >= prediction_start_date].index[0]
                
                # Prepare test data for prediction, ensuring there are enough historical data points
                test_data = scaled_data[prediction_start_index - 30:]

                # Create sequences of 30 days of historical data for the LSTM model
                x_test = []
                for x in range(30, len(test_data)):
                    x_test.append(test_data[x-30:x])

                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                #Load the pre-trained LSTM model
                model = load_model('/workspace/stock_price_prediction/universal_lstm_model.h5')

                # Predict the stock price
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)

                # Align predictions with actual data
                prediction_dates = data['Date'][prediction_start_index:]

                # Filter actual data from the prediction start date
                actual_data_from_2023 = data[data['Date'] >= prediction_start_date]

                # Plot actual vs predicted prices from january 2023 onwards
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

                # Display the predicted price for the next day
                st.subheader(f'Tomorrow\'s predicted price for {ticker}: ${predicted_price[0][0]:.2f}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
