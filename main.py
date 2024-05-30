import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

st.title('STOCK PRICE PREDICTION WITH SENTIMENT ANALYSIS')

# Function to fetch news and calculate sentiment scores
def fetch_news_sentiment(ticker):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, 'html.parser')
    news_table = html.find(id='news-table')

    parsed_data = []
    last_date = None  # Initialize last_date to track the last valid date

    # Parse the news table for headlines and dates
    for row in news_table.findAll('tr'):
        if row.a:
            title = row.a.get_text()
            date_data = row.td.text.strip().split(' ')

            date, time = '', ''
            if len(date_data) == 1:
                time = date_data[0]
                news_date = last_date  # Use the last valid date
            else:
                date = date_data[0]
                time = date_data[1]
                # Handle 'Today' and 'Yesterday'
                if date == 'Today':
                    news_date = datetime.now().date()
                elif date == 'Yesterday':
                    news_date = datetime.now().date() - timedelta(days=1)
                else:
                    news_date = datetime.strptime(date, '%b-%d-%y').date()
                last_date = news_date  # Update the last valid date

            # Only include news from the last 3 days
            if news_date and news_date >= datetime.now().date() - timedelta(days=3):
                parsed_data.append([ticker, news_date, time, title])

    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

    vader = SentimentIntensityAnalyzer()
    f = lambda title: vader.polarity_scores(title)['compound']
    df['compound'] = df['title'].apply(f)

    return df

# Function to adjust the predicted price based on sentiment score
def adjust_price_with_sentiment(predicted_price, sentiment_df):
    if not sentiment_df.empty:
        avg_sentiment = sentiment_df['compound'].mean()
        adjustment_factor = 1 + (avg_sentiment * 0.01)  # Minor influence factor
        adjusted_price = predicted_price * adjustment_factor
        avg_influence = avg_sentiment * 0.01  # 1% influence factor
        influence_percentage = avg_influence * 100  # Convert to percentage
        return adjusted_price, avg_sentiment, influence_percentage
    return predicted_price, 0,0

# Streamlit app input
ticker = st.text_input('Select a stock ticker to predict its price:')

if ticker:
    try:
        # Download stock data from Yahoo Finance
        data = yf.download(ticker, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        if data.empty:
            st.error(f"No data found for ticker {ticker}. Please enter a valid ticker.")
        else:
            data.reset_index(inplace=True)
            data['Date'] = pd.to_datetime(data['Date'])

            # Debugging: Check the date range
            st.write(f"Data range: {data['Date'].min()} to {data['Date'].max()}")
            
            st.write()

            st.write(f"Model trained on data from 2010-01-01 to 2023-01-1 and now prediction showing from 2023 to till today ")

            # Normalize the 'Close' prices
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            # Prediction from 2023 to today
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
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Predict
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)

                # Align predictions with actual data
                prediction_dates = data['Date'][prediction_start_index:]

                # Filter actual data from the prediction start date
                actual_data_from_2023 = data[data['Date'] >= prediction_start_date]

                # Plot actual vs predicted
                st.subheader(f'{ticker} Stock Price Prediction')
                fig, ax = plt.subplots(figsize=(12,6))
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
                predicted_price = scaler.inverse_transform(predicted_price)[0][0]

                # Fetch news sentiment data
                sentiment_df = fetch_news_sentiment(ticker)

                # Adjust predicted price based on sentiment
                adjusted_price, avg_sentiment, influence_percentage_ = adjust_price_with_sentiment(predicted_price, sentiment_df)
                st.write("")
                st.subheader(f'Tomorrow\'s predicted price for {ticker}: ${adjusted_price:.2f}')
                #st.write(sentiment_df)  # Display sentiment data for debugging

                # Model evaluation
                evaluation = model.evaluate(x_test, scaled_data[prediction_start_index:], verbose=0)
                st.write("")
                st.subheader('Model Evaluation')
                st.write(f'Model Loss: {evaluation:.4f}')

                if avg_sentiment != 0:
                    st.subheader('Sentiment Analysis Impact')
                    st.write(f"Sentiment score influenced the prediction by an average of {influence_percentage_:.4f}%")
                else:
                    st.subheader('Sentiment Analysis Impact')
                    st.write('No recent news headlines found. Prediction is based on Stacked LSTM model with stock price data only.')
    except Exception as e:
        st.error(f"An error occurred: {e}")
