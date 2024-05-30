import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import joblib

# Function to download and prepare data for a list of tickers
def prepare_data(tickers, start_date, end_date, prediction_days):
    all_data = []
    combined_data = []

    # Download data for all tickers first to fit the scaler
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        combined_data.append(data['Close'].values.reshape(-1, 1))
    
    combined_data = np.concatenate(combined_data, axis=0)
    
    # Fit the scaler on the combined data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(combined_data)
    
    # Prepare the sequences for each ticker
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data['Close'].values
        data = data.reshape(-1, 1)
        scaled_data = scaler.transform(data)
        
        # Create sequences
        for x in range(prediction_days, len(scaled_data)):
            all_data.append((scaled_data[x-prediction_days:x], scaled_data[x]))
    
    # Split into input features (x) and labels (y)
    x_data, y_data = zip(*all_data)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    return x_data, y_data, scaler

# Define tickers, dates, and prediction days
tickers = ['RS', 'AMZN', 'TSLA', 'HDB', 'AMD', 'NVDA']
start_date = '2010-01-01'
end_date = '2023-01-01'
prediction_days = 30

# Prepare data
x_train, y_train, scaler = prepare_data(tickers, start_date, end_date, prediction_days)

# Reshape data for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Save the model
model.save('universal_lstm_model.h5')

# Function to predict stock price for a given ticker and plot actual vs. predicted prices
def predict_and_plot(ticker, train_start_date, predict_end_date, prediction_days, model, scaler):
    # Download data for prediction range
    data = yf.download(ticker, start=train_start_date, end=predict_end_date)
    data['Actual'] = data['Close']
    actual_prices = data['Close'].values
    
    # Scale the data
    scaled_data = scaler.transform(actual_prices.reshape(-1, 1))
    
    # Prepare data for prediction
    prediction_data = []
    for x in range(prediction_days, len(scaled_data)):
        prediction_data.append(scaled_data[x-prediction_days:x])
    
    prediction_data = np.array(prediction_data)
    prediction_data = np.reshape(prediction_data, (prediction_data.shape[0], prediction_data.shape[1], 1))
    
    # Predict prices
    predicted_prices = model.predict(prediction_data)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    
    # Get actual prices for the same range
    actual_prices = actual_prices[prediction_days:]
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color='blue', label='Actual Price')
    plt.plot(predicted_prices, color='red', label='Predicted Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return predicted_prices

# Define new dates for prediction
predict_start_date = '2023-01-01'
predict_end_date = datetime.now()

# Example prediction and plotting
predicted_prices = predict_and_plot('AAPL', start_date, predict_end_date, prediction_days, model, scaler)
print(f"Predicted price for AAPL on the next day: {predicted_prices[-1][0]}")
