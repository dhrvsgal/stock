from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

# Load the data
data = pd.read_excel(r'C:\Users\User\Desktop\model\AAPL23.xlsx')

# Let's say 'Close' column represents the closing stock prices
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Prepare the data for LSTM
def prepare_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 50
X, y = prepare_data(scaled_data, window_size)

# Reshape data for LSTM [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train LSTM model
lstm_units = 50
epochs = 5

model = Sequential()
model.add(LSTM(units=lstm_units, return_sequences=True))
model.add(LSTM(units=lstm_units))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        days = int(request.form['days'])
        if days <= 0:
            return render_template('index.html', error='Please enter a valid number of days.')

        # Make predictions
        last_window = scaled_data[-window_size:]
        predictions = []
        for _ in range(days):
            X_pred = last_window[-window_size:].reshape((1, window_size, 1))
            pred = model.predict(X_pred)
            predictions.append(pred[0][0])
            last_window = np.append(last_window, pred[0][0])
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        return render_template('index.html', predictions=predictions)
    except:
        return render_template('index.html', error='An error occurred. Please try again.')

if __name__ == '__main__':
    app.run(debug=True)
