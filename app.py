import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load your stock CSV
df = pd.read_csv("EW-MAX.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

st.title("Stock Price Prediction & Portfolio Simulation")

# Rename the stock display name (optional)
stock_name = "EW-MAX (Example Stock)"
st.markdown(f"### Selected Stock: **{stock_name}**")

# Investment slider
investment = st.slider("💰 Portfolio Investment (INR)", 1000, 100000, 10000, step=1000)

df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle divide-by-zero and NaN safely
    rsi = rsi.fillna(0)
    rsi = np.clip(rsi, 0, 100)

    return rsi

df['RSI'] = compute_rsi(df['Close'])
df.dropna(inplace=True)

# Display recent data
st.subheader("Latest Data")
st.write(df.tail())

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Create LSTM sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i, df.columns.get_loc('Close')])
    return np.array(X), np.array(y)

seq_len = 60
X, y = create_sequences(scaled_data, seq_len)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Button to train and simulate
if st.button("Predict & Simulate Portfolio"):
    with st.spinner("Training LSTM model..."):
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

        # Predictions
        predicted = model.predict(X_test)
        actual = y_test

        # Inverse transform
        pred_full = np.zeros((len(predicted), df.shape[1]))
        act_full = np.zeros((len(actual), df.shape[1]))
        pred_full[:, df.columns.get_loc('Close')] = predicted.reshape(-1)
        act_full[:, df.columns.get_loc('Close')] = actual.reshape(-1)
        pred_price = scaler.inverse_transform(pred_full)[:, df.columns.get_loc('Close')]
        act_price = scaler.inverse_transform(act_full)[:, df.columns.get_loc('Close')]

        # Portfolio simulation
        latest_actual = act_price[-1]
        latest_predicted = pred_price[-1]
        shares = investment / latest_actual
        future_value = shares * latest_predicted
        gain = future_value - investment

        # Show results
        st.subheader("Price Prediction")
        st.line_chart(pd.DataFrame({
            "Actual": act_price,
            "Predicted": pred_price
        }))

        st.subheader("💼 Portfolio Summary")
        st.write(f"**Initial Investment:** ₹{investment}")
        st.write(f"**Predicted Future Value:** ₹{future_value:.2f}")
        st.write(f"**Estimated Gain:** ₹{gain:.2f} ({(gain/investment)*100:.2f}%)")

# Show technical indicators
st.subheader("Technical Indicators")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['Close'], label='Close Price')
ax.plot(df['SMA_10'], label='SMA 10')
ax.plot(df['SMA_50'], label='SMA 50')
ax.set_title("Moving Averages")
ax.legend()
st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(10, 2))
ax2.plot(df['RSI'], label='RSI', color='purple')
ax2.axhline(70, color='red', linestyle='--', label='Overbought')
ax2.axhline(30, color='green', linestyle='--', label='Oversold')
ax2.legend()
ax2.set_title("RSI Indicator")
st.pyplot(fig2)
