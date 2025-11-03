import pytest
import pandas as pd
import numpy as np
from app import compute_rsi, create_sequences
from sklearn.preprocessing import MinMaxScaler

# ---------- LOAD DATA ----------
@pytest.fixture(scope="module")
def df():
    df = pd.read_csv("EW-MAX.csv")
    assert not df.empty, "❌ CSV file is empty."
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# ---------- TEST 1: DATA VALIDATION ----------
def test_data_validity(df):
    assert 'Close' in df.columns, "❌ 'Close' column missing in dataset."
    assert df['Close'].notnull().all(), "❌ Missing values in 'Close' column."
    assert df.index.is_monotonic_increasing, "❌ Date index is not sorted properly."

# ---------- TEST 2: RSI COMPUTATION ----------
def test_rsi_computation(df):
    df['RSI'] = compute_rsi(df['Close'])
    assert df['RSI'].between(0, 100).all(), "❌ RSI out of expected 0–100 range."

# ---------- TEST 3: SEQUENCE CREATION ----------
def test_lstm_sequences(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    X, y = create_sequences(scaled_data, 60)
    assert len(X) > 0 and len(y) > 0, "❌ Sequence generation failed."
    assert X.shape[1] == 60, "❌ Sequence length mismatch (should be 60)."

# ---------- TEST 4: MOVING AVERAGES ----------
def test_sma_computation(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    assert df['SMA_10'].iloc[-1] is not np.nan, "❌ SMA_10 not computed properly."
    assert df['SMA_50'].iloc[-1] is not np.nan, "❌ SMA_50 not computed properly."

# ---------- TEST 5: DATA INTEGRITY ----------
def test_data_integrity(df):
    assert len(df) > 100, "❌ Insufficient data rows (<100)."
    assert df['Close'].dtype in [float, np.float64, int, np.int64], "❌ Invalid data type for Close column."
