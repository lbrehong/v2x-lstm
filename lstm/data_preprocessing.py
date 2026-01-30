import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm

from config import (
    MIN_LAT, MIN_LON,
    FEATURE_COLS, TARGET_COLS,
    create_all_scalers,
)


def compute_pdr_rolling(df, time_column, window_size=1, packet_interval_ms=20):
    """
    Computes rolling PDR based on a fixed packet transmission rate.

    Args:
        df: DataFrame containing the timestamps of received packets.
        time_column: Column name for timestamps.
        window_size: Time window (in seconds) for rolling PDR computation.
        packet_interval_ms: Fixed interval at which packets are sent (default: 20ms).

    Returns:
        DataFrame with PDR column added.
    """
    df["tx_timestamp_sec"] = df[time_column] / 1000
    expected_packets_per_window = window_size * (1000 / packet_interval_ms)
    timestamps = df["tx_timestamp_sec"].to_numpy(dtype=float)
    rolling_window = deque()
    pdr_values = np.zeros(len(df))

    for i, t in enumerate(timestamps):
        while rolling_window and rolling_window[0] < t - window_size:
            rolling_window.popleft()
        rolling_window.append(t)
        pdr_values[i] = len(rolling_window) / expected_packets_per_window
        if pdr_values[i] > 1.0:
            pdr_values[i] = 1.0

    new_column = pd.Series(pdr_values, index=df.index)
    try:
        df["pdr"] = new_column
    except ValueError:
        if len(df) != len(new_column):
            print("Length mismatch")
            raise
    return df


def generate_lstm_sequences(data, feature_cols, target_cols, seq_length, batch_size=1024):
    """
    Yields LSTM sequences in batches to prevent memory overload.

    Args:
        data: DataFrame with raw network data.
        feature_cols: List of feature column names.
        target_cols: List of target column names.
        seq_length: Number of past timesteps to use as input for LSTM.
        batch_size: Number of sequences to yield per batch (default: 1024).

    Yields:
        Tuple of (x_batch, y_batch) arrays.
    """
    num_samples = len(data) - seq_length

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating sequences", unit="batch"):
        x_batch, y_batch = [], []
        for j in range(i, min(i + batch_size, num_samples)):
            x_batch.append(data[feature_cols].iloc[j: j + seq_length].values)
            y_batch.append(data[target_cols].iloc[j + seq_length].values)

        yield np.array(x_batch), np.array(y_batch)


def preprocess_lstm_input(df, new, rat, target_cols, seq_length):
    """
    Prepares LSTM input by computing PDR, normalizing features, and creating time series sequences.

    Args:
        df: DataFrame with raw network data.
        new: Is this data for a new model?
        rat: RAT type ('5g', 'pc5' or 'dsrc').
        target_cols: List of target column names (e.g., ['latency_ms', 'pdr']).
        seq_length: Number of past timesteps to use as input for LSTM.

    Returns:
        Tuple of (X_sequences, y_sequences, scalers).
    """
    if rat not in FEATURE_COLS:
        raise ValueError(f"!!! Invalid RAT type: {rat}. Must be '5g', 'pc5' or 'dsrc'.")

    feature_cols = FEATURE_COLS[rat]
    scalers = create_all_scalers()
    gps_scaler = scalers['tx_latitude']

    # Select relevant features and targets
    df = df[feature_cols].dropna()
    # Filter out rows where GPS is at default minimum values
    df = df[~((df['tx_latitude'] == MIN_LAT) & (df['tx_longitude'] == MIN_LON))]

    # Normalize features
    print("___ Time to normalize data...")
    gps_flag = False
    for col in feature_cols:
        if col in ('tx_latitude', 'tx_longitude'):
            if not gps_flag:
                try:
                    df[['tx_latitude', 'tx_longitude']] = gps_scaler.transform(df[['tx_latitude', 'tx_longitude']])
                except ValueError:
                    print("!!! NaN GPS caught, skipping normalization")
                gps_flag = True
        else:
            df[col] = scalers[col].transform(df[[col]])

    # Convert to sequences for LSTM
    print("___ Time to convert into 3D data...")
    x_sequences_train, y_sequences_train = [], []
    if new:
        print(f"___ Training data... length {len(df) - seq_length}")
        for x_batch, y_batch in generate_lstm_sequences(df, feature_cols, target_cols, seq_length):
            x_sequences_train.append(x_batch)
            y_sequences_train.append(y_batch)
        x_sequences_train = np.concatenate(x_sequences_train, axis=0)
        y_sequences_train = np.concatenate(y_sequences_train, axis=0)

    print("___ Time to convert into numpy arrays...")
    return np.array(x_sequences_train), np.array(y_sequences_train), scalers


def inverse_transform(scalers, csv):
    """Inverse transform scaled values in a prediction log CSV."""
    log_df = pd.read_csv(csv, header=None,
                         names=["latitude", "longitude", "pred_latency", "actual_latency",
                                "rmse_latency", "pred_pdr", "actual_pdr", "rmse_pdr"])

    log_df[['latitude', 'longitude']] = scalers['tx_latitude'].inverse_transform(log_df[['latitude', 'longitude']])
    log_df['pred_latency'] = scalers['latency_ms'].inverse_transform(log_df[['pred_latency']])
    log_df['actual_latency'] = scalers['latency_ms'].inverse_transform(log_df[['actual_latency']])
    log_df['pred_pdr'] = scalers['pdr'].inverse_transform(log_df['pred_pdr'])
    log_df['actual_pdr'] = scalers['pdr'].inverse_transform(log_df['actual_pdr'])

    log_df.to_csv(csv + "_scaled", index=False)
