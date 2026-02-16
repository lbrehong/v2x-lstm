"""
Data preprocessing utilities for RAT prediction models.

This module provides functions for:
- Computing rolling PDR from packet timestamps
- Normalizing features with predefined scalers
- Generating LSTM-compatible time series sequences
- Inverse transforming predictions back to original scale
"""
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
    Compute rolling Packet Delivery Rate (PDR) based on expected transmission rate.

    PDR is calculated as the ratio of received packets to expected packets
    within a sliding time window. This provides a localized reliability metric
    that accounts for varying network conditions over time.

    Algorithm:
        1. Convert timestamps to seconds
        2. Maintain a sliding window of received packet timestamps
        3. For each packet, count packets within the window
        4. Divide by expected packets based on transmission interval

    Args:
        df: DataFrame containing received packet data with timestamps
        time_column: Name of the timestamp column (in milliseconds)
        window_size: Size of rolling window in seconds (default: 1)
        packet_interval_ms: Expected interval between transmissions in ms (default: 20)

    Returns:
        DataFrame with 'pdr' column added (values in [0, 1])

    Note:
        PDR is clamped to 1.0 maximum to handle any timing irregularities
    """
    # Convert timestamps from milliseconds to seconds
    df["tx_timestamp_sec"] = df[time_column] / 1000

    # Calculate expected packets based on transmission rate
    expected_packets_per_window = window_size * (1000 / packet_interval_ms)

    timestamps = df["tx_timestamp_sec"].to_numpy(dtype=float)
    rolling_window = deque()  # Efficient FIFO for sliding window
    pdr_values = np.zeros(len(df))

    for i, t in enumerate(timestamps):
        # Remove packets outside the window
        while rolling_window and rolling_window[0] < t - window_size:
            rolling_window.popleft()

        # Add current packet to window
        rolling_window.append(t)

        # Compute PDR as ratio of received to expected packets
        pdr_values[i] = len(rolling_window) / expected_packets_per_window
        if pdr_values[i] > 1.0:
            pdr_values[i] = 1.0  # Cap at 100%

    # Add PDR column to DataFrame
    new_column = pd.Series(pdr_values, index=df.index)
    try:
        df["pdr"] = new_column
    except ValueError:
        if len(df) != len(new_column):
            print("Error: Length mismatch between DataFrame and PDR values")
            raise
    return df


def generate_lstm_sequences(data, feature_cols, target_cols, seq_length, batch_size=1024):
    """
    Generate LSTM-compatible sequences in memory-efficient batches.

    Creates sliding window sequences where each sample consists of
    seq_length timesteps of features, with the target being the values
    at the next timestep.

    Example (seq_length=3):
        Input:  [t0, t1, t2] -> Target: t3
        Input:  [t1, t2, t3] -> Target: t4

    Args:
        data: DataFrame with normalized network data
        feature_cols: List of input feature column names
        target_cols: List of target column names (typically ['latency_ms', 'pdr'])
        seq_length: Number of timesteps per input sequence
        batch_size: Number of sequences per batch for memory efficiency (default: 1024)

    Yields:
        Tuple of (x_batch, y_batch) numpy arrays where:
        - x_batch shape: (batch_size, seq_length, n_features)
        - y_batch shape: (batch_size, n_targets)
    """
    num_samples = len(data) - seq_length

    # Extract numpy arrays once for fast slicing (avoids DataFrame.iloc overhead per iteration)
    features = data[feature_cols].values
    targets = data[target_cols].values

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating sequences", unit="batch"):
        end = min(i + batch_size, num_samples)
        x_batch = np.array([features[j: j + seq_length] for j in range(i, end)])
        y_batch = targets[i + seq_length: end + seq_length]
        yield x_batch, y_batch


def preprocess_lstm_input(df, new, rat, target_cols, seq_length):
    """
    Full preprocessing pipeline for LSTM input preparation.

    Pipeline steps:
        1. Filter to RAT-specific feature columns
        2. Remove rows with missing values or invalid GPS
        3. Normalize all features using predefined scalers
        4. Generate sliding window sequences for LSTM

    Args:
        df: DataFrame with raw network data (must include PDR column)
        new: Flag indicating new model training (always True in current implementation)
        rat: RAT type identifier ('5g', 'pc5', or 'dsrc')
        target_cols: Target column names, typically ['latency_ms', 'pdr']
        seq_length: Number of timesteps per input sequence (TIMESTEPS constant)

    Returns:
        Tuple of:
        - X_sequences: Input array of shape (n_samples, seq_length, n_features)
        - y_sequences: Target array of shape (n_samples, n_targets)
        - scalers: Dictionary of fitted scalers for inverse transformation

    Raises:
        ValueError: If RAT type is not recognized
    """
    if rat not in FEATURE_COLS:
        raise ValueError(f"Invalid RAT type: {rat}. Must be '5g', 'pc5', or 'dsrc'.")

    feature_cols = FEATURE_COLS[rat]
    scalers = create_all_scalers()
    gps_scaler = scalers['tx_latitude']

    # Select only the columns needed for this RAT
    df = df[feature_cols].copy()

    # Replace RSRP sentinel values (16383 = modem error) with NaN before dropna
    for rsrp_col in ("rsrp_1", "rsrp_2"):
        if rsrp_col in df.columns:
            df[rsrp_col] = df[rsrp_col].where(df[rsrp_col] <= 0)

    df = df.dropna()

    # Filter out rows with default/invalid GPS coordinates
    df = df[~((df['tx_latitude'] == MIN_LAT) & (df['tx_longitude'] == MIN_LON))]

    # Normalize all features to [0, 1] range using predefined bounds
    print("Normalizing features...")
    gps_flag = False
    for col in feature_cols:
        if col in ('tx_latitude', 'tx_longitude'):
            # GPS coordinates normalized together (2D scaler)
            if not gps_flag:
                try:
                    df[['tx_latitude', 'tx_longitude']] = gps_scaler.transform(
                        df[['tx_latitude', 'tx_longitude']])
                except ValueError:
                    print("Warning: NaN GPS values detected, skipping GPS normalization")
                gps_flag = True
        else:
            # Each other feature normalized independently
            df[col] = scalers[col].transform(df[[col]])

    # Generate 3D sequences for LSTM: (samples, timesteps, features)
    print("Converting to LSTM sequences...")
    x_sequences_train, y_sequences_train = [], []
    if new:
        print(f"Generating {len(df) - seq_length} sequences...")
        for x_batch, y_batch in generate_lstm_sequences(df, feature_cols, target_cols, seq_length):
            x_sequences_train.append(x_batch)
            y_sequences_train.append(y_batch)
        # Concatenate all batches into single arrays
        x_sequences_train = np.concatenate(x_sequences_train, axis=0)
        y_sequences_train = np.concatenate(y_sequences_train, axis=0)

    print("Preprocessing complete.")
    return np.array(x_sequences_train), np.array(y_sequences_train), scalers


def inverse_transform(scalers, csv):
    """
    Convert scaled prediction log values back to original units.

    Reads a prediction log CSV with normalized values and creates
    a new file with denormalized (human-readable) values.

    Args:
        scalers: Dictionary of fitted MinMaxScaler objects
        csv: Path to the prediction log CSV file

    Output:
        Creates a new file at '{csv}_scaled' with denormalized values
    """
    log_df = pd.read_csv(csv, header=None,
                         names=["latitude", "longitude", "pred_latency", "actual_latency",
                                "mae_latency", "rmse_latency",
                                "pred_pdr", "actual_pdr", "mae_pdr", "rmse_pdr"])

    # Inverse transform GPS coordinates
    log_df[['latitude', 'longitude']] = scalers['tx_latitude'].inverse_transform(
        log_df[['latitude', 'longitude']])

    # Inverse transform latency values (both predicted and actual)
    log_df['pred_latency'] = scalers['latency_ms'].inverse_transform(log_df[['pred_latency']])
    log_df['actual_latency'] = scalers['latency_ms'].inverse_transform(log_df[['actual_latency']])

    # Inverse transform PDR values
    log_df['pred_pdr'] = scalers['pdr'].inverse_transform(log_df['pred_pdr'])
    log_df['actual_pdr'] = scalers['pdr'].inverse_transform(log_df['actual_pdr'])

    # MAE and RMSE columns are already in original units — leave as-is

    log_df.to_csv(csv + "_scaled", index=False)
