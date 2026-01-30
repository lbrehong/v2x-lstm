import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from tqdm import tqdm

MIN_LAT, MAX_LAT = 43.554669, 43.568290
MIN_LON, MAX_LON = 1.463952, 1.472176
MIN_THROUGHPUT, MAX_THROUGHPUT = 0, 100 # in Mbps
MIN_LATENCY, MAX_LATENCY = 4,50 # in ms
MIN_PDR, MAX_PDR = 0, 1
MIN_SINR_5G, MAX_SINR_5G = 200, 375
MIN_RSRP_5G, MAX_RSRP_5G = -127, -67
MIN_RSRP_DSRC, MAX_RSRP_DSRC = -150, -45

#FEATURE_COLS_PC5 = ['tx_seq_num','tx_timestamp_ms','tx_latitude','tx_longitude','latency_ms','pdr']
#FEATURE_COLS_DSRC = ['tx_seq_num','tx_timestamp_ms','tx_latitude','tx_longitude','rsrp_1','rsrp_2','latency_ms','pdr']
FEATURE_COLS_5G = ['tx_latitude','tx_longitude','latency_ms','sinr','rsrp','pdr']
FEATURE_COLS_PC5 = ['tx_latitude','tx_longitude','latency_ms','pdr']
FEATURE_COLS_DSRC = ['tx_latitude','tx_longitude','rsrp_1','rsrp_2','latency_ms','pdr']
#TARGET_COLS = ['throughput', 'pdr']
TARGET_COLS = ['latency_ms', 'pdr']


#%%
# Define preprocessing functions
def compute_pdr_rolling(df, time_column, window_size=1, packet_interval_ms=20):
    """
    Computes rolling PDR based on a fixed packet transmission rate.

    :param df: DataFrame containing the timestamps of received packets.
    :param time_column: Column name for timestamps.
    :param window_size: Time window (in seconds) for rolling PDR computation.
    :param packet_interval_ms: Fixed interval at which packets are sent (default: 20ms).
    :return: PDR rolling average series.
    """
    # Convert timestamps to seconds
    df["tx_timestamp_sec"] = df[time_column] / 1000
    # Compute rolling window size in terms of expected packets
    expected_packets_per_window = window_size * (1000 / packet_interval_ms)
    timestamps = df["tx_timestamp_sec"].to_numpy(dtype=float)
    rolling_window = deque()
    pdr_values = np.zeros(len(df)) # Ensure same length as df

    for i,t in enumerate(timestamps):
        # Remove timestamps outside the 1s window
        while rolling_window and rolling_window[0] < t - window_size:
            rolling_window.popleft()
        # Add current timestamp
        rolling_window.append(t)
        #if len(rolling_window) > expected_packets_per_window:
            #print(f"Warning: PDR value may exceed 1.0 at index {i} ({pdr_values[i]:.4f}). Popping leftmost value in window.")
        #    rolling_window.popleft()
        # Compute PDR
        pdr_values[i] = len(rolling_window) / expected_packets_per_window
        if pdr_values[i] > 1.0:
            #print(f"Warning: PDR value still exceeds 1.0 at index {i} ({pdr_values[i]:.4f})")
            pdr_values[i] = 1.0

    new_column = pd.Series(pdr_values, index=df.index)
    try :
        df["pdr"] = new_column
    except ValueError as e:
        if len(df) != len(new_column):
            print("Length mismatch")
    return df


# Convert to 3D in batches to preserve memory
def generate_lstm_sequences(data, feature_cols, target_cols, seq_length, batch_size=1024):
    """Yields LSTM sequences in batches to prevent memory overload.
    :param data: DataFrame with raw network data.
    :param feature_cols: List of feature column names.
    :param target_cols: List of target column names.
    :param seq_length: Number of past timesteps to use as input for LSTM.
    :param batch_size: Number of sequences to yield per batch (default: 1024).
    :return: Generator of LSTM sequences.
    """
    num_samples = len(data) - seq_length

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating sequences", unit="batch"):
        x_batch, y_batch = [], []
        for j in range(i, min(i + batch_size, num_samples)):
            x_batch.append(data[feature_cols].iloc[j: j + seq_length].values)
            y_batch.append(data[target_cols].iloc[j + seq_length].values)

        yield np.array(x_batch), np.array(y_batch)  # Yield batch instead of storing everything


#%%
def preprocess_lstm_input(df, new, rat, target_cols, seq_length):
    """
    Prepares LSTM input by computing PDR, normalizing features, and creating time series sequences.

    :param rat: RAT type ('5g', 'pc5' or 'dsrc').
    :param new: Is this data for a new model?
    :param df: DataFrame with raw network data.
    :param rat: RAT for this LSTM, used to determine feature column names.
    :param target_cols: List of target column names (e.g., ['throughput', 'PDR']).
    :param seq_length: Number of past timesteps to use as input for LSTM.
    """
    # Initialize variables
    gps_flag = False
    if rat == "pc5":
        feature_cols = FEATURE_COLS_PC5
    elif rat == "dsrc":
        feature_cols = FEATURE_COLS_DSRC
    elif rat == "5g":
        feature_cols = FEATURE_COLS_5G
    else:
        raise ValueError("!!! Invalid RAT type. Must be '5g', 'pc5' or 'dsrc'.")


    # Compute PDR and add it to dataframe
    #print("___ Time to compute PDR...")
    #compute_pdr_rolling(df, time_column, window_size_sec, packet_interval_ms)

    # Select relevant features and targets (e.g., get rid of seqnum and timestamp column)
    df = df[feature_cols].dropna()  # Ensure no NaNs
    # Filter out rows where 'tx_latitude' and 'tx_longitude' are equal to MIN_LAT and MIN_LON
    df = df[~((df['tx_latitude'] == MIN_LAT) & (df['tx_longitude'] == MIN_LON))]

    # Normalize features and target values
    gps_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]])
    throughput_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_THROUGHPUT], [MAX_THROUGHPUT]])
    latency_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_LATENCY], [MAX_LATENCY]])
    pdr_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_PDR], [MAX_PDR]])
    sinr_5g_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_SINR_5G], [MAX_SINR_5G]])
    rsrp_5g_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_RSRP_5G], [MAX_RSRP_5G]])
    rsrp_dsrc_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_RSRP_DSRC], [MAX_RSRP_DSRC]])

    scalers = {
        'tx_latitude': gps_scaler,
        'tx_longitude': gps_scaler,
        'latency_ms': latency_scaler,
        'throughput': throughput_scaler,
        'pdr': pdr_scaler,
        'sinr': sinr_5g_scaler,
        'rsrp': rsrp_5g_scaler,
        'rsrp_1': rsrp_dsrc_scaler,
        'rsrp_2': rsrp_dsrc_scaler
    }

    print("___ Time to normalize data...")
    for col in feature_cols: #+ target_cols:
        if col == 'tx_latitude' or col == 'tx_longitude':
            if not gps_flag:
                try:
                    df[['tx_latitude', 'tx_longitude']] = gps_scaler.transform(df[['tx_latitude', 'tx_longitude']])
                except ValueError as e:
                    print("!!! NaN GPS caught, skipping normalization")
                gps_flag = True
            else:
                continue
        else:
            df[col] = scalers[col].transform(df[[col]])  # Apply fitted scaler


    # Convert to sequences for LSTM
    print("___ Time to convert into 3D data...")
    x_sequences_train, y_sequences_train = [], []
    if new: # This is a new model, we need to concat 3D training data
        print(f"___ Training data... length {len(df) - seq_length}")
        for x_batch, y_batch in generate_lstm_sequences(df, feature_cols, target_cols, seq_length):
            x_sequences_train.append(x_batch)
            y_sequences_train.append(y_batch)
        x_sequences_train = np.concatenate(x_sequences_train, axis=0)
        y_sequences_train = np.concatenate(y_sequences_train, axis=0)


    print("___ Time to convert into numpy arrays...")
    return np.array(x_sequences_train), np.array(y_sequences_train), scalers


#%%

def inverse_transform(scalers, csv):
    log_df = pd.read_csv(csv, header=None, names=["latitude","longitude","pred_latency","actual_latency","rmse_latency","pred_pdr","actual_pdr","rmse_pdr"])

    log_df[['latitude', 'longitude']] = scalers['tx_latitude'].inverse_transform(log_df[['latitude', 'longitude']])
    log_df['pred_latency'] = scalers['latency_ms'].inverse_transform(log_df[['pred_latency']])
    log_df['actual_latency'] = scalers['latency_ms'].inverse_transform(log_df[['actual_latency']])
    log_df['pred_pdr'] = scalers['pdr'].inverse_transform(log_df['pred_pdr'])
    log_df['actual_pdr'] = scalers['pdr'].inverse_transform(log_df['actual_pdr'])

    log_df.to_csv(csv + "_scaled", index=False)
