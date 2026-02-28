"""
Neural network model definitions for RAT performance prediction.

This module provides:
- Dual-output RNN architectures (LSTM, GRU, SimpleRNN) for latency and PDR prediction
- Custom RMSE metric for training evaluation
- Data streaming generators for memory-efficient training
- Incremental learning functions for online model updates
"""
import numpy as np
import os
import time
import keras
import tensorflow.keras.backend as K
from keras.models import Model
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tqdm import tqdm

from config import (
    MODEL_DIR,
    create_gps_scaler, create_latency_scaler, create_pdr_scaler,
)


def rmse(y_true, y_pred):
    """
    Root Mean Square Error metric for Keras models.

    Args:
        y_true: Ground truth tensor
        y_pred: Predicted values tensor

    Returns:
        RMSE value as a tensor
    """
    y_true = K.cast(y_true, np.float32)
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def build_model(model_type, timesteps, features):
    """
    Build a dual-output RNN model for latency and PDR prediction.

    Args:
        model_type: Type of RNN layer ('lstm', 'gru', 'rnn')
        timesteps: Number of input timesteps
        features: Number of input features

    Returns:
        Compiled Keras model
    """
    input_layer = Input(shape=(timesteps, features))

    if model_type == "lstm":
        rnn_layer = LSTM(64, return_sequences=False)(input_layer)
    elif model_type == "gru":
        rnn_layer = GRU(64, return_sequences=False)(input_layer)
    elif model_type == "rnn":
        rnn_layer = SimpleRNN(64, return_sequences=False)(input_layer)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    dense_layer = Dense(32, activation='relu')(rnn_layer)
    latency_output = Dense(1, activation='sigmoid', name="latency_ms")(dense_layer)
    pdr_output = Dense(1, activation='sigmoid', name="pdr")(dense_layer)

    model = Model(inputs=input_layer, outputs=[latency_output, pdr_output])
    model.compile(
        Adam(learning_rate=0.001),
        loss={'latency_ms': 'mse', 'pdr': 'mse'},
        loss_weights={'latency_ms': 1.0, 'pdr': 1.5},
        metrics={'latency_ms': [rmse], 'pdr': [rmse]}
    )
    return model


class DataStreamGenerator(keras.utils.Sequence):
    """
    Keras Sequence for streaming batched data during training.

    Enables memory-efficient training by loading data in batches
    rather than holding the entire dataset in memory.

    Attributes:
        X_data: Input feature sequences (N, timesteps, features)
        y_data_dict: Dictionary of target arrays {'latency_ms': [...], 'pdr': [...]}
        batch_size: Number of samples per batch
    """

    def __init__(self, X_data, y_data_dict, batch_size=1):
        """
        Initialize the data generator.

        Args:
            X_data: Input sequences array
            y_data_dict: Target values dictionary with 'latency_ms' and 'pdr' keys
            batch_size: Batch size for training (default: 1)
        """
        self.X_data = X_data
        self.y_data_dict = y_data_dict
        self.batch_size = batch_size

    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(len(self.X_data) / self.batch_size))

    def __getitem__(self, idx):
        """
        Get a single batch of data.

        Args:
            idx: Batch index

        Returns:
            Tuple of (batch_x, batch_y_dict) for multi-output model
        """
        batch_x = self.X_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_latency = self.y_data_dict["latency_ms"][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_pdr = self.y_data_dict["pdr"][idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), {"latency_ms": np.array(batch_y_latency), "pdr": np.array(batch_y_pdr)}


def incremental_train(model, new_X, new_y, epochs=1, batch_size=1):
    """
    Retrain model incrementally with new data samples.

    Used for online learning scenarios where the model is updated
    as new data arrives.

    Args:
        model: Keras model to retrain
        new_X: New input sequences
        new_y: New target values dictionary
        epochs: Number of training epochs (default: 1)
        batch_size: Training batch size (default: 1)
    """
    print("Retraining with new data...")
    generator = DataStreamGenerator(new_X, new_y, batch_size=batch_size)
    model.fit(generator, epochs=epochs, verbose=1, callbacks=[EarlyStopping(patience=2)])


def predict_and_retrain(model, x_data, y_data, steps, start=0):
    """
    Online prediction and update workflow for step-by-step learning.

    Processes data one sample at a time, making predictions and
    immediately retraining the model with the actual values.

    Args:
        model: Keras model to use
        x_data: Input data array
        y_data: Target data array
        steps: Number of steps to process
        start: Starting index (default: 0)
    """
    for step in range(start, steps):
        new_x, new_y = generate_new_measurement(x_data, y_data, step)
        prediction = model.predict(new_x)
        print(prediction[0])
        print(new_y[0])
        print(f"Step {step + 1}: Predicted: {prediction[0, 0]:.4f},{prediction[0, 1]:.4f}, "
              f"Actual: {new_y[0, 0]:.4f},{new_y[0, 1]:.4f}")

        print(f"Retraining model with new data at step {step + 1}...")
        incremental_train(model, new_x, new_y)

    print("Updated model ready for future predictions.")


def generate_new_measurement(x_new_data, y_new_data, index):
    """
    Extract a single sample from the dataset for prediction/retraining.

    Args:
        x_new_data: Input sequences array
        y_new_data: Target values array
        index: Sample index to extract

    Returns:
        Tuple of (x_sample, y_sample) reshaped for model input

    Raises:
        IndexError: If index exceeds available data
    """
    if index >= len(x_new_data):
        raise IndexError("No more new measurements available in dataset.")

    x_new = x_new_data[index]
    y_new = y_new_data[index]
    return x_new.reshape(1, *x_new.shape), y_new.reshape(1, *y_new.shape)


def _process_batch(model, x_new, y_new, gps_scaler, latency_scaler):
    """
    Predict on a batch and compute MAE/RMSE metrics.

    Args:
        model: Keras model
        x_new: List of input sequences
        y_new: List of target arrays
        gps_scaler: Scaler for GPS inverse transform
        latency_scaler: Scaler for latency inverse transform

    Returns:
        List of CSV log entry strings
    """
    x_batch = np.array(x_new)
    preds = model.predict(x_batch, verbose=0)
    pred_latencies_norm = preds[0].flatten()
    pred_pdrs_norm = preds[1].flatten()

    # Batch inverse transforms
    lat_lons = gps_scaler.inverse_transform(x_batch[:, -1, :2])
    pred_latencies_norm = np.clip(pred_latencies_norm, 0.0, 1.0)
    pred_latencies = latency_scaler.inverse_transform(
        pred_latencies_norm.reshape(-1, 1)).flatten()
    pred_pdrs = np.clip(pred_pdrs_norm, 0.0, 1.0)

    y_arr = np.array(y_new)
    actual_latencies = latency_scaler.inverse_transform(
        y_arr[:, 0].reshape(-1, 1)).flatten()
    actual_pdrs = np.clip(y_arr[:, 1], 0.0, 1.0)

    # Compute errors vectorized
    mae_lats = np.abs(pred_latencies - actual_latencies)
    rmse_lat = np.sqrt(np.mean((pred_latencies - actual_latencies) ** 2))
    mae_pdrs_arr = np.abs(pred_pdrs - actual_pdrs)
    rmse_pdr = np.sqrt(np.mean((pred_pdrs - actual_pdrs) ** 2))

    entries = []
    for j in range(len(x_new)):
        entries.append(
            f"{lat_lons[j, 0]},{lat_lons[j, 1]},"
            f"{pred_latencies[j]},{actual_latencies[j]},{mae_lats[j]},{rmse_lat},"
            f"{pred_pdrs[j]},{actual_pdrs[j]},{mae_pdrs_arr[j]},{rmse_pdr}")
    return entries


def automatic_train(model, X_new_data, y_new_data, batch_size=32, N=500, validation=0.15,
                    log_file="prediction_log.csv", rat="dsrc", model_type="lstm"):
    """
    Automatic incremental training with prediction logging.

    Implements a streaming learning approach where:
    1. Predictions are made on incoming data
    2. Results are logged with actual values and errors
    3. Model is retrained every N samples
    4. Validation set is held out for monitoring

    Args:
        model: Keras model to train and evaluate
        X_new_data: New input sequences (N, timesteps, features)
        y_new_data: New target values (N, 2) for latency and PDR
        batch_size: Training batch size (default: 32)
        N: Retraining interval - number of samples before each update (default: 500)
        validation: Fraction of data to hold out for validation (default: 0.15)
        log_file: Path to CSV file for logging predictions
        rat: RAT type identifier for model naming
        model_type: Model architecture name for file naming

    Output:
        Saves retrained model and writes prediction log to CSV
    """
    # Initialize scalers for inverse transformation of predictions
    gps_scaler = create_gps_scaler()
    latency_scaler = create_latency_scaler(rat)

    # Initialize prediction log file with header
    with open(log_file, "w") as f:
        f.write("latitude,longitude,pred_latency,actual_latency,mae_latency,rmse_latency,pred_pdr,actual_pdr,mae_pdr,rmse_pdr\n")

    # Buffers for accumulating samples before retraining
    x_new, y_new = [], []
    log_entries = []

    # Split off validation set from the beginning of data
    validation_split = int(validation * len(X_new_data))
    X_val, y_val = X_new_data[:validation_split], y_new_data[:validation_split]
    X_new_data, y_new_data = X_new_data[validation_split:], y_new_data[validation_split:]

    # Process each sample in streaming fashion
    for i in tqdm(range(len(X_new_data)), desc="Processing new data", unit="seq"):
        x_new.append(X_new_data[i])
        y_new.append(y_new_data[i])

        # Trigger retraining every N samples
        if len(x_new) >= N:
            log_entries.extend(_process_batch(model, x_new, y_new, gps_scaler, latency_scaler))

            # Prepare targets for multi-output model training
            y_new_dict = {
                "latency_ms": np.array(y_new)[:, 0],
                "pdr": np.array(y_new)[:, 1]
            }

            y_val_dict = {
                "latency_ms": np.array(y_val)[:, 0],
                "pdr": np.array(y_val)[:, 1]
            }

            # Retrain model with accumulated samples
            generator = DataStreamGenerator(x_new, y_new_dict, batch_size)
            model.fit(generator, epochs=1, verbose=1, callbacks=[EarlyStopping(patience=2)],
                      validation_data=(np.array(X_val), y_val_dict))

            # Clear buffers for next batch
            x_new, y_new = [], []

            # Flush log entries to file
            with open(log_file, "a") as f:
                for entry in log_entries:
                    f.write(entry + "\n")
            log_entries = []

    # Process remaining tail samples that didn't reach N
    if len(x_new) > 0:
        log_entries.extend(_process_batch(model, x_new, y_new, gps_scaler, latency_scaler))

        # Flush remaining log entries
        with open(log_file, "a") as f:
            for entry in log_entries:
                f.write(entry + "\n")
        log_entries = []

    # Save final retrained model with timestamp
    save_path = os.path.join(MODEL_DIR, f"retrained_{model_type}_{rat}_{int(time.time())}.keras")
    model.save(save_path)
    print(f"Retrained model saved to {save_path}")
