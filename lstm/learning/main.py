"""
Main entry point for LSTM/GRU/RNN model training and prediction.

This module handles:
- Model training from raw CSV log files or preprocessed NPZ archives
- Incremental learning with new data
- Training metrics logging to CSV
- Automatic retraining workflow

Usage:
    python -m learning.main --rat <5g|pc5|dsrc> --model <lstm|gru|rnn> --data /path/to/logs
    python -m learning.main --rat <5g|pc5|dsrc> --model <lstm|gru|rnn> --npz /path/to/preprocessed.npz
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import time
import argparse
import json
import csv
import pandas as pd
import numpy as np
import warnings

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from keras.callbacks import EarlyStopping, Callback
from keras.models import load_model

from config import (
    TIMESTEPS, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT,
    get_tx_interval_ms, PDR_WINDOW, TARGET_COLS, MODEL_DIR, OUTPUT_DIR,
    FEATURES_COUNT,
)
from utils import find_files_with_string, get_latest_model, ensure_dir_exists
from learning.data_preprocessing import preprocess_lstm_input, compute_pdr_rolling
from learning.model import build_model, automatic_train, rmse


class MetricsLogger(Callback):
    """
    Keras callback to log training metrics to CSV after each epoch.

    Logs include per-output losses (latency, PDR), total loss, and RMSE
    for both training and validation sets.

    Attributes:
        filename: Base name for the output log file
        rat: RAT type identifier for file naming
    """

    def __init__(self, filename, rat):
        """
        Initialize the metrics logger.

        Args:
            filename: Model type identifier (lstm, gru, rnn)
            rat: RAT type (5g, pc5, dsrc)
        """
        super().__init__()
        self.filename = filename
        self.rat = rat

    def on_train_begin(self, logs=None):
        """Create CSV file with header at training start."""
        filepath = os.path.join(OUTPUT_DIR, f"{self.filename}_{self.rat}_training_log.csv")
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Time (seconds)",
                             "Train Loss (Latency)", "Train Loss (PDR)", "Total Train Loss",
                             "Val Loss (Latency)", "Val Loss (PDR)", "Total Val Loss",
                             "Train RMSE (Latency)", "Train RMSE (PDR)", "Total Train RMSE",
                             "Val RMSE (Latency)", "Val RMSE (PDR)", "Total Val RMSE"])

    def on_epoch_begin(self, epoch, logs=None):
        """Record epoch start time for duration tracking."""
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Log all metrics at the end of each epoch."""
        epoch_time = time.time() - self.start_time

        # Extract per-output training metrics
        train_loss_latency = logs.get("latency_ms_loss")
        train_rmse_latency = logs.get("latency_ms_rmse")
        train_loss_pdr = logs.get("pdr_loss")
        train_rmse_pdr = logs.get("pdr_rmse")
        total_train_loss = logs.get("loss")

        # Extract per-output validation metrics
        val_loss_latency = logs.get("val_latency_ms_loss")
        val_rmse_latency = logs.get("val_latency_ms_rmse")
        val_loss_pdr = logs.get("val_pdr_loss")
        val_rmse_pdr = logs.get("val_pdr_rmse")
        total_val_loss = logs.get("val_loss")

        # Compute total RMSE from total loss
        total_train_rmse = np.sqrt(total_train_loss) if total_train_loss else None
        total_val_rmse = np.sqrt(total_val_loss) if total_val_loss else None

        # Append metrics to CSV
        filepath = os.path.join(OUTPUT_DIR, f"{self.filename}_{self.rat}_training_log.csv")
        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1, epoch_time,
                train_loss_latency, train_loss_pdr, total_train_loss,
                val_loss_latency, val_loss_pdr, total_val_loss,
                train_rmse_latency, train_rmse_pdr, total_train_rmse,
                val_rmse_latency, val_rmse_pdr, total_val_rmse
            ])


def parse_args():
    """Parse command-line arguments for model training."""
    parser = argparse.ArgumentParser(description="LSTM/GRU/RNN Model Training and Prediction for RAT Selection")
    parser.add_argument('--npz', type=str, help="Path to preprocessed NPZ archive (skips CSV processing)")
    parser.add_argument('--data', type=str, help="Path to folder containing training CSV log files")
    parser.add_argument('--new_data', type=str, help="Path to folder containing new data for incremental learning")
    parser.add_argument('--rat', type=str, required=True, choices=['5g', 'pc5', 'dsrc'],
                        help="RAT type: 5g (5G SA), pc5 (C-V2X PC5), or dsrc")
    parser.add_argument("--load", type=str,
                        help="Model path to load, empty for most recent, or 'none' to train from scratch")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (default: 100)")
    parser.add_argument("--model", type=str, required=True, choices=['lstm', 'gru', 'rnn'],
                        help="RNN architecture type: lstm, gru, or rnn (SimpleRNN)")
    return parser.parse_args()


def load_csv_data(base_path, file_list, pdr_window, tx_interval):
    """
    Load CSV files, interpolate GPS gaps, and compute rolling PDR.

    Args:
        base_path: Directory containing the CSV files
        file_list: List of filenames to process
        pdr_window: Rolling window size in seconds for PDR computation
        tx_interval: Expected transmission interval in milliseconds

    Returns:
        Concatenated DataFrame with PDR column added
    """
    dfs = []
    for file in file_list:
        print(f"Processing file: {file}")
        df_part = pd.read_csv(os.path.join(base_path, file))
        df_part["tx_latitude"] = df_part["tx_latitude"].interpolate().bfill()
        df_part["tx_longitude"] = df_part["tx_longitude"].interpolate().bfill()
        print("Computing rolling PDR...")
        df_part = compute_pdr_rolling(df_part, "tx_timestamp_ms", pdr_window, tx_interval)
        dfs.append(df_part)
    return pd.concat(dfs, ignore_index=True)


def prepare_data(df, df_new, rat, data_npz, output_dir):
    """
    Load from NPZ archive or preprocess from DataFrames.

    Args:
        df: Training DataFrame (can be None if loading from NPZ)
        df_new: New data DataFrame for incremental learning (can be None)
        rat: RAT type identifier
        data_npz: Path to NPZ archive (None to preprocess from DataFrames)
        output_dir: Directory to save cached NPZ data

    Returns:
        Tuple of (X_train, y_train, X_new_data, y_new_data).
        X_new_data and y_new_data may be None if no new data provided.
    """
    if data_npz and os.path.exists(data_npz):
        data = np.load(data_npz)
        X_train = data["x_train"]
        y_train = data["y_train"]
        X_new_data = data["x_new_data"] if "x_new_data" in data else None
        y_new_data = data["y_new_data"] if "y_new_data" in data else None
        if X_new_data is not None:
            print(f"Loaded preprocessed data: {X_train.shape[0]} training, {X_new_data.shape[0]} new samples")
        else:
            print(f"Loaded preprocessed data: {X_train.shape[0]} training samples (no new data)")
    else:
        print("Starting data preprocessing...")
        print("Processing training set...")
        (X_train, y_train, scalers) = preprocess_lstm_input(
            df, new=True, rat=rat, target_cols=TARGET_COLS, seq_length=TIMESTEPS)

        X_new_data, y_new_data = None, None
        if df_new is not None:
            print("Processing new data set...")
            (X_new_data, y_new_data, scalers) = preprocess_lstm_input(
                df_new, new=True, rat=rat, target_cols=TARGET_COLS, seq_length=TIMESTEPS)
        print("Preprocessing complete.")

        npz_path = os.path.join(output_dir, f"{rat}_lstm_data.npz")
        save_data = dict(x_train=X_train, y_train=y_train)
        if X_new_data is not None:
            save_data["x_new_data"] = X_new_data
            save_data["y_new_data"] = y_new_data
        np.savez_compressed(npz_path, **save_data)
        print(f"Preprocessed data saved to: {npz_path}")

    return X_train, y_train, X_new_data, y_new_data


def train_single_model(model_type, timesteps, features, X_train, y_train_dict, rat, epochs):
    """
    Build, train, and save a single model architecture.

    Args:
        model_type: RNN type ('lstm', 'gru', 'rnn')
        timesteps: Input sequence length
        features: Number of input features
        X_train: Training input sequences
        y_train_dict: Training targets as {'latency_ms': ..., 'pdr': ...}
        rat: RAT type identifier
        epochs: Maximum training epochs

    Returns:
        Tuple of (trained_model, training_history)
    """
    display_name = {"lstm": "LSTM", "gru": "GRU", "rnn": "SimpleRNN"}[model_type]
    print("=" * 60)
    print(f"Training {display_name} model...")

    model = build_model(model_type, timesteps, features)
    metrics_logger = MetricsLogger(model_type, rat)
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train_dict, epochs=epochs, batch_size=BATCH_SIZE,
                        callbacks=[metrics_logger, early_stopping],
                        validation_split=VALIDATION_SPLIT, verbose=1)

    save_path = os.path.join(MODEL_DIR, f"{model_type}_{rat}_{int(time.time())}.keras")
    model.save(save_path)
    print(f"{display_name} model saved to {save_path}")

    return model, history


def main():
    """Main training pipeline."""
    ensure_dir_exists(MODEL_DIR)
    ensure_dir_exists(OUTPUT_DIR)

    args = parse_args()

    PATH = args.data
    PATH_NEW = args.new_data
    RAT = args.rat
    MODEL_TYPE = args.model
    DATA_NPZ = args.npz
    epochs = args.epochs if args.epochs else EPOCHS
    tx_interval = get_tx_interval_ms(RAT)

    # Validate input path
    if PATH and not os.path.exists(PATH):
        raise ValueError(f"Training data path not found: {PATH}")

    # Get feature count for this RAT type (used for model input shape)
    FEATURES = FEATURES_COUNT[RAT]

    # Find trimmed CSV files matching the RAT type
    files = []
    if PATH:
        files = find_files_with_string(PATH, f"trim_{RAT}")
    new_file = None
    if PATH_NEW:
        if not os.path.exists(PATH_NEW):
            raise ValueError(f"New data path not found: {PATH_NEW}")
        new_files = find_files_with_string(PATH_NEW, f"trim_{RAT}")
        if not new_files:
            raise ValueError(f"No matching files found for trim_{RAT} in {PATH_NEW}")
        new_file = new_files[0]

    # Determine model loading strategy
    if args.load == "none":
        MODEL_PATH = None
        LOAD = False
    elif args.load:
        MODEL_PATH = os.path.join(MODEL_DIR, os.path.basename(args.load))
        LOAD = True
    else:
        existing = get_latest_model(MODEL_TYPE, RAT)
        if existing:
            MODEL_PATH = existing
            LOAD = True
        else:
            MODEL_PATH = None
            LOAD = False

    # Load and preprocess raw CSV data (skip if NPZ provided)
    df, df_new = None, None
    if PATH and not DATA_NPZ:
        df = load_csv_data(PATH, files, PDR_WINDOW, tx_interval)
        if PATH_NEW:
            df_new = load_csv_data(PATH_NEW, [new_file], PDR_WINDOW, tx_interval)

    # Prepare training data
    X_train, y_train, X_new_data, y_new_data = prepare_data(
        df, df_new, RAT, DATA_NPZ, OUTPUT_DIR)

    # Convert targets to dictionary format for multi-output model
    y_train_dict = {
        "latency_ms": y_train[:, 0],
        "pdr": y_train[:, 1]
    }

    # Load existing model or train new one from scratch
    if LOAD and os.path.exists(MODEL_PATH):
        # Validate model type matches file
        if MODEL_TYPE not in MODEL_PATH:
            raise ValueError(f"Model file '{MODEL_PATH}' does not match type '{MODEL_TYPE}'")

        # Load existing model and training history
        print(f"Loading existing {MODEL_TYPE} model for {RAT}...")
        model_path = get_latest_model(MODEL_TYPE, RAT)
        model = load_model(model_path, custom_objects={'rmse': rmse})

        with open(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_{RAT}_training_history.json"), "r") as f:
            history = json.load(f)

        print(f"{MODEL_TYPE.upper()} model loaded: {model_path}")
    else:
        # Build and train new model from scratch
        if not LOAD:
            print(f"No model specified, building new {MODEL_TYPE} model...")
        elif not os.path.exists(MODEL_PATH):
            print(f"No existing model found, building new {MODEL_TYPE} model...")

        model, history = train_single_model(
            MODEL_TYPE, TIMESTEPS, FEATURES, X_train, y_train_dict, RAT, epochs)

        # Persist training history as JSON for later analysis
        with open(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_{RAT}_training_history.json"), "w") as f:
            json.dump(history.history, f)

    # Incremental learning: retrain model with new data in streaming fashion
    if X_new_data is not None:
        print("=" * 60)
        print(f"Starting automatic incremental retraining for {MODEL_TYPE}...")
        print("=" * 60)
        automatic_train(model, X_new_data, y_new_data, 32, 500, 0.15,
                        os.path.join(OUTPUT_DIR, f"prediction_log_{MODEL_TYPE}_{RAT}.csv"), RAT, MODEL_TYPE)
    else:
        print("No new data provided. Skipping incremental retraining.")


if __name__ == "__main__":
    main()
