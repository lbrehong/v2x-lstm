"""
Main entry point for LSTM/GRU/RNN model training and prediction.

This module handles:
- Model training from raw CSV log files or preprocessed NPZ archives
- Incremental learning with new data
- Training metrics logging to CSV
- Automatic retraining workflow

Usage:
    python main.py --rat <5g|pc5|dsrc> --model <lstm|gru|rnn> --data /path/to/logs
    python main.py --rat <5g|pc5|dsrc> --model <lstm|gru|rnn> --npz /path/to/preprocessed.npz
"""
from dotenv import load_dotenv
load_dotenv()

import time
import argparse
import json
import csv
import os
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
    TX_INTERVAL_MS, PDR_WINDOW, TARGET_COLS, MODEL_DIR, OUTPUT_DIR,
    FEATURES_COUNT,
)
from utils import find_files_with_string, get_latest_model, ensure_dir_exists
from data_preprocessing import preprocess_lstm_input, compute_pdr_rolling
from model import build_model, automatic_train, rmse


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


if __name__ == "__main__":
    # Ensure output directories exist before any file operations
    ensure_dir_exists(MODEL_DIR)
    ensure_dir_exists(OUTPUT_DIR)

    # Parse command-line arguments
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
    args = parser.parse_args()

    # Extract arguments into local variables
    PATH = args.data
    PATH_NEW = args.new_data
    RAT = args.rat
    MODEL_TYPE = args.model
    DATA_NPZ = args.npz
    epochs = args.epochs if args.epochs else EPOCHS
    # 5G uses different TX interval (46ms) compared to PC5/DSRC (20ms)
    tx_interval = 46 if RAT == "5g" else TX_INTERVAL_MS

    # Validate input path
    if PATH and not os.path.exists(PATH):
        raise ValueError(f"Training data path not found: {PATH}")

    # Get feature count for this RAT type (used for model input shape)
    FEATURES = FEATURES_COUNT[RAT]

    # Find trimmed CSV files matching the RAT type
    if PATH:
        files = find_files_with_string(PATH, f"trim_{RAT}")
        new_file = find_files_with_string(PATH_NEW, f"trim_{RAT}")[0]

    # Determine model loading strategy
    if args.load == "none":
        # Train from scratch
        MODEL_PATH = None
        LOAD = False
    elif args.load:
        # Load specific model file
        MODEL_PATH = os.path.join(MODEL_DIR, os.path.basename(args.load))
        LOAD = True
    else:
        # Auto-detect most recent models for all architectures
        part_lstm = get_latest_model("lstm", RAT)
        part_gru = get_latest_model("gru", RAT)
        part_rnn = get_latest_model("rnn", RAT)
        if part_lstm and part_gru and part_rnn:
            MODEL_PATH = part_lstm
            LOAD = True
        else:
            MODEL_PATH = None
            LOAD = False

    # Load and preprocess raw CSV data (skip if NPZ provided)
    if PATH and not DATA_NPZ:
        # Process each training file: interpolate GPS gaps and compute rolling PDR
        dfs = []
        for file in files:
            print(f"Processing training file: {file}")
            df_part = pd.read_csv(os.path.join(PATH, file))
            # Fill GPS gaps using linear interpolation and backward fill
            df_part["tx_latitude"] = df_part["tx_latitude"].interpolate().bfill()
            df_part["tx_longitude"] = df_part["tx_longitude"].interpolate().bfill()
            print("Computing rolling PDR...")
            df_part = compute_pdr_rolling(df_part, "tx_timestamp_ms", PDR_WINDOW, tx_interval)
            dfs.append(df_part)

        # Concatenate all training files into single DataFrame
        df = pd.concat(dfs, ignore_index=True)

        # Process new data file for incremental learning
        print(f"Processing new data file: {new_file}")
        df_new = pd.read_csv(os.path.join(PATH_NEW, new_file))
        df_new["tx_latitude"] = df_new["tx_latitude"].interpolate().bfill()
        df_new["tx_longitude"] = df_new["tx_longitude"].interpolate().bfill()
        print("Computing rolling PDR...")
        df_new = compute_pdr_rolling(df_new, "tx_timestamp_ms", PDR_WINDOW, tx_interval)

    # Load preprocessed data from NPZ or preprocess from DataFrames
    if DATA_NPZ and os.path.exists(DATA_NPZ):
        # Load preprocessed sequences from NPZ archive
        data = np.load(DATA_NPZ)
        X_train = data["x_train"]
        y_train = data["y_train"]
        X_new_data = data["x_new_data"]
        y_new_data = data["y_new_data"]
        print(f"Loaded preprocessed data: {X_train.shape[0]} training, {X_new_data.shape[0]} new samples")
    else:
        # Preprocess raw data: normalize features and generate LSTM sequences
        print("Starting data preprocessing...")
        print("Processing training set...")
        (X_train, y_train, scalers) = preprocess_lstm_input(
            df, new=True, rat=RAT, target_cols=TARGET_COLS, seq_length=TIMESTEPS)
        print("Processing new data set...")
        (X_new_data, y_new_data, scalers) = preprocess_lstm_input(
            df_new, new=True, rat=RAT, target_cols=TARGET_COLS, seq_length=TIMESTEPS)
        print("Preprocessing complete.")

        # Cache preprocessed data for faster future runs
        npz_path = os.path.join(OUTPUT_DIR, f"{RAT}_lstm_data.npz")
        np.savez_compressed(npz_path, x_train=X_train, y_train=y_train,
                            x_new_data=X_new_data, y_new_data=y_new_data)
        print(f"Preprocessed data saved to: {npz_path}")

    # Convert targets to dictionary format for multi-output model
    y_train_dict = {
        "latency_ms": y_train[:, 0],
        "pdr": y_train[:, 1]
    }

    # Load existing models or train new ones from scratch
    if LOAD and os.path.exists(MODEL_PATH):
        # Validate model type matches file
        if MODEL_TYPE not in MODEL_PATH:
            raise ValueError(f"Model file '{MODEL_PATH}' does not match type '{MODEL_TYPE}'")

        # Load all three model architectures and their training histories
        print(f"Loading existing models for {RAT}...")
        model_lstm = load_model(part_lstm, custom_objects={'rmse': rmse})
        model_gru = load_model(part_gru, custom_objects={'rmse': rmse})
        model_rnn = load_model(part_rnn, custom_objects={'rmse': rmse})

        # Load training history for potential analysis
        with open(os.path.join(OUTPUT_DIR, f"lstm_{RAT}_training_history.json"), "r") as f:
            history_lstm = json.load(f)
        with open(os.path.join(OUTPUT_DIR, f"gru_{RAT}_training_history.json"), "r") as f:
            history_gru = json.load(f)
        with open(os.path.join(OUTPUT_DIR, f"rnn_{RAT}_training_history.json"), "r") as f:
            history_rnn = json.load(f)

        if model_lstm and history_lstm:
            print(f"LSTM model loaded: {part_lstm}")
        if model_gru and history_gru:
            print(f"GRU model loaded: {part_gru}")
        if model_rnn and history_rnn:
            print(f"RNN model loaded: {part_rnn}")
    else:
        # Build and train new models from scratch
        if not LOAD:
            print("No model specified, building new models...")
        elif not os.path.exists(MODEL_PATH):
            print("No existing model found, building new models...")

        # Build all three model architectures
        model_lstm = build_model("lstm", TIMESTEPS, FEATURES)
        model_gru = build_model("gru", TIMESTEPS, FEATURES)
        model_rnn = build_model("rnn", TIMESTEPS, FEATURES)

        # Early stopping prevents overfitting by monitoring loss
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        # Train LSTM model
        print("=" * 60)
        print("Training LSTM model...")
        metrics_logger = MetricsLogger("lstm", RAT)
        history_lstm = model_lstm.fit(X_train, y_train_dict, epochs=epochs, batch_size=BATCH_SIZE,
                                       callbacks=[metrics_logger, early_stopping],
                                       validation_split=VALIDATION_SPLIT, verbose=1)
        save_path = os.path.join(MODEL_DIR, f"lstm_{RAT}_{int(time.time())}.keras")
        model_lstm.save(save_path)
        print(f"LSTM model saved to {save_path}")

        # Train GRU model
        print("=" * 60)
        print("Training GRU model...")
        metrics_logger = MetricsLogger("gru", RAT)
        history_gru = model_gru.fit(X_train, y_train_dict, epochs=epochs, batch_size=BATCH_SIZE,
                                     callbacks=[metrics_logger, early_stopping],
                                     validation_split=VALIDATION_SPLIT, verbose=1)
        save_path = os.path.join(MODEL_DIR, f"gru_{RAT}_{int(time.time())}.keras")
        model_gru.save(save_path)
        print(f"GRU model saved to {save_path}")

        # Train SimpleRNN model
        print("=" * 60)
        print("Training SimpleRNN model...")
        metrics_logger = MetricsLogger("rnn", RAT)
        history_rnn = model_rnn.fit(X_train, y_train_dict, epochs=epochs, batch_size=BATCH_SIZE,
                                     callbacks=[metrics_logger, early_stopping],
                                     validation_split=VALIDATION_SPLIT, verbose=1)
        save_path = os.path.join(MODEL_DIR, f"rnn_{RAT}_{int(time.time())}.keras")
        model_rnn.save(save_path)
        print(f"SimpleRNN model saved to {save_path}")

        # Persist training history as JSON for later analysis
        with open(os.path.join(OUTPUT_DIR, f"lstm_{RAT}_training_history.json"), "w") as f:
            json.dump(history_lstm.history, f)
        with open(os.path.join(OUTPUT_DIR, f"gru_{RAT}_training_history.json"), "w") as f:
            json.dump(history_gru.history, f)
        with open(os.path.join(OUTPUT_DIR, f"rnn_{RAT}_training_history.json"), "w") as f:
            json.dump(history_rnn.history, f)

    # Incremental learning: retrain models with new data in streaming fashion
    print("=" * 60)
    print("Starting automatic incremental retraining...")
    print("=" * 60)
    automatic_train(model_lstm, X_new_data, y_new_data, 32, 500, 0.15,
                    os.path.join(OUTPUT_DIR, f"prediction_log_lstm_{RAT}.csv"), RAT, "lstm")
    automatic_train(model_gru, X_new_data, y_new_data, 32, 500, 0.15,
                    os.path.join(OUTPUT_DIR, f"prediction_log_gru_{RAT}.csv"), RAT, "gru")
    automatic_train(model_rnn, X_new_data, y_new_data, 32, 500, 0.15,
                    os.path.join(OUTPUT_DIR, f"prediction_log_rnn_{RAT}.csv"), RAT, "rnn")
