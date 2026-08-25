"""
Main entry point for LSTM/GRU/RNN model training and prediction (PyTorch).

Faithful PyTorch mirror of learning/main.py. Reuses the exact same
preprocessing pipeline (learning.data_preprocessing) and file-discovery
utilities (utils.py) as the TensorFlow entry point - only the model
construction/training backend differs.

Usage:
    python -m learning.main_torch --rat <5g|pc5|dsrc> --model <lstm|gru|rnn> --data /path/to/logs
    python -m learning.main_torch --rat <5g|pc5|dsrc> --model <lstm|gru|rnn> --npz /path/to/preprocessed.npz
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

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import config
from config import (
    TIMESTEPS, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, EARLY_STOPPING_PATIENCE,
    get_tx_interval_ms, PDR_WINDOW, TARGET_COLS, MODEL_DIR, FEATURES_COUNT,
)
from utils import find_files_with_string, get_latest_model, ensure_dir_exists
from learning.data_preprocessing import preprocess_lstm_input, compute_pdr_rolling
from learning.model_torch import (
    build_model_torch, fit_torch, EarlyStopping, automatic_train_torch,
    save_torch_model, load_torch_model,
)


def load_csv_data(base_path, file_list, pdr_window, tx_interval):
    """
    Load CSV files, interpolate GPS gaps, and compute rolling PDR.

    Framework-agnostic duplicate of learning.main.load_csv_data, kept local
    so the PyTorch training path does not require TensorFlow to be
    installed (importing learning.main pulls in keras).
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

    Framework-agnostic duplicate of learning.main.prepare_data - see
    load_csv_data docstring for why this isn't imported from learning.main.
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


class TorchMetricsLogger:
    """
    Per-epoch CSV metrics logger, mirroring learning.main.MetricsLogger.

    Written to a *_pt suffixed file so it can coexist with the TensorFlow
    training log for the same rat/model_type during side-by-side comparisons.
    """

    def __init__(self, filename, rat):
        self.filename = filename
        self.rat = rat
        self.start_time = None

    def set_model(self, model):
        pass

    def on_train_begin(self):
        filepath = os.path.join(config.OUTPUT_DIR, f"{self.filename}_{self.rat}_training_log_pt.csv")
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Time (seconds)",
                              "Train Loss (Latency)", "Train Loss (PDR)", "Total Train Loss",
                              "Val Loss (Latency)", "Val Loss (PDR)", "Total Val Loss",
                              "Train RMSE (Latency)", "Train RMSE (PDR)", "Total Train RMSE",
                              "Val RMSE (Latency)", "Val RMSE (PDR)", "Total Val RMSE"])

    def on_epoch_begin(self, epoch):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs):
        epoch_time = time.time() - self.start_time

        train_loss_latency = logs.get("latency_ms_loss")
        train_rmse_latency = logs.get("latency_ms_rmse")
        train_loss_pdr = logs.get("pdr_loss")
        train_rmse_pdr = logs.get("pdr_rmse")
        total_train_loss = logs.get("loss")

        val_loss_latency = logs.get("val_latency_ms_loss")
        val_rmse_latency = logs.get("val_latency_ms_rmse")
        val_loss_pdr = logs.get("val_pdr_loss")
        val_rmse_pdr = logs.get("val_pdr_rmse")
        total_val_loss = logs.get("val_loss")

        total_train_rmse = np.sqrt(total_train_loss) if total_train_loss else None
        total_val_rmse = np.sqrt(total_val_loss) if total_val_loss else None

        filepath = os.path.join(config.OUTPUT_DIR, f"{self.filename}_{self.rat}_training_log_pt.csv")
        with open(filepath, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1, epoch_time,
                train_loss_latency, train_loss_pdr, total_train_loss,
                val_loss_latency, val_loss_pdr, total_val_loss,
                train_rmse_latency, train_rmse_pdr, total_train_rmse,
                val_rmse_latency, val_rmse_pdr, total_val_rmse,
            ])

    def on_train_end(self):
        pass


def parse_args():
    """Parse command-line arguments (identical to learning.main plus --seed)."""
    parser = argparse.ArgumentParser(
        description="LSTM/GRU/RNN Model Training and Prediction for RAT Selection (PyTorch)")
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
    parser.add_argument("--seed", type=int, help="Random seed for reproducible training")
    return parser.parse_args()


def train_single_model_torch(model_type, timesteps, features, X_train, y_train_dict, rat, epochs):
    """Build, train, and save a single PyTorch model (mirrors train_single_model)."""
    display_name = {"lstm": "LSTM", "gru": "GRU", "rnn": "SimpleRNN"}[model_type]
    print("=" * 60)
    print(f"Training {display_name} model (PyTorch)...")

    model = build_model_torch(model_type, timesteps, features)
    metrics_logger = TorchMetricsLogger(model_type, rat)
    early_stopping = EarlyStopping(monitor="loss", patience=EARLY_STOPPING_PATIENCE,
                                    restore_best_weights=True)
    history = fit_torch(model, X_train, y_train_dict, epochs=epochs, batch_size=BATCH_SIZE,
                         validation_split=VALIDATION_SPLIT,
                         callbacks=[metrics_logger, early_stopping], verbose=1)

    save_path = os.path.join(MODEL_DIR, f"{model_type}_{rat}_{int(time.time())}.pt")
    save_torch_model(model, save_path)
    print(f"{display_name} model saved to {save_path}")

    return model, history


def main():
    """Main training pipeline (PyTorch)."""
    ensure_dir_exists(MODEL_DIR)
    ensure_dir_exists(config.OUTPUT_DIR)

    args = parse_args()

    if args.seed is not None:
        from learning.seed import set_torch_seed
        set_torch_seed(args.seed)

    PATH = args.data
    PATH_NEW = args.new_data
    RAT = args.rat
    MODEL_TYPE = args.model
    DATA_NPZ = args.npz
    epochs = args.epochs if args.epochs else EPOCHS
    tx_interval = get_tx_interval_ms(RAT)

    if PATH and not os.path.exists(PATH):
        raise ValueError(f"Training data path not found: {PATH}")

    FEATURES = FEATURES_COUNT[RAT]

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

    if args.load == "none":
        MODEL_PATH = None
        LOAD = False
    elif args.load:
        MODEL_PATH = os.path.join(MODEL_DIR, os.path.basename(args.load))
        LOAD = True
    else:
        existing = get_latest_model(MODEL_TYPE, RAT, extension="pt")
        if existing:
            MODEL_PATH = existing
            LOAD = True
        else:
            MODEL_PATH = None
            LOAD = False

    df, df_new = None, None
    if PATH and not DATA_NPZ:
        df = load_csv_data(PATH, files, PDR_WINDOW, tx_interval)
        if PATH_NEW:
            df_new = load_csv_data(PATH_NEW, [new_file], PDR_WINDOW, tx_interval)

    X_train, y_train, X_new_data, y_new_data = prepare_data(
        df, df_new, RAT, DATA_NPZ, config.OUTPUT_DIR)

    y_train_dict = {
        "latency_ms": y_train[:, 0],
        "pdr": y_train[:, 1],
    }

    if LOAD and MODEL_PATH and os.path.exists(MODEL_PATH):
        if MODEL_TYPE not in MODEL_PATH:
            raise ValueError(f"Model file '{MODEL_PATH}' does not match type '{MODEL_TYPE}'")

        print(f"Loading existing {MODEL_TYPE} model for {RAT}...")
        model_path = get_latest_model(MODEL_TYPE, RAT, extension="pt")
        model = load_torch_model(model_path)

        with open(os.path.join(config.OUTPUT_DIR,
                                f"{MODEL_TYPE}_{RAT}_training_history_pt.json"), "r") as f:
            history = json.load(f)

        print(f"{MODEL_TYPE.upper()} model loaded: {model_path}")
    else:
        if not LOAD:
            print(f"No model specified, building new {MODEL_TYPE} model (PyTorch)...")
        elif not MODEL_PATH or not os.path.exists(MODEL_PATH):
            print(f"No existing model found, building new {MODEL_TYPE} model (PyTorch)...")

        model, history = train_single_model_torch(
            MODEL_TYPE, TIMESTEPS, FEATURES, X_train, y_train_dict, RAT, epochs)

        with open(os.path.join(config.OUTPUT_DIR,
                                f"{MODEL_TYPE}_{RAT}_training_history_pt.json"), "w") as f:
            json.dump(history, f)

    if X_new_data is not None:
        print("=" * 60)
        print(f"Starting automatic incremental retraining for {MODEL_TYPE} (PyTorch)...")
        print("=" * 60)
        automatic_train_torch(model, X_new_data, y_new_data, 32, 500, 0.15,
                               os.path.join(config.OUTPUT_DIR,
                                            f"prediction_log_{MODEL_TYPE}_{RAT}_pt.csv"),
                               RAT, MODEL_TYPE)
    else:
        print("No new data provided. Skipping incremental retraining.")


if __name__ == "__main__":
    main()
