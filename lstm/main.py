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
    def __init__(self, filename, rat):
        super().__init__()
        self.filename = filename
        self.rat = rat

    def on_train_begin(self, logs=None):
        filepath = os.path.join(OUTPUT_DIR, f"{self.filename}_{self.rat}_training_log.csv")
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Time (seconds)",
                             "Train Loss (Latency)", "Train Loss (PDR)", "Total Train Loss",
                             "Val Loss (Latency)", "Val Loss (PDR)", "Total Val Loss",
                             "Train RMSE (Latency)", "Train RMSE (PDR)", "Total Train RMSE",
                             "Val RMSE (Latency)", "Val RMSE (PDR)", "Total Val RMSE"])

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
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
    # Ensure directories exist
    ensure_dir_exists(MODEL_DIR)
    ensure_dir_exists(OUTPUT_DIR)

    parser = argparse.ArgumentParser(description="LSTM Model Training and Prediction")
    parser.add_argument('--npz', type=str, help="Path to the archive containing processed training and new data")
    parser.add_argument('--data', type=str, help="Path to the folder for training data log files")
    parser.add_argument('--new_data', type=str, help="Path to the folder for the new data log file")
    parser.add_argument('--rat', type=str, required=True, choices=['5g', 'pc5', 'dsrc'], help="RAT type (5g, pc5 or dsrc)")
    parser.add_argument("--load", type=str, help="Path to load the model, or empty for most recent, or 'none' to train a new one")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train the model")
    parser.add_argument("--model", type=str, required=True, choices=['lstm', 'gru', 'rnn'], help="Type of model to use (lstm, gru, rnn.)")
    args = parser.parse_args()

    # Set constants
    PATH = args.data
    PATH_NEW = args.new_data
    RAT = args.rat
    MODEL_TYPE = args.model
    DATA_NPZ = args.npz
    epochs = args.epochs if args.epochs else EPOCHS
    tx_interval = 46 if RAT == "5g" else TX_INTERVAL_MS

    if PATH and not os.path.exists(PATH):
        raise ValueError(f"!!! Log file not found. {PATH}")

    FEATURES = FEATURES_COUNT[RAT]

    if PATH:
        files = find_files_with_string(PATH, f"trim_{RAT}")
        new_file = find_files_with_string(PATH_NEW, f"trim_{RAT}")[0]

    if args.load == "none":
        MODEL_PATH = None
        LOAD = False
    elif args.load:
        MODEL_PATH = os.path.join(MODEL_DIR, os.path.basename(args.load))
        LOAD = True
    else:
        part_lstm = get_latest_model("lstm", RAT)
        part_gru = get_latest_model("gru", RAT)
        part_rnn = get_latest_model("rnn", RAT)
        if part_lstm and part_gru and part_rnn:
            MODEL_PATH = part_lstm
            LOAD = True
        else:
            MODEL_PATH = None
            LOAD = False

    # Import data from log file
    if PATH and not DATA_NPZ:
        dfs = []
        for file in files:
            print(f"___ Processing file: {file}")
            df_part = pd.read_csv(os.path.join(PATH, file))
            df_part["tx_latitude"] = df_part["tx_latitude"].interpolate().bfill()
            df_part["tx_longitude"] = df_part["tx_longitude"].interpolate().bfill()
            print("___ Time to compute PDR...")
            df_part = compute_pdr_rolling(df_part, "tx_timestamp_ms", PDR_WINDOW, tx_interval)
            dfs.append(df_part)

        df = pd.concat(dfs, ignore_index=True)

        print(f"___ Processing file: {new_file}")
        df_new = pd.read_csv(os.path.join(PATH_NEW, new_file))
        df_new["tx_latitude"] = df_new["tx_latitude"].interpolate().bfill()
        df_new["tx_longitude"] = df_new["tx_longitude"].interpolate().bfill()
        print("___ Time to compute PDR...")
        df_new = compute_pdr_rolling(df_new, "tx_timestamp_ms", PDR_WINDOW, tx_interval)

    # Load or preprocess data
    if DATA_NPZ and os.path.exists(DATA_NPZ):
        data = np.load(DATA_NPZ)
        X_train = data["x_train"]
        y_train = data["y_train"]
        X_new_data = data["x_new_data"]
        y_new_data = data["y_new_data"]
        print("Preprocessed data loaded!")
    else:
        print("___ Starting data preprocessing.")
        print("______ Training set.")
        (X_train, y_train, scalers) = preprocess_lstm_input(df, new=True, rat=RAT, target_cols=TARGET_COLS, seq_length=TIMESTEPS)
        print("______ New set.")
        (X_new_data, y_new_data, scalers) = preprocess_lstm_input(df_new, new=True, rat=RAT, target_cols=TARGET_COLS, seq_length=TIMESTEPS)
        print("___ Preprocessing complete.")

        npz_path = os.path.join(OUTPUT_DIR, f"{RAT}_lstm_data.npz")
        np.savez_compressed(npz_path, x_train=X_train, y_train=y_train,
                            x_new_data=X_new_data, y_new_data=y_new_data)
        print(f"Preprocessed data saved! {npz_path}")

    y_train_dict = {
        "latency_ms": y_train[:, 0],
        "pdr": y_train[:, 1]
    }

    # Load existing model or train new one
    if LOAD and os.path.exists(MODEL_PATH):
        if MODEL_TYPE not in MODEL_PATH:
            raise ValueError("!!! Model file and model type do not match.")

        print(f"___ Loading existing model: {MODEL_PATH}")
        model_lstm = load_model(part_lstm, custom_objects={'rmse': rmse})
        model_gru = load_model(part_gru, custom_objects={'rmse': rmse})
        model_rnn = load_model(part_rnn, custom_objects={'rmse': rmse})

        with open(os.path.join(OUTPUT_DIR, f"lstm_{RAT}_training_history.json"), "r") as f:
            history_lstm = json.load(f)
        with open(os.path.join(OUTPUT_DIR, f"gru_{RAT}_training_history.json"), "r") as f:
            history_gru = json.load(f)
        with open(os.path.join(OUTPUT_DIR, f"rnn_{RAT}_training_history.json"), "r") as f:
            history_rnn = json.load(f)

        if model_lstm and history_lstm:
            print(f"___ Model loaded successfully: {part_lstm}")
        if model_gru and history_gru:
            print(f"___ Model loaded successfully: {part_gru}")
        if model_rnn and history_rnn:
            print(f"___ Model loaded successfully: {part_rnn}")
    else:
        if not LOAD:
            print("___ No model selected. Building...")
        elif not os.path.exists(MODEL_PATH):
            print("___ No model found. Building...")

        model_lstm = build_model("lstm", TIMESTEPS, FEATURES)
        model_gru = build_model("gru", TIMESTEPS, FEATURES)
        model_rnn = build_model("rnn", TIMESTEPS, FEATURES)

        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        print("___ Initial training...")
        print("______ LSTM")
        metrics_logger = MetricsLogger("lstm", RAT)
        history_lstm = model_lstm.fit(X_train, y_train_dict, epochs=epochs, batch_size=BATCH_SIZE,
                                       callbacks=[metrics_logger, early_stopping],
                                       validation_split=VALIDATION_SPLIT, verbose=1)
        save_path = os.path.join(MODEL_DIR, f"lstm_{RAT}_{int(time.time())}.keras")
        model_lstm.save(save_path)
        print(f"______ Model saved to {save_path}")

        print("______ GRU")
        metrics_logger = MetricsLogger("gru", RAT)
        history_gru = model_gru.fit(X_train, y_train_dict, epochs=epochs, batch_size=BATCH_SIZE,
                                     callbacks=[metrics_logger, early_stopping],
                                     validation_split=VALIDATION_SPLIT, verbose=1)
        save_path = os.path.join(MODEL_DIR, f"gru_{RAT}_{int(time.time())}.keras")
        model_gru.save(save_path)
        print(f"______ Model saved to {save_path}")

        print("______ RNN")
        metrics_logger = MetricsLogger("rnn", RAT)
        history_rnn = model_rnn.fit(X_train, y_train_dict, epochs=epochs, batch_size=BATCH_SIZE,
                                     callbacks=[metrics_logger, early_stopping],
                                     validation_split=VALIDATION_SPLIT, verbose=1)
        save_path = os.path.join(MODEL_DIR, f"rnn_{RAT}_{int(time.time())}.keras")
        model_rnn.save(save_path)
        print(f"______ Model saved to {save_path}")

        # Save history
        with open(os.path.join(OUTPUT_DIR, f"lstm_{RAT}_training_history.json"), "w") as f:
            json.dump(history_lstm.history, f)
        with open(os.path.join(OUTPUT_DIR, f"gru_{RAT}_training_history.json"), "w") as f:
            json.dump(history_gru.history, f)
        with open(os.path.join(OUTPUT_DIR, f"rnn_{RAT}_training_history.json"), "w") as f:
            json.dump(history_rnn.history, f)

    # Automatic retraining
    print("____________________________________________________")
    print("Automatic retraining.")
    automatic_train(model_lstm, X_new_data, y_new_data, 32, 500, 0.15,
                    os.path.join(OUTPUT_DIR, f"prediction_log_lstm_{RAT}.csv"), RAT, "lstm")
    automatic_train(model_gru, X_new_data, y_new_data, 32, 500, 0.15,
                    os.path.join(OUTPUT_DIR, f"prediction_log_gru_{RAT}.csv"), RAT, "gru")
    automatic_train(model_rnn, X_new_data, y_new_data, 32, 500, 0.15,
                    os.path.join(OUTPUT_DIR, f"prediction_log_rnn_{RAT}.csv"), RAT, "rnn")
