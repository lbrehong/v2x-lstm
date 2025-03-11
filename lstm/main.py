#%% md
# # This is a sample Jupyter Notebook
# 
# Below is an example of a code cell. 
# Put your cursor into the cell and press Shift+Enter to execute it and select the next one, or click 'Run Cell' button.
# 
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# 
# To learn more about Jupyter Notebooks in PyCharm, see [help](https://www.jetbrains.com/help/pycharm/ipython-notebook-support.html).
# For an overview of PyCharm, go to Help -> Learn IDE features or refer to [our documentation](https://www.jetbrains.com/help/pycharm/getting-started.html).
#%%
from dotenv import load_dotenv
load_dotenv()

import time
import argparse
import glob
import re
import json, csv
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from keras.callbacks import EarlyStopping, History, Callback
from keras.models import load_model
from data_preprocessing import preprocess_lstm_input, compute_pdr_rolling
from model import build_model, predict_and_retrain, automatic_train, rmse

#%%
TIMESTEPS = 10
FEATURES = 6
EPOCHS = 100 # number of epochs to train the model (early stopping set at 5)
SAMPLES = 20000 # number of samples to generate when testing
TRAIN_RATIO = 0.4 # ratio of samples to use for training
BATCH_SIZE = 32 # batch size for training, e.g. what is the number of samples to use for each epoch
VALIDATION_SPLIT = 0.15
TX_INTERVAL_MS = 20 # in ms, 50 packets per second
PDR_WINDOW = 10 # in seconds
TARGET_COLS = ['latency_ms', 'pdr']
MODEL_DIR = "models"
NAN = "NaN"
MIN_LAT, MAX_LAT = 43.554669, 43.568290
MIN_LON, MAX_LON = 1.463952, 1.472176


#%%
class MetricsLogger(Callback):
    def __init__(self, filename, rat):
        super().__init__()
        self.filename = filename
        self.rat = rat

    def on_train_begin(self, logs=None):
        with open(f"{self.filename}_{self.rat}_training_log.csv", mode='w', newline='') as file:
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

        # Retrieve losses separately for latency and PDR
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

        # Compute RMSE separately for each target
        total_train_rmse = np.sqrt(total_train_loss) if total_train_loss else None
        total_val_rmse = np.sqrt(total_val_loss) if total_val_loss else None

        # Write to log file
        with open(f"{self.filename}_{self.rat}_training_log.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1, epoch_time,
                train_loss_latency, train_loss_pdr, total_train_loss,
                val_loss_latency, val_loss_pdr, total_val_loss,
                train_rmse_latency, train_rmse_pdr, total_train_rmse,
                val_rmse_latency, val_rmse_pdr, total_val_rmse
            ])


#%%
# Ensure the models directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def find_files_with_string(directory, search_string):
    all_files = os.listdir(directory)  # Get all files and directories
    matching_files = [f for f in all_files if search_string in f]  # Filter files
    return matching_files

def get_latest_model(model_type, rat):
    """
    Finds the most recent model file based on the RAT type.
    :param type: Model type ('lstm', 'gru', 'rnn').
    :param rat: RAT type ('pc5' or 'dsrc').
    :return: Path to the most recent model file.
    """
    model_files = glob.glob(os.path.join(MODEL_DIR, model_type + "_" + rat + "_" + "*.keras"))
    model_files_with_time = []
    regex = re.compile(rf"{model_type}_{rat}_(\d+)\.keras")
    for file in model_files:
        match = regex.search(os.path.basename(file))
        if match:
            model_files_with_time.append((file, int(match.group(1))))
    if not model_files_with_time:
        print("No existing models found.")
        return None
    # Sort by extracted timecode (newest first)
    latest_model = max(model_files_with_time, key=lambda x: x[1])[0]
    print("Found a model:", latest_model)
    return latest_model


#%%
### Generate dummy data for demonstration (testing purposes)
# def generate_initial_data(samples=SAMPLES, timesteps=TIMESTEPS, features=FEATURES):
#     X_train = np.random.random((samples, timesteps, features))
#     y_train = np.random.random((samples,2))
#     return X_train, y_train
#
# # Initial training data
# X_train_generated, y_train_generated = generate_initial_data()
# print(f"X_train shape: {X_train_generated.shape}, y_train shape: {y_train_generated.shape}")

# # data normalizer
# def normalizer_init():
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     return scaler
#
# # Normalize data. Requires an array of single values, except for GPS that requires [latitude, longitude] elements
# def normalize(scaler, data):
#     return scaler.transform(data)


#%%

if __name__ == "__main__":
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
    if args.epochs:
        EPOCHS = int(args.epochs)
    if PATH:
        if not os.path.exists(PATH):
            raise ValueError(f"!!! Log file not found. {PATH}")

    if RAT == "pc5":
        FEATURES = 4
        if PATH:
            files = find_files_with_string(PATH, "trim_pc5")
            new_file = find_files_with_string(PATH_NEW, "trim_pc5")[0]
    elif RAT == "dsrc":
        FEATURES = 6
        if PATH:
            files = find_files_with_string(PATH, "trim_dsrc")
            new_file = find_files_with_string(PATH_NEW, "trim_dsrc")[0]
    elif RAT == "5g":
        FEATURES = 6
        TX_INTERVAL_MS = 46
        if PATH:
            files = find_files_with_string(PATH, "trim_5g")
            new_file = find_files_with_string(PATH_NEW, "trim_5g")[0]
    else:
        raise ValueError("Invalid RAT type. Must be '5g', 'pc5' or 'dsrc'.")

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
            # Fill NaN values in latitude/longitude (interpolate or forward-fill)
            df_part["tx_latitude"] = df_part["tx_latitude"].interpolate().bfill()
            df_part["tx_longitude"] = df_part["tx_longitude"].interpolate().bfill()
            # Compute PDR and add it to dataframe
            print("___ Time to compute PDR...")
            df_part = compute_pdr_rolling(df_part, "tx_timestamp_ms", PDR_WINDOW, TX_INTERVAL_MS)
            dfs.append(df_part)

        # Add to main dataframe
        df = pd.concat(dfs, ignore_index=True)

        print(f"___ Processing file: {new_file}")
        df_new = pd.read_csv(os.path.join(PATH_NEW, new_file))
        # Fill NaN values in latitude/longitude (interpolate or forward-fill)
        df_new["tx_latitude"] = df_new["tx_latitude"].interpolate().bfill()
        df_new["tx_longitude"] = df_new["tx_longitude"].interpolate().bfill()
        # Compute PDR and add it to dataframe
        print("___ Time to compute PDR...")
        compute_pdr_rolling(df_new, "tx_timestamp_ms", PDR_WINDOW, TX_INTERVAL_MS)


    # Load or preprocess data
    if DATA_NPZ and os.path.exists(DATA_NPZ):
        # Load preprocessed data
        data = np.load(DATA_NPZ)
        X_train = data["x_train"]
        y_train = data["y_train"]
        X_new_data = data["x_new_data"]
        y_new_data = data["y_new_data"]
        print("Preprocessed data loaded!")

    else:
        print("___ Starting data preprocessing.")
        print("______ Training set.")
        (X_train,y_train,scalers) = preprocess_lstm_input(df, new=True, rat=RAT, target_cols=TARGET_COLS, seq_length=TIMESTEPS)
        print("______ New set.")
        (X_new_data,y_new_data,scalers) = preprocess_lstm_input(df_new, new=True, rat=RAT, target_cols=TARGET_COLS, seq_length=TIMESTEPS)
        print("___ Preprocessing complete.")
        # Save preprocessed data
        np.savez_compressed(RAT + "_lstm_data.npz",
                            x_train=X_train, y_train=y_train,
                            x_new_data=X_new_data, y_new_data=y_new_data)
        print("Preprocessed data saved! " + RAT + "_lstm_data.npz")



    # Split y data into two separate columns for the two target values
    y_train_dict = {
        "latency_ms": y_train[:, 0],  # Extract latency values
        "pdr": y_train[:, 1]  # Extract PDR values
    }

    # Load existing model or train new one
    # Check for consistency of arguments
    if LOAD and os.path.exists(MODEL_PATH):
        #if not MODEL_PATH.__contains__(RAT):
        #    raise ValueError("!!! Model and log file RAT types do not match.")
        if not MODEL_PATH.__contains__(MODEL_TYPE):
            raise ValueError("!!! Model file and model type do not match.")

        # Load existing model
        print("___ Loading existing model: " + MODEL_PATH)
        model_lstm = load_model(part_lstm, custom_objects={'rmse': rmse})
        model_gru = load_model(part_gru, custom_objects={'rmse': rmse})
        model_rnn = load_model(part_rnn, custom_objects={'rmse': rmse})
        with open("lstm_" + RAT + "_training_history.json", "r") as f:
            history_lstm = json.load(f)
        with open("gru_" + RAT + "_training_history.json", "r") as f:
            history_gru = json.load(f)
        with open("rnn_" + RAT + "_training_history.json", "r") as f:
            history_rnn = json.load(f)
        if model_lstm and history_lstm:
            print("___ Model loaded successfully: " + part_lstm)
        if model_gru and history_gru:
            print("___ Model loaded successfully: " + part_gru)
        if model_rnn and history_rnn:
            print("___ Model loaded successfully: " + part_rnn)

    else:
        # Train new model
        if not LOAD:
            print("___ No model selected. Building...")
        elif not os.path.exists(MODEL_PATH):
            print("___ No model found. Building...")

        model_lstm = build_model("lstm", TIMESTEPS, FEATURES)
        model_gru = build_model("gru", TIMESTEPS, FEATURES)
        model_rnn = build_model("rnn", TIMESTEPS, FEATURES)
        # model.summary() # Print model summary
        early_stopping = EarlyStopping(monitor='loss', patience=5,
                                       restore_best_weights=True)  # Stop training if loss does not improve


        print("___ Initial training...")
        print("______ LSTM")
        metrics_logger = MetricsLogger("lstm", RAT)
        history_lstm = model_lstm.fit(X_train, y_train_dict, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[metrics_logger,early_stopping], validation_split=VALIDATION_SPLIT, verbose=1)
        save_path = os.path.join(MODEL_DIR, "lstm_" + RAT + "_" + str(int(time.time())) + ".keras")
        model_lstm.save(save_path)
        print(f"______ Model saved to {save_path}")
        print("______ GRU")
        metrics_logger = MetricsLogger("gru", RAT)
        history_gru = model_gru.fit(X_train, y_train_dict, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[metrics_logger,early_stopping], validation_split=VALIDATION_SPLIT, verbose=1)
        save_path = os.path.join(MODEL_DIR, "gru_" + RAT + "_" + str(int(time.time())) + ".keras")
        model_gru.save(save_path)
        print(f"______ Model saved to {save_path}")
        print("______ RNN")
        metrics_logger = MetricsLogger("rnn", RAT)
        history_rnn = model_rnn.fit(X_train, y_train_dict, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[metrics_logger,early_stopping], validation_split=VALIDATION_SPLIT, verbose=1)
        save_path = os.path.join(MODEL_DIR, "rnn_" + RAT + "_" + str(int(time.time())) + ".keras")
        model_rnn.save(save_path)
        print(f"______ Model saved to {save_path}")

        # Save history
        with open("lstm_" + RAT + "_training_history.json", "w") as f:
            json.dump(history_lstm.history, f)
        with open("gru_" + RAT + "_training_history.json", "w") as f:
            json.dump(history_gru.history, f)
        with open("rnn_" + RAT + "_training_history.json", "w") as f:
            json.dump(history_rnn.history, f)


#        if MODEL_TYPE == "lstm":
#            model = build_lstm_model(TIMESTEPS, FEATURES)
#        elif MODEL_TYPE == "gru":
#            model = build_gru_model(TIMESTEPS, FEATURES)
#        elif MODEL_TYPE == "rnn":
#            model = build_rnn_model(TIMESTEPS, FEATURES)
#        else:
#            raise ValueError("!!! Invalid model type. Must be 'lstm', 'gru' or 'rnn'.")
#        # model.summary() # Print model summary
#        early_stopping = EarlyStopping(monitor='loss', patience=5,
#                                       restore_best_weights=True)  # Stop training if loss does not improve
#        print("___ Initial training...")
#        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], validation_split=VALIDATION_SPLIT, verbose=1)
#        save_path = os.path.join(MODEL_DIR, MODEL_TYPE + "_" + RAT + "_" + str(int(time.time())) + ".keras")
#        model.save(save_path)
#        print(f"___ Model saved to {save_path}")


    # Automatic retraining
    print("____________________________________________________")
    print("Automatic retraining.")
    automatic_train(model_lstm, X_new_data, y_new_data, 32, 500, 0.15, "prediction_log_" + "lstm" + "_" + RAT +  ".csv", RAT, "lstm")
    automatic_train(model_gru, X_new_data, y_new_data, 32, 500, 0.15, "prediction_log_" + "gru" + "_" + RAT +  ".csv", RAT, "gru")
    automatic_train(model_rnn, X_new_data, y_new_data, 32, 500, 0.15, "prediction_log_" + "rnn" + "_" + RAT +  ".csv", RAT, "rnn")


#%%


#TODO list
    # Cleanup and reorganize, remove unused code
        # Lines 255 onwards should go into model.py and regroup all 3 models?