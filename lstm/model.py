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
    """Root Mean Square Error loss function."""
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
    latency_output = Dense(1, name="latency_ms")(dense_layer)
    pdr_output = Dense(1, name="pdr")(dense_layer)

    model = Model(inputs=input_layer, outputs=[latency_output, pdr_output])
    model.compile(
        Adam(learning_rate=0.001),
        loss={'latency_ms': 'mse', 'pdr': 'mse'},
        loss_weights={'latency_ms': 1.0, 'pdr': 1.5},
        metrics={'latency_ms': [rmse], 'pdr': [rmse]}
    )
    return model


class DataStreamGenerator(keras.utils.Sequence):
    """Generator for streaming data to the model during training."""

    def __init__(self, X_data, y_data_dict, batch_size=1):
        self.X_data = X_data
        self.y_data_dict = y_data_dict
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X_data) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.X_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_latency = self.y_data_dict["latency_ms"][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_pdr = self.y_data_dict["pdr"][idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), {"latency_ms": np.array(batch_y_latency), "pdr": np.array(batch_y_pdr)}


def incremental_train(model, new_X, new_y, epochs=1, batch_size=1):
    """Retrain model with new data incrementally."""
    print("Retraining with new data...")
    generator = DataStreamGenerator(new_X, new_y, batch_size=batch_size)
    model.fit(generator, epochs=epochs, verbose=1, callbacks=[EarlyStopping(patience=2)])


def predict_and_retrain(model, x_data, y_data, steps, start=0):
    """Online prediction and update workflow."""
    for step in range(start, steps):
        new_x, new_y = generate_new_measurement(x_data, y_data, step)
        prediction = model.predict(new_x)
        print(prediction[0])
        print(new_y[0])
        print(f"Step {step + 1}: Predicted: {prediction[0, 0]:.4f},{prediction[0, 1]:.4f}, Actual: {new_y[0, 0]:.4f},{new_y[0, 1]:.4f}")

        print(f"___ Retraining model with new data at step {step + 1}...")
        incremental_train(model, new_x, new_y)

    print("___ Updated model ready for future predictions.")


def generate_new_measurement(x_new_data, y_new_data, index):
    """Grab new data for predicting and re-training."""
    if index >= len(x_new_data):
        raise IndexError("!!! No more new measurements available.")

    x_new = x_new_data[index]
    y_new = y_new_data[index]
    return x_new.reshape(1, *x_new.shape), y_new.reshape(1, *y_new.shape)


def automatic_train(model, X_new_data, y_new_data, batch_size=32, N=500, validation=0.15,
                    log_file="prediction_log.csv", rat="dsrc", model_type="lstm"):
    """
    Automatic training with logging and periodic model updates.

    Args:
        model: Keras model to train
        X_new_data: New input data
        y_new_data: New target data
        batch_size: Training batch size
        N: Number of samples before retraining
        validation: Validation split ratio
        log_file: Path to log file
        rat: RAT type
        model_type: Model type for saving
    """
    gps_scaler = create_gps_scaler()
    latency_scaler = create_latency_scaler()

    with open(log_file, "w") as f:
        f.write("latitude,longitude,pred_latency,actual_latency,rmse_latency,pred_pdr,actual_pdr,rmse_pdr\n")

    x_new, y_new = [], []
    log_entries = []
    validation_split = int(validation * len(X_new_data))

    X_val, y_val = X_new_data[:validation_split], y_new_data[:validation_split]
    X_new_data, y_new_data = X_new_data[validation_split:], y_new_data[validation_split:]

    for i in tqdm(range(len(X_new_data)), desc="Processing new data", unit="seq"):
        x_new.append(X_new_data[i])
        y_new.append(y_new_data[i])

        if len(x_new) >= N:
            for j in range(len(x_new)):
                scaled_lat_lon = x_new[j][-1][:2].reshape(1, -1)
                lat, lon = gps_scaler.inverse_transform(scaled_lat_lon)[0]

                pred = model.predict(np.expand_dims(x_new[j], axis=0))

                pred_latency, pred_pdr = pred[0].flatten()[0], pred[1].flatten()[0]
                pred_latency = latency_scaler.inverse_transform(np.array(pred_latency).reshape(-1, 1))[0][0]
                pred_pdr = max(0.0, min(pred_pdr, 1.0))

                actual_latency, actual_pdr = y_new[j][0], y_new[j][1]
                actual_latency = latency_scaler.inverse_transform(np.array(actual_latency).reshape(-1, 1))[0][0]
                actual_pdr = max(0.0, min(actual_pdr, 1.0))

                rmse_latency = abs(pred_latency - actual_latency)
                rmse_pdr = abs(pred_pdr - actual_pdr)

                log_entries.append(f"{lat},{lon},{pred_latency},{actual_latency},{rmse_latency},{pred_pdr},{actual_pdr},{rmse_pdr}")

            y_new_dict = {
                "latency_ms": np.array(y_new)[:, 0],
                "pdr": np.array(y_new)[:, 1]
            }

            y_val_dict = {
                "latency_ms": np.array(y_val)[:, 0],
                "pdr": np.array(y_val)[:, 1]
            }

            generator = DataStreamGenerator(x_new, y_new_dict, batch_size)
            model.fit(generator, epochs=1, verbose=1, callbacks=[EarlyStopping(patience=2)],
                      validation_data=(np.array(X_val), y_val_dict))

            x_new, y_new = [], []

            with open(log_file, "a") as f:
                for entry in log_entries:
                    f.write(entry + "\n")
            log_entries = []

    save_path = os.path.join(MODEL_DIR, f"retrained_{model_type}_{rat}_{int(time.time())}.keras")
    model.save(save_path)
    print(f"______ Model saved to {save_path}")
