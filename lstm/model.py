import numpy as np
import os
import time
import keras
import pandas as pd
import tensorflow.keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#%%

MODEL_DIR = "models"
MIN_LAT, MAX_LAT = 43.554669, 43.568290
MIN_LON, MAX_LON = 1.463952, 1.472176
MIN_LATENCY, MAX_LATENCY = 4,50 # in ms
MIN_PDR, MAX_PDR = 0, 1



### Define loss function
def rmse(y_true, y_pred):
    y_true = K.cast(y_true, np.float32) # Ensure same type because auto-casting pulls float64
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# Define model
def build_model(type, timesteps, features):
    input_layer = Input(shape=(timesteps, features)) # Input layer

    if type == "lstm":
        rnn_layer = LSTM(64, return_sequences=False)(input_layer) # LSTM layer
    elif type == "gru":
        rnn_layer = GRU(64, return_sequences=False)(input_layer) # GRU layer
    elif type == "rnn":
        rnn_layer = SimpleRNN(64, return_sequences=False)(input_layer) # SimpleRNN layer
    else:
        raise ValueError("Unknown model type")

    dense_layer = Dense(32, activation='relu')(rnn_layer) # Hidden Dense layer
    latency_output = Dense(1, name="latency_ms")(dense_layer) # Output layer for latency
    pdr_output = Dense(1, name="pdr")(dense_layer) # Output layer for PDR

    model = Model(inputs=input_layer, outputs=[latency_output, pdr_output])
    model.compile(Adam(learning_rate=0.001), loss={'latency_ms': 'mse', 'pdr': 'mse'}, loss_weights={'latency_ms': 1.0, 'pdr': 1.5}, metrics={'latency_ms': [rmse], 'pdr': [rmse]})
    return model



#%%
### Define data stream generator
class DataStreamGenerator(keras.utils.Sequence): # not used yet
    def __init__(self, X_data, y_data_dict, labels, batch_size=1):
        self.X_data = X_data
        self.y_data_dict = y_data_dict  # Now correctly formatted as a dictionary
        self.labels = np.array(labels)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X_data) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.X_data[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Fetch separate targets for latency & PDR
        batch_y_latency = self.y_data_dict["latency_ms"][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_pdr = self.y_data_dict["pdr"][idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), {"latency_ms": np.array(batch_y_latency), "pdr": np.array(batch_y_pdr)}

#%%
### Train the model
def incremental_train(model, new_X, new_y, epochs=1, batch_size=1):
    print("Retraining with new data...")
    generator = DataStreamGenerator(new_X, new_y, batch_size=batch_size)
    model.fit(generator, epochs=epochs, verbose=1, callbacks=[EarlyStopping(patience=2)])
    #model.fit(new_X, new_y, epochs=epochs, batch_size=batch_size, verbose=0)

# Online prediction and update workflow
def predict_and_retrain(model, x_data, y_data, steps, start=0): # TODO prevent out of bounds
    for step in range(start,steps):  # Feed new data into model from new set
        # Predict
        new_x, new_y = generate_new_measurement(x_data,y_data,step)
        prediction = model.predict(new_x)
        print(prediction[0])
        print(new_y[0])
        print(f"Step {step + 1}: Predicted: {prediction[0,0]:.4f},{prediction[0,1]:.4f}, Actual: {new_y[0,0]:.4f},{new_y[0,1]:.4f}")

        # Retrain with new measurement
        print(f"___ Retraining model with new data at step {step + 1}...")
        incremental_train(model, new_x, new_y)

    print("___ Updated model ready for future predictions.")

# Grab new data for predicting and re-training
def generate_new_measurement(x_new_data, y_new_data, index):
    # Ensure index is within bounds
    if index >= len(x_new_data):
        raise IndexError("!!! No more new measurements available.")

    x_new = x_new_data[index]
    y_new = y_new_data[index]
    return x_new.reshape(1, *x_new.shape), y_new.reshape(1, *y_new.shape)

#%%
def automatic_train(model, X_new_data, y_new_data, batch_size=32, N=500, validation=0.15, log_file="prediction_log.csv", rat="dsrc", type="lstm"):
    gps_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]])
    latency_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_LATENCY], [MAX_LATENCY]])
    pdr_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_PDR], [MAX_PDR]])

    with open(log_file, "w") as f:
        f.write("latitude,longitude,pred_latency,actual_latency,rmse_latency,pred_pdr,actual_pdr,rmse_pdr" + "\n")

    x_new, y_new = [], []
    log_entries = []
    validation_split = int(validation * len(X_new_data))  # 15% for validation

    X_val, y_val = X_new_data[:validation_split], y_new_data[:validation_split]
    X_new_data, y_new_data = X_new_data[validation_split:], y_new_data[validation_split:]

    for i in tqdm(range(len(X_new_data)), desc="Processing new data", unit="seq"):
        x_new.append(X_new_data[i])
        y_new.append(y_new_data[i])

        if len(x_new) >= N:  # Retrain every 500 new points
            latency_errors = []
            pdr_errors = []

            # Log predictions before training
            for j in range(len(x_new)):
                # Extract last latitude, longitude from the sequence (before scaling)
                scaled_lat_lon = x_new[j][-1][:2].reshape(1, -1)

                # Inverse transform to get actual GPS coordinates
                lat, lon = gps_scaler.inverse_transform(scaled_lat_lon)[0]

                # Get prediction
                pred = model.predict(np.expand_dims(x_new[j], axis=0))  # Predict single sequence

                # Ensure pred is a tuple with two arrays (latency, PDR)
                pred_latency, pred_pdr = pred[0].flatten()[0], pred[1].flatten()[0]  # Extract first values from both arrays
                pred_latency = latency_scaler.inverse_transform(np.array(pred_latency).reshape(-1, 1))[0][0]
                pred_pdr = max(0.0, min(pred_pdr, 1.0)) # Cap PDR at 1.0

                # Actual values (correct shape)
                actual_latency, actual_pdr = y_new[j][0], y_new[j][1]
                actual_latency = latency_scaler.inverse_transform(np.array(actual_latency).reshape(-1, 1))[0][0]
                actual_pdr = max(0.0, min(actual_pdr, 1.0)) # Cap PDR at 1.0

                # Store squared errors for batch RMSE calculation
                latency_errors.append((pred_latency - actual_latency) ** 2)
                pdr_errors.append((pred_pdr - actual_pdr) ** 2)

                # Compute RMSE over batch
                rmse_latency = abs(pred_latency - actual_latency)
                rmse_pdr = abs(pred_pdr - actual_pdr)

                log_entries.append(f"{lat},{lon},{pred_latency},{actual_latency},{rmse_latency},{pred_pdr},{actual_pdr},{rmse_pdr}")


            # Convert y_new to dictionary format for multi-output training
            y_new_dict = {
                "latency_ms": np.array(y_new)[:, 0],  # Extract latency values
                "pdr": np.array(y_new)[:, 1]       # Extract PDR values
            }

            # Convert validation set to dictionary format
            y_val_dict = {
                "latency_ms": np.array(y_val)[:, 0],
                "pdr": np.array(y_val)[:, 1]
            }

            # Train the model
            generator = DataStreamGenerator(x_new, y_new_dict, batch_size)
            model.fit(generator, epochs=1, verbose=1, callbacks=[EarlyStopping(patience=2)], validation_data=(np.array(X_val), y_val_dict))

            x_new, y_new = [], []  # Clear batch

            with open(log_file, "a") as f:
                for entry in log_entries:
                    f.write(entry + "\n")
            log_entries = []

    save_path = os.path.join(MODEL_DIR, "retrained_" + type + "_" + rat + "_" + str(int(time.time())) + ".keras")
    model.save(save_path)
    print(f"______ Model saved to {save_path}")

#%%