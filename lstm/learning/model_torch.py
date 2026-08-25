"""
PyTorch model definitions for RAT performance prediction.

Faithful port of learning/model.py (TensorFlow/Keras) to PyTorch:
- Same dual-output architecture (LSTM/GRU/SimpleRNN -> Dense(32, relu) ->
  two Dense(1, sigmoid) heads for latency_ms and pdr)
- Same loss (MSE per head, weighted 1.0/1.5) and Adam hyperparameters
  (lr=0.001, eps=1e-7 to match Keras's Adam default, unlike PyTorch's 1e-8)
- Same validation_split/shuffle semantics as Keras model.fit: the last
  `validation_split` fraction of the (unshuffled) arrays is held out for
  validation, and the remaining training data is reshuffled every epoch
- Same EarlyStopping semantics as keras.callbacks.EarlyStopping: monitors
  the training loss (not validation loss), and unconditionally restores the
  best epoch's weights at the end of training when restore_best_weights=True
  (verified against the current Keras source, keras/src/callbacks/
  early_stopping.py: on_train_end restores whenever
  restore_best_weights and best_weights is not None, regardless of whether
  patience was ever triggered)
- Same per-batch RMSE aggregation as Keras: Keras wraps a plain metric
  function in a MeanMetricWrapper that averages the metric's per-batch
  value across batches, which is NOT sqrt(mean(mse)) over the whole epoch
  (sqrt is concave, so the two differ). rmse_torch is only correct for a
  one-shot evaluation over a full array; fit_torch aggregates it per-batch
  to stay comparable with the TF training logs.

Weight initialization: by default this module applies a Keras-like
initializer (glorot_uniform for input-hidden and Dense weights, orthogonal
for hidden-hidden weights, zero biases, +1 forget-gate bias for LSTM to
mirror Keras's unit_forget_bias=True) so a *standalone* PyTorch training run
(no transplant) has an initial weight distribution comparable to Keras.
This does NOT make the weights bit-identical to a given TF run - the RNGs
differ. For an exact, bit-identical comparison between a TF and a PyTorch
run, build the TF model with a seed and transplant its weights via
learning.weight_transfer before training the PyTorch model.
"""
import copy

import numpy as np
import torch
import torch.nn as nn


def rmse_torch(y_pred, y_true):
    """
    Root Mean Square Error over a full tensor (one-shot evaluation only).

    Not suitable for per-epoch logging aggregation - see module docstring.
    """
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def _apply_keras_like_init(model, model_type):
    """Initialize RNN/Dense weights to match Keras's default distributions."""
    for name, param in model.rnn.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)

    if model_type == "lstm":
        units = model.rnn.hidden_size
        with torch.no_grad():
            model.rnn.bias_ih_l0[units:2 * units] += 1.0

    for layer in (model.dense, model.latency_head, model.pdr_head):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


class RATPredictor(nn.Module):
    """
    Dual-output RNN model for latency and PDR prediction (PyTorch).

    Mirrors learning.model.build_model: one RNN layer (64 units, last
    timestep only) -> Dense(32, relu) -> two Dense(1, sigmoid) heads.
    """

    def __init__(self, model_type, timesteps, features):
        super().__init__()
        self.model_type = model_type
        self.timesteps = timesteps
        self.features = features

        if model_type == "lstm":
            self.rnn = nn.LSTM(input_size=features, hidden_size=64, batch_first=True)
        elif model_type == "gru":
            self.rnn = nn.GRU(input_size=features, hidden_size=64, batch_first=True)
        elif model_type == "rnn":
            self.rnn = nn.RNN(input_size=features, hidden_size=64, batch_first=True,
                               nonlinearity="tanh")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.dense = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.latency_head = nn.Linear(32, 1)
        self.pdr_head = nn.Linear(32, 1)

        _apply_keras_like_init(self, model_type)

    def forward(self, x):
        x = x.float()
        _, state = self.rnn(x)
        last_hidden = state[0][-1] if self.model_type == "lstm" else state[-1]
        dense_out = self.relu(self.dense(last_hidden))
        latency = torch.sigmoid(self.latency_head(dense_out))
        pdr = torch.sigmoid(self.pdr_head(dense_out))
        return latency, pdr


def build_model_torch(model_type, timesteps, features):
    """
    Build a dual-output RNN model with an attached Adam optimizer.

    Args:
        model_type: Type of RNN layer ('lstm', 'gru', 'rnn')
        timesteps: Number of input timesteps
        features: Number of input features

    Returns:
        RATPredictor instance with `.optimizer` attached (mirrors the
        compiled Keras model owning its optimizer)
    """
    model = RATPredictor(model_type, timesteps, features)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)
    return model


class EarlyStopping:
    """
    Early stopping mirroring keras.callbacks.EarlyStopping semantics.

    Monitors the training loss (not validation loss) by default, tracks the
    best epoch's weights, and unconditionally restores them at the end of
    training when restore_best_weights=True - whether training stopped due
    to patience or ran for the full number of epochs.
    """

    def __init__(self, monitor="loss", patience=0, restore_best_weights=False, min_delta=0.0):
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.min_delta = min_delta
        self.model = None
        self.best = float("inf")
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def set_model(self, model):
        self.model = model

    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float("inf")
        self.best_weights = None
        self.stop_training = False

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, logs):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = copy.deepcopy(self.model.state_dict())

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(self.model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True

    def on_train_end(self):
        if self.restore_best_weights and self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)


def keras_style_validation_split(X, y_latency, y_pdr, validation_split):
    """
    Replicate Keras's model.fit(..., validation_split=...) split exactly:
    the last `validation_split` fraction of the arrays, in their original
    order, is held out for validation - no shuffling happens before this
    split (shuffling, if any, only applies to the remaining training data,
    and only during training itself).
    """
    n_val = int(len(X) * validation_split)
    if n_val <= 0:
        return X, y_latency, y_pdr, None, None, None
    return (X[:-n_val], y_latency[:-n_val], y_pdr[:-n_val],
            X[-n_val:], y_latency[-n_val:], y_pdr[-n_val:])


def fit_torch(model, X_train, y_train_dict, epochs, batch_size, validation_split=0.0,
              callbacks=None, verbose=1):
    """
Fonction d'entraînement PyTorch qui reproduit le comportement de
keras.Model.fit().

Elle reproduit exactement le comportement de `validation_split` de Keras :
la dernière fraction `validation_split` des tableaux (dans leur ordre
d'origine, avant tout mélange) est réservée à la validation. Les données
restantes, utilisées pour l'entraînement, sont mélangées à chaque epoch
(comportement par défaut de Keras avec `shuffle=True`).

Paramètres :
    model :
        Modèle RATPredictor possédant un attribut `.optimizer`
        (voir `build_model_torch`).

    X_train :
        Séquences d'entrée de forme `(N, timesteps, features)`.

    y_train_dict :
        Dictionnaire contenant les valeurs cibles :
        `{'latency_ms': (N,), 'pdr': (N,)}`.

    epochs :
        Nombre maximal d'epochs d'entraînement.

    batch_size :
        Taille des batches utilisés pendant l'entraînement.

    validation_split :
        Fraction des données réservée à la validation (prélevée à la fin
        des tableaux).

    callbacks :
        Liste d'objets implémentant les méthodes `on_train_begin`,
        `on_epoch_begin`, `on_epoch_end(epoch, logs)` et `on_train_end`,
        ainsi que, éventuellement, `set_model`.

    verbose :
        Si la valeur est vraie, affiche les métriques de chaque epoch.

Retour :
    Un dictionnaire `history` contenant les listes des métriques calculées
    à chaque epoch, avec les mêmes clés que `history.history` de Keras
    (`loss`, `latency_ms_loss`, `pdr_loss`, `latency_ms_rmse`,
    `pdr_rmse` ainsi que leurs équivalents `val_*`).

Cette fonction réalise les étapes suivantes :
    1. Sépare les données en ensembles d'entraînement et de validation.
    2. Mélange les données d'entraînement au début de chaque epoch.
    3. Entraîne le modèle batch par batch.
    4. Calcule les fonctions de perte (loss) et les métriques RMSE.
    5. Évalue le modèle sur les données de validation.
    6. Gère les callbacks, notamment `EarlyStopping`.
    7. Enregistre l'historique des métriques pour chaque epoch.

    """
    callbacks = callbacks or []
    device = next(model.parameters()).device

    X = np.asarray(X_train, dtype=np.float32)
    y_latency = np.asarray(y_train_dict["latency_ms"], dtype=np.float32).reshape(-1, 1)
    y_pdr = np.asarray(y_train_dict["pdr"], dtype=np.float32).reshape(-1, 1)

    X_tr, y_tr_lat, y_tr_pdr, X_val, y_val_lat, y_val_pdr = keras_style_validation_split(
        X, y_latency, y_pdr, validation_split)
    n_samples = len(X_tr)

    for cb in callbacks:
        if hasattr(cb, "set_model"):
            cb.set_model(model)
    for cb in callbacks:
        cb.on_train_begin()

    history = {}

    def _record(logs):
        for key, value in logs.items():
            history.setdefault(key, []).append(value)

    for epoch in range(epochs):
        for cb in callbacks:
            cb.on_epoch_begin(epoch)

        perm = np.random.permutation(n_samples)
        #pour le shuflle qui le comportement par défaut de keras
        #  permutation aléatoire complète des indices du train set, recalculée à chaque epoch  
        model.train()
        batch_losses, lat_losses, pdr_losses, lat_rmses, pdr_rmses = [], [], [], [], []

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.from_numpy(X_tr[idx]).to(device)
            yb_lat = torch.from_numpy(y_tr_lat[idx]).to(device)
            yb_pdr = torch.from_numpy(y_tr_pdr[idx]).to(device)

            model.optimizer.zero_grad()
            pred_lat, pred_pdr = model(xb)
            loss_lat = torch.mean((pred_lat - yb_lat) ** 2)
            loss_pdr = torch.mean((pred_pdr - yb_pdr) ** 2)
            loss = 1.0 * loss_lat + 1.5 * loss_pdr
            loss.backward()
            model.optimizer.step()

            batch_losses.append(loss.item())
            lat_losses.append(loss_lat.item())
            pdr_losses.append(loss_pdr.item())
            lat_rmses.append(rmse_torch(pred_lat.detach(), yb_lat).item())
            pdr_rmses.append(rmse_torch(pred_pdr.detach(), yb_pdr).item())

        logs = {
            "loss": float(np.mean(batch_losses)),
            "latency_ms_loss": float(np.mean(lat_losses)),
            "pdr_loss": float(np.mean(pdr_losses)),
            "latency_ms_rmse": float(np.mean(lat_rmses)),
            "pdr_rmse": float(np.mean(pdr_rmses)),
        }

        if X_val is not None and len(X_val) > 0:
            model.eval()
            val_lat_losses, val_pdr_losses, val_lat_rmses, val_pdr_rmses = [], [], [], []
            with torch.no_grad():
                for start in range(0, len(X_val), batch_size):
                    xb = torch.from_numpy(X_val[start:start + batch_size]).to(device)
                    yb_lat = torch.from_numpy(y_val_lat[start:start + batch_size]).to(device)
                    yb_pdr = torch.from_numpy(y_val_pdr[start:start + batch_size]).to(device)
                    pred_lat, pred_pdr = model(xb)
                    val_lat_losses.append(torch.mean((pred_lat - yb_lat) ** 2).item())
                    val_pdr_losses.append(torch.mean((pred_pdr - yb_pdr) ** 2).item())
                    val_lat_rmses.append(rmse_torch(pred_lat, yb_lat).item())
                    val_pdr_rmses.append(rmse_torch(pred_pdr, yb_pdr).item())

            logs["val_latency_ms_loss"] = float(np.mean(val_lat_losses))
            logs["val_pdr_loss"] = float(np.mean(val_pdr_losses))
            logs["val_loss"] = 1.0 * logs["val_latency_ms_loss"] + 1.5 * logs["val_pdr_loss"]
            logs["val_latency_ms_rmse"] = float(np.mean(val_lat_rmses))
            logs["val_pdr_rmse"] = float(np.mean(val_pdr_rmses))

        _record(logs)

        for cb in callbacks:
            cb.on_epoch_end(epoch, logs)

        if verbose:
            metrics_str = " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
            print(f"Epoch {epoch + 1}/{epochs} - {metrics_str}")

        if any(getattr(cb, "stop_training", False) for cb in callbacks):
            break

    for cb in callbacks:
        cb.on_train_end()

    return history


class TorchDataStreamGenerator(torch.utils.data.Dataset):
    """
    Batch-indexed dataset for streaming training, mirroring
    learning.model.DataStreamGenerator (a keras.utils.Sequence). Indexing
    returns whole batches (not single samples), matching the Keras Sequence
    semantics this replaces.
    """

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


def _run_batch(model, xb, yb_lat, yb_pdr, device, train):
    xb = torch.from_numpy(np.asarray(xb, dtype=np.float32)).to(device)
    yb_lat = torch.from_numpy(np.asarray(yb_lat, dtype=np.float32)).reshape(-1, 1).to(device)
    yb_pdr = torch.from_numpy(np.asarray(yb_pdr, dtype=np.float32)).reshape(-1, 1).to(device)

    if train:
        model.optimizer.zero_grad()
    pred_lat, pred_pdr = model(xb)
    loss = torch.mean((pred_lat - yb_lat) ** 2) + 1.5 * torch.mean((pred_pdr - yb_pdr) ** 2)
    if train:
        loss.backward()
        model.optimizer.step()
    return loss.item()


def incremental_train_torch(model, new_X, new_y, epochs=1, batch_size=1):
    """Retrain model incrementally with new data samples (mirrors incremental_train)."""
    print("Retraining with new data...")
    device = next(model.parameters()).device
    generator = TorchDataStreamGenerator(new_X, new_y, batch_size=batch_size)

    early_stopping = EarlyStopping(monitor="loss", patience=2)
    early_stopping.set_model(model)
    early_stopping.on_train_begin()

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for i in range(len(generator)):
            batch_x, batch_y = generator[i]
            loss = _run_batch(model, batch_x, batch_y["latency_ms"], batch_y["pdr"], device, train=True)
            batch_losses.append(loss)
        early_stopping.on_epoch_end(epoch, {"loss": float(np.mean(batch_losses))})
        if early_stopping.stop_training:
            break

    early_stopping.on_train_end()


def generate_new_measurement(x_new_data, y_new_data, index):
    """Extract a single sample from the dataset (identical to learning.model's version)."""
    if index >= len(x_new_data):
        raise IndexError("No more new measurements available in dataset.")
    x_new = x_new_data[index]
    y_new = y_new_data[index]
    return x_new.reshape(1, *x_new.shape), y_new.reshape(1, *y_new.shape)


def predict_and_retrain_torch(model, x_data, y_data, steps, start=0):
    """Online prediction and update workflow, mirroring predict_and_retrain."""
    device = next(model.parameters()).device
    for step in range(start, steps):
        new_x, new_y = generate_new_measurement(x_data, y_data, step)
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(new_x.astype(np.float32)).to(device)
            pred_lat, pred_pdr = model(xb)
        print(f"Step {step + 1}: Predicted: {pred_lat.item():.4f},{pred_pdr.item():.4f}, "
              f"Actual: {new_y[0, 0]:.4f},{new_y[0, 1]:.4f}")

        print(f"Retraining model with new data at step {step + 1}...")
        incremental_train_torch(model, new_x, {"latency_ms": new_y[:, 0], "pdr": new_y[:, 1]})

    print("Updated model ready for future predictions.")


def _process_batch_torch(model, x_new, y_new, gps_scaler, latency_scaler):
    """Predict on a batch and compute MAE/RMSE metrics (mirrors _process_batch)."""
    device = next(model.parameters()).device
    x_batch = np.array(x_new, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        pred_lat, pred_pdr = model(torch.from_numpy(x_batch).to(device))
    pred_latencies_norm = pred_lat.cpu().numpy().flatten()
    pred_pdrs_norm = pred_pdr.cpu().numpy().flatten()

    lat_lons = gps_scaler.inverse_transform(x_batch[:, -1, :2])
    pred_latencies_norm = np.clip(pred_latencies_norm, 0.0, 1.0)
    pred_latencies = latency_scaler.inverse_transform(pred_latencies_norm.reshape(-1, 1)).flatten()
    pred_pdrs = np.clip(pred_pdrs_norm, 0.0, 1.0)

    y_arr = np.array(y_new)
    actual_latencies = latency_scaler.inverse_transform(y_arr[:, 0].reshape(-1, 1)).flatten()
    actual_pdrs = np.clip(y_arr[:, 1], 0.0, 1.0)

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


def automatic_train_torch(model, X_new_data, y_new_data, batch_size=32, N=500, validation=0.15,
                           log_file="prediction_log.csv", rat="dsrc", model_type="lstm"):
    """Automatic incremental training with prediction logging (mirrors automatic_train)."""
    import os
    import time
    from config import MODEL_DIR, create_gps_scaler, create_latency_scaler

    gps_scaler = create_gps_scaler()
    latency_scaler = create_latency_scaler(rat)
    device = next(model.parameters()).device

    with open(log_file, "w") as f:
        f.write("latitude,longitude,pred_latency,actual_latency,mae_latency,rmse_latency,"
                "pred_pdr,actual_pdr,mae_pdr,rmse_pdr\n")

    x_new, y_new = [], []
    log_entries = []

    validation_split = int(validation * len(X_new_data))
    X_val, y_val = X_new_data[:validation_split], y_new_data[:validation_split]
    X_new_data, y_new_data = X_new_data[validation_split:], y_new_data[validation_split:]

    for i in range(len(X_new_data)):
        x_new.append(X_new_data[i])
        y_new.append(y_new_data[i])

        if len(x_new) >= N:
            log_entries.extend(_process_batch_torch(model, x_new, y_new, gps_scaler, latency_scaler))

            y_new_dict = {"latency_ms": np.array(y_new)[:, 0], "pdr": np.array(y_new)[:, 1]}

            generator = TorchDataStreamGenerator(x_new, y_new_dict, batch_size)
            early_stopping = EarlyStopping(monitor="loss", patience=2)
            early_stopping.set_model(model)
            early_stopping.on_train_begin()
            model.train()
            batch_losses = []
            for b in range(len(generator)):
                batch_x, batch_y = generator[b]
                loss = _run_batch(model, batch_x, batch_y["latency_ms"], batch_y["pdr"], device, train=True)
                batch_losses.append(loss)
            early_stopping.on_epoch_end(0, {"loss": float(np.mean(batch_losses))})
            early_stopping.on_train_end()

            x_new, y_new = [], []

            with open(log_file, "a") as f:
                for entry in log_entries:
                    f.write(entry + "\n")
            log_entries = []

    if len(x_new) > 0:
        log_entries.extend(_process_batch_torch(model, x_new, y_new, gps_scaler, latency_scaler))
        with open(log_file, "a") as f:
            for entry in log_entries:
                f.write(entry + "\n")

    save_path = os.path.join(MODEL_DIR, f"retrained_{model_type}_{rat}_{int(time.time())}.pt")
    save_torch_model(model, save_path)
    print(f"Retrained model saved to {save_path}")


def save_torch_model(model, path):
    """Save model architecture metadata + weights (not a full pickle of the module)."""
    torch.save({
        "model_type": model.model_type,
        "timesteps": model.timesteps,
        "features": model.features,
        "state_dict": model.state_dict(),
    }, path)


def load_torch_model(path, map_location="cpu"):
    """Reconstruct a RATPredictor from a checkpoint saved by save_torch_model."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model = RATPredictor(checkpoint["model_type"], checkpoint["timesteps"], checkpoint["features"])
    model.load_state_dict(checkpoint["state_dict"])
    model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-7)
    return model
