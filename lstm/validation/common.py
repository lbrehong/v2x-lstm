"""
Shared helpers for cross-framework (TensorFlow vs PyTorch) validation.

All comparisons in this package force CPU execution on both frameworks:
GPU cuDNN kernels (TF and PyTorch each have their own, distinct from the
CPU/Eigen or CPU/MKL path) introduce float reordering noise on the order of
1e-4 to 1e-3 that has nothing to do with a porting bug, and would otherwise
be indistinguishable from a real divergence.

Even on CPU, expect a residual noise floor of ~1e-6 to 1e-5 in float32 due
to differing BLAS libraries (TF/Eigen vs PyTorch/MKL) and operation
ordering. A rel_error near that floor means the port is faithful; a jump of
several orders of magnitude at a specific stage of the chain indicates a
real bug at that stage.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np


def force_cpu_torch():
    """Call before building/running PyTorch models in validation scripts."""
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return torch.device("cpu")


def rel_error(tf_val, pt_val, eps=1e-8):
    """Elementwise relative error abs(tf_val - pt_val) / max(abs(tf_val), eps)."""
    tf_val = np.asarray(tf_val, dtype=np.float64)
    pt_val = np.asarray(pt_val, dtype=np.float64)
    denom = np.maximum(np.abs(tf_val), eps)
    return np.abs(tf_val - pt_val) / denom


def build_model_pair(model_type, rat, timesteps, features, seed):
    """
    Build a TF model (seeded) and a PyTorch model with transplanted, identical weights.

    Returns:
        (tf_model, torch_model), both on CPU, with guaranteed-identical initial weights.
    """
    from learning.weight_transfer import build_tf_model_with_seed, transplant
    from learning.model_torch import build_model_torch

    tf_model = build_tf_model_with_seed(model_type, timesteps, features, seed)
    torch_model = build_model_torch(model_type, timesteps, features)
    torch_model.to(force_cpu_torch())
    transplant(tf_model, torch_model, model_type)
    return tf_model, torch_model


def _build_tf_cell(rnn_layer, model_type):
    """A Keras RNN cell shares its full layer's weight structure exactly (kernel/
    recurrent_kernel/bias), so the trained/transplanted layer's weights load
    into the cell unchanged - no TF/PyTorch gate remapping needed here."""
    import keras

    units = rnn_layer.units
    features = rnn_layer.get_weights()[0].shape[0]
    if model_type == "lstm":
        cell = keras.layers.LSTMCell(units)
    elif model_type == "gru":
        cell = keras.layers.GRUCell(units, reset_after=True)
    else:
        cell = keras.layers.SimpleRNNCell(units)
    cell.build((None, features))
    cell.set_weights(rnn_layer.get_weights())
    return cell


def _build_torch_cell(rnn_module, model_type, features):
    """A PyTorch RNN cell (nn.LSTMCell/GRUCell/RNNCell) uses the exact same
    weight_ih/weight_hh/bias_ih/bias_hh layout and gate order as the full
    nn.LSTM/GRU/RNN layer, so weights copy across unchanged - no re-derivation
    of the Keras->PyTorch gate mapping is needed here."""
    import torch
    import torch.nn as nn

    units = rnn_module.hidden_size
    if model_type == "lstm":
        cell = nn.LSTMCell(features, units)
    elif model_type == "gru":
        cell = nn.GRUCell(features, units)
    else:
        cell = nn.RNNCell(features, units, nonlinearity="tanh")

    with torch.no_grad():
        cell.weight_ih.copy_(rnn_module.weight_ih_l0)
        cell.weight_hh.copy_(rnn_module.weight_hh_l0)
        cell.bias_ih.copy_(rnn_module.bias_ih_l0)
        cell.bias_hh.copy_(rnn_module.bias_hh_l0)
    return cell


def _forward_chain_tf(tf_model, model_type, x):
    import keras

    rnn_layer = next(layer for layer in tf_model.layers
                      if isinstance(layer, (keras.layers.LSTM, keras.layers.GRU, keras.layers.SimpleRNN)))
    dense_layer = next(layer for layer in tf_model.layers
                        if isinstance(layer, keras.layers.Dense) and layer.name not in ("latency_ms", "pdr"))
    latency_layer = tf_model.get_layer("latency_ms")
    pdr_layer = tf_model.get_layer("pdr")

    cell = _build_tf_cell(rnn_layer, model_type)
    batch = x.shape[0]
    units = rnn_layer.units
    h = np.zeros((batch, units), dtype=np.float32)
    c = np.zeros((batch, units), dtype=np.float32)
    states = [h, c] if model_type == "lstm" else [h]

    hidden_states, cell_states = [], []
    for t in range(x.shape[1]):
        _, states = cell(x[:, t, :].astype(np.float32), states)
        states = [s.numpy() if hasattr(s, "numpy") else np.asarray(s) for s in states]
        hidden_states.append(states[0])
        if model_type == "lstm":
            cell_states.append(states[1])

    last_hidden = states[0]
    dense_w, dense_b = dense_layer.get_weights()
    dense_pre = last_hidden @ dense_w + dense_b
    dense_out = np.maximum(dense_pre, 0.0)

    lat_w, lat_b = latency_layer.get_weights()
    lat_pre = dense_out @ lat_w + lat_b
    lat_post = 1.0 / (1.0 + np.exp(-lat_pre))

    pdr_w, pdr_b = pdr_layer.get_weights()
    pdr_pre = dense_out @ pdr_w + pdr_b
    pdr_post = 1.0 / (1.0 + np.exp(-pdr_pre))

    return {
        "input": x,
        "hidden_states": np.stack(hidden_states, axis=1),
        "cell_states": np.stack(cell_states, axis=1) if cell_states else None,
        "last_hidden": last_hidden,
        "dense_out": dense_out,
        "latency_pre": lat_pre, "latency_post": lat_post,
        "pdr_pre": pdr_pre, "pdr_post": pdr_post,
        "prediction": np.concatenate([lat_post, pdr_post], axis=-1),
    }


def _forward_chain_torch(torch_model, model_type, x):
    import torch

    rnn_module = torch_model.rnn
    cell = _build_torch_cell(rnn_module, model_type, x.shape[-1])
    batch = x.shape[0]
    units = rnn_module.hidden_size
    x_t = torch.from_numpy(x.astype(np.float32))
    h = torch.zeros(batch, units)
    c = torch.zeros(batch, units)

    hidden_states, cell_states = [], []
    with torch.no_grad():
        for t in range(x.shape[1]):
            if model_type == "lstm":
                h, c = cell(x_t[:, t, :], (h, c))
                cell_states.append(c.numpy())
            else:
                h = cell(x_t[:, t, :], h)
            hidden_states.append(h.numpy())

        last_hidden = h.numpy()
        dense_w = torch_model.dense.weight.detach().numpy()
        dense_b = torch_model.dense.bias.detach().numpy()
        dense_pre = last_hidden @ dense_w.T + dense_b
        dense_out = np.maximum(dense_pre, 0.0)

        lat_w = torch_model.latency_head.weight.detach().numpy()
        lat_b = torch_model.latency_head.bias.detach().numpy()
        lat_pre = dense_out @ lat_w.T + lat_b
        lat_post = 1.0 / (1.0 + np.exp(-lat_pre))

        pdr_w = torch_model.pdr_head.weight.detach().numpy()
        pdr_b = torch_model.pdr_head.bias.detach().numpy()
        pdr_pre = dense_out @ pdr_w.T + pdr_b
        pdr_post = 1.0 / (1.0 + np.exp(-pdr_pre))

    return {
        "input": x,
        "hidden_states": np.stack(hidden_states, axis=1),
        "cell_states": np.stack(cell_states, axis=1) if cell_states else None,
        "last_hidden": last_hidden,
        "dense_out": dense_out,
        "latency_pre": lat_pre, "latency_post": lat_post,
        "pdr_pre": pdr_pre, "pdr_post": pdr_post,
        "prediction": np.concatenate([lat_post, pdr_post], axis=-1),
    }


def instrumented_forward(tf_model, torch_model, model_type, x):
    """
    Run a manual, timestep-by-timestep forward pass through both models
    (same weights), capturing every stage of the chain: input -> per-timestep
    hidden state (+ cell state for LSTM) -> last hidden state -> dense layer
    output -> sigmoid pre/post-activation -> final prediction.

    Args:
        tf_model, torch_model: a pair from build_model_pair (identical weights)
        model_type: 'lstm', 'gru', or 'rnn'
        x: Input batch, shape (batch, timesteps, features)

    Returns:
        (chain_tf, chain_pt) - two dicts with identical keys and array shapes,
        ready for elementwise rel_error comparison.
    """
    x = np.asarray(x, dtype=np.float32)
    chain_tf = _forward_chain_tf(tf_model, model_type, x)
    chain_pt = _forward_chain_torch(torch_model, model_type, x)
    return chain_tf, chain_pt
