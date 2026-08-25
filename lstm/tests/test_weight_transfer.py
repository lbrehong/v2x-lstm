"""
Tests for learning/weight_transfer.py - Keras -> PyTorch weight
transplantation for LSTM, GRU, and SimpleRNN.

Serves as an automated regression guard for validation type 1 at the
'init' stage: with identical (transplanted) weights and CPU execution,
TF and PyTorch forward passes should match within a tight tolerance.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

from learning.weight_transfer import build_tf_model_with_seed, transplant, weight_rel_error
from learning.model_torch import build_model_torch


@pytest.mark.parametrize("model_type", ["lstm", "gru", "rnn"])
def test_transplanted_predictions_match(model_type):
    timesteps, features, seed = 10, 6, 123

    tf_model = build_tf_model_with_seed(model_type, timesteps, features, seed)
    torch_model = build_model_torch(model_type, timesteps, features)
    transplant(tf_model, torch_model, model_type)

    rng = np.random.default_rng(seed)
    X = rng.random((8, timesteps, features)).astype(np.float32)

    tf_pred_lat, tf_pred_pdr = tf_model.predict(X, verbose=0)
    torch_model.eval()
    with torch.no_grad():
        pt_pred_lat, pt_pred_pdr = torch_model(torch.from_numpy(X))

    lat_err = np.abs(tf_pred_lat.flatten() - pt_pred_lat.numpy().flatten())
    pdr_err = np.abs(tf_pred_pdr.flatten() - pt_pred_pdr.numpy().flatten())

    assert lat_err.max() < 1e-3, f"latency_ms max abs error {lat_err.max()}"
    assert pdr_err.max() < 1e-3, f"pdr max abs error {pdr_err.max()}"


@pytest.mark.parametrize("model_type", ["lstm", "gru", "rnn"])
def test_weight_rel_error_is_near_zero_after_transplant(model_type):
    timesteps, features, seed = 10, 4, 7

    tf_model = build_tf_model_with_seed(model_type, timesteps, features, seed)
    torch_model = build_model_torch(model_type, timesteps, features)
    transplant(tf_model, torch_model, model_type)

    errors = weight_rel_error(tf_model, torch_model, model_type)
    for name, err in errors.items():
        assert err < 1e-5, f"{name}: rel_error={err}"


def test_gru_rejects_reset_after_false():
    """
    The Keras->PyTorch GRU mapping assumes reset_after=True (the TF2/Keras3
    default, matching PyTorch's native "linear before reset" formulation).
    extract_tf_weights must fail loudly rather than silently mis-map an
    older reset_after=False layer.
    """
    import keras
    from learning.weight_transfer import extract_tf_weights

    inputs = keras.layers.Input(shape=(10, 4))
    gru_out = keras.layers.GRU(64, reset_after=False, return_sequences=False)(inputs)
    dense = keras.layers.Dense(32, activation='relu')(gru_out)
    latency = keras.layers.Dense(1, activation='sigmoid', name='latency_ms')(dense)
    pdr = keras.layers.Dense(1, activation='sigmoid', name='pdr')(dense)
    model = keras.models.Model(inputs=inputs, outputs=[latency, pdr])

    with pytest.raises(ValueError, match="reset_after"):
        extract_tf_weights(model, "gru")
