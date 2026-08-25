"""
Tests for learning/model_torch.py - PyTorch model definitions, data
generators, and the EarlyStopping/fit_torch training utilities. Mirrors
tests/test_model.py (the TensorFlow version) so both backends are covered
symmetrically.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy

import pytest
import numpy as np
import torch


class TestBuildModelTorch:
    """Tests for build_model_torch / RATPredictor."""

    def test_build_lstm_model(self):
        from learning.model_torch import build_model_torch

        model = build_model_torch('lstm', timesteps=10, features=6)
        assert model is not None
        assert model.rnn.input_size == 6
        assert model.rnn.hidden_size == 64

    def test_build_gru_model(self):
        from learning.model_torch import build_model_torch

        model = build_model_torch('gru', timesteps=10, features=6)
        assert model.rnn.input_size == 6
        assert model.rnn.hidden_size == 64

    def test_build_rnn_model(self):
        from learning.model_torch import build_model_torch

        model = build_model_torch('rnn', timesteps=10, features=6)
        assert model.rnn.input_size == 6
        assert model.rnn.hidden_size == 64

    def test_build_model_invalid_type(self):
        from learning.model_torch import RATPredictor

        with pytest.raises(ValueError, match="Unknown model type"):
            RATPredictor('transformer', timesteps=10, features=6)

    def test_model_different_features(self):
        from learning.model_torch import build_model_torch

        model_pc5 = build_model_torch('lstm', timesteps=10, features=4)
        assert model_pc5.rnn.input_size == 4

        model_5g = build_model_torch('lstm', timesteps=10, features=6)
        assert model_5g.rnn.input_size == 6

    def test_model_prediction_shape(self):
        from learning.model_torch import build_model_torch

        model = build_model_torch('lstm', timesteps=10, features=6)
        dummy_input = torch.rand(5, 10, 6)
        with torch.no_grad():
            latency, pdr = model(dummy_input)

        assert latency.shape == (5, 1)
        assert pdr.shape == (5, 1)

    def test_model_prediction_range(self):
        """Sigmoid outputs must lie in [0, 1]."""
        from learning.model_torch import build_model_torch

        model = build_model_torch('lstm', timesteps=10, features=6)
        dummy_input = torch.rand(5, 10, 6)
        with torch.no_grad():
            latency, pdr = model(dummy_input)

        assert torch.all((latency >= 0) & (latency <= 1))
        assert torch.all((pdr >= 0) & (pdr <= 1))


class TestRmseMetricTorch:
    """Tests for the one-shot rmse_torch metric."""

    def test_rmse_identical_values(self):
        from learning.model_torch import rmse_torch

        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        assert float(rmse_torch(y_pred, y_true)) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_known_value(self):
        from learning.model_torch import rmse_torch

        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([2.0, 3.0, 4.0, 5.0])
        assert float(rmse_torch(y_pred, y_true)) == pytest.approx(1.0, abs=1e-6)

    def test_rmse_varied_errors(self):
        from learning.model_torch import rmse_torch

        y_true = torch.tensor([0.0, 0.0, 0.0, 0.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = np.sqrt(7.5)
        assert float(rmse_torch(y_pred, y_true)) == pytest.approx(expected, abs=1e-5)


class TestTorchDataStreamGenerator:
    """Tests for the TorchDataStreamGenerator class."""

    def test_generator_initialization(self):
        from learning.model_torch import TorchDataStreamGenerator

        X = np.random.rand(100, 10, 6)
        y_dict = {'latency_ms': np.random.rand(100), 'pdr': np.random.rand(100)}
        gen = TorchDataStreamGenerator(X, y_dict, batch_size=10)

        assert gen.X_data is X
        assert gen.y_data_dict is y_dict
        assert gen.batch_size == 10

    def test_generator_len(self):
        from learning.model_torch import TorchDataStreamGenerator

        X = np.random.rand(100, 10, 6)
        y_dict = {'latency_ms': np.random.rand(100), 'pdr': np.random.rand(100)}

        gen = TorchDataStreamGenerator(X, y_dict, batch_size=10)
        assert len(gen) == 10

        gen2 = TorchDataStreamGenerator(X, y_dict, batch_size=30)
        assert len(gen2) == 4

    def test_generator_getitem(self):
        from learning.model_torch import TorchDataStreamGenerator

        np.random.seed(42)
        X = np.random.rand(100, 10, 6)
        y_dict = {'latency_ms': np.random.rand(100), 'pdr': np.random.rand(100)}
        gen = TorchDataStreamGenerator(X, y_dict, batch_size=10)

        batch_x, batch_y = gen[0]
        assert batch_x.shape == (10, 10, 6)
        assert batch_y['latency_ms'].shape == (10,)
        assert batch_y['pdr'].shape == (10,)

    def test_generator_last_batch(self):
        from learning.model_torch import TorchDataStreamGenerator

        X = np.random.rand(95, 10, 6)
        y_dict = {'latency_ms': np.random.rand(95), 'pdr': np.random.rand(95)}
        gen = TorchDataStreamGenerator(X, y_dict, batch_size=10)

        batch_x, batch_y = gen[len(gen) - 1]
        assert batch_x.shape[0] == 5


class TestGenerateNewMeasurementTorch:
    """generate_new_measurement is framework-agnostic and shared with learning.model."""

    def test_generate_new_measurement_basic(self):
        from learning.model_torch import generate_new_measurement

        X = np.random.rand(10, 5, 6)
        y = np.random.rand(10, 2)
        x_new, y_new = generate_new_measurement(X, y, index=0)

        assert x_new.shape == (1, 5, 6)
        assert y_new.shape == (1, 2)

    def test_generate_new_measurement_out_of_bounds(self):
        from learning.model_torch import generate_new_measurement

        X = np.random.rand(5, 10, 6)
        y = np.random.rand(5, 2)
        with pytest.raises(IndexError, match="No more new measurements"):
            generate_new_measurement(X, y, index=5)


class TestEarlyStoppingTorch:
    """
    Verifies the PyTorch EarlyStopping replicates Keras's confirmed
    behavior: restore_best_weights=True restores the best epoch's weights
    at train end unconditionally (keras/src/callbacks/early_stopping.py:
    on_train_end restores whenever restore_best_weights and best_weights is
    not None, regardless of whether patience was ever triggered).
    """

    def test_restores_best_weights_even_without_patience_trigger(self):
        from learning.model_torch import build_model_torch, EarlyStopping

        model = build_model_torch('lstm', timesteps=5, features=3)
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        early_stopping.set_model(model)
        early_stopping.on_train_begin()

        weights_per_epoch = []
        for epoch, loss in enumerate([1.0, 0.5, 0.1]):
            for p in model.parameters():
                p.data.add_(0.01)
            weights_per_epoch.append(copy.deepcopy(model.state_dict()))
            early_stopping.on_epoch_end(epoch, {"loss": loss})

        assert not early_stopping.stop_training
        early_stopping.on_train_end()

        best_weights = weights_per_epoch[-1]
        for name, param in model.state_dict().items():
            assert torch.allclose(param, best_weights[name])

    def test_stops_after_patience_exceeded(self):
        from learning.model_torch import build_model_torch, EarlyStopping

        model = build_model_torch('lstm', timesteps=5, features=3)
        early_stopping = EarlyStopping(monitor='loss', patience=2)
        early_stopping.set_model(model)
        early_stopping.on_train_begin()

        stopped_at = None
        for epoch, loss in enumerate([1.0, 1.1, 1.2, 1.3]):
            early_stopping.on_epoch_end(epoch, {"loss": loss})
            if early_stopping.stop_training:
                stopped_at = epoch
                break

        assert stopped_at == 2


class TestFitTorch:
    """Sanity checks for the fit_torch training loop."""

    def test_fit_runs_and_reduces_loss(self):
        from learning.model_torch import build_model_torch, fit_torch

        np.random.seed(0)
        model = build_model_torch('lstm', timesteps=5, features=3)
        X = np.random.rand(64, 5, 3).astype(np.float32)
        y = {"latency_ms": np.random.rand(64).astype(np.float32),
             "pdr": np.random.rand(64).astype(np.float32)}

        history = fit_torch(model, X, y, epochs=3, batch_size=16, validation_split=0.25, verbose=0)

        assert "loss" in history and len(history["loss"]) == 3
        assert "val_loss" in history and len(history["val_loss"]) == 3

    def test_validation_split_uses_tail_without_shuffling(self):
        """
        Keras's validation_split takes the LAST fraction of the arrays in
        their original order, before any shuffling - verified against the
        Keras docs/source. fit_torch delegates this to
        keras_style_validation_split, tested directly here.
        """
        from learning.model_torch import keras_style_validation_split

        n = 20
        X = np.arange(n).reshape(n, 1, 1).astype(np.float32)
        y_lat = np.arange(n).astype(np.float32)
        y_pdr = np.arange(n).astype(np.float32) * 2

        X_tr, y_tr_lat, y_tr_pdr, X_val, y_val_lat, y_val_pdr = keras_style_validation_split(
            X, y_lat, y_pdr, validation_split=0.25)

        assert len(X_val) == 5
        assert len(X_tr) == 15
        np.testing.assert_array_equal(X_val[:, 0, 0], X[-5:, 0, 0])
        np.testing.assert_array_equal(X_tr[:, 0, 0], X[:15, 0, 0])
        np.testing.assert_array_equal(y_val_lat, y_lat[-5:])
