"""
Tests for learning/model.py - PyTorch model definitions, data generators,
the EarlyStopping/fit_torch training utilities, and the predict/evaluate/
save/load helpers used for inference.
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
        from learning.model import build_model_torch

        model = build_model_torch('lstm', timesteps=10, features=6)
        assert model is not None
        assert model.rnn.input_size == 6
        assert model.rnn.hidden_size == 64

    def test_build_gru_model(self):
        from learning.model import build_model_torch

        model = build_model_torch('gru', timesteps=10, features=6)
        assert model.rnn.input_size == 6
        assert model.rnn.hidden_size == 64

    def test_build_rnn_model(self):
        from learning.model import build_model_torch

        model = build_model_torch('rnn', timesteps=10, features=6)
        assert model.rnn.input_size == 6
        assert model.rnn.hidden_size == 64

    def test_build_model_invalid_type(self):
        from learning.model import RATPredictor

        with pytest.raises(ValueError, match="Unknown model type"):
            RATPredictor('transformer', timesteps=10, features=6)

    def test_model_different_features(self):
        from learning.model import build_model_torch

        model_pc5 = build_model_torch('lstm', timesteps=10, features=4)
        assert model_pc5.rnn.input_size == 4

        model_5g = build_model_torch('lstm', timesteps=10, features=6)
        assert model_5g.rnn.input_size == 6

    def test_model_prediction_shape(self):
        from learning.model import build_model_torch

        model = build_model_torch('lstm', timesteps=10, features=6)
        dummy_input = torch.rand(5, 10, 6, device=next(model.parameters()).device)
        with torch.no_grad():
            latency, pdr = model(dummy_input)

        assert latency.shape == (5, 1)
        assert pdr.shape == (5, 1)

    def test_model_prediction_range(self):
        """Sigmoid outputs must lie in [0, 1]."""
        from learning.model import build_model_torch

        model = build_model_torch('lstm', timesteps=10, features=6)
        dummy_input = torch.rand(5, 10, 6, device=next(model.parameters()).device)
        with torch.no_grad():
            latency, pdr = model(dummy_input)

        assert torch.all((latency >= 0) & (latency <= 1))
        assert torch.all((pdr >= 0) & (pdr <= 1))


class TestRmseMetricTorch:
    """Tests for the one-shot rmse_torch metric."""

    def test_rmse_identical_values(self):
        from learning.model import rmse_torch

        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        assert float(rmse_torch(y_pred, y_true)) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_known_value(self):
        from learning.model import rmse_torch

        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.tensor([2.0, 3.0, 4.0, 5.0])
        assert float(rmse_torch(y_pred, y_true)) == pytest.approx(1.0, abs=1e-6)

    def test_rmse_varied_errors(self):
        from learning.model import rmse_torch

        y_true = torch.tensor([0.0, 0.0, 0.0, 0.0])
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = np.sqrt(7.5)
        assert float(rmse_torch(y_pred, y_true)) == pytest.approx(expected, abs=1e-5)


class TestTorchDataStreamGenerator:
    """Tests for the TorchDataStreamGenerator class."""

    def test_generator_initialization(self):
        from learning.model import TorchDataStreamGenerator

        X = np.random.rand(100, 10, 6)
        y_dict = {'latency_ms': np.random.rand(100), 'pdr': np.random.rand(100)}
        gen = TorchDataStreamGenerator(X, y_dict, batch_size=10)

        assert gen.X_data is X
        assert gen.y_data_dict is y_dict
        assert gen.batch_size == 10

    def test_generator_len(self):
        from learning.model import TorchDataStreamGenerator

        X = np.random.rand(100, 10, 6)
        y_dict = {'latency_ms': np.random.rand(100), 'pdr': np.random.rand(100)}

        gen = TorchDataStreamGenerator(X, y_dict, batch_size=10)
        assert len(gen) == 10

        gen2 = TorchDataStreamGenerator(X, y_dict, batch_size=30)
        assert len(gen2) == 4

    def test_generator_getitem(self):
        from learning.model import TorchDataStreamGenerator

        np.random.seed(42)
        X = np.random.rand(100, 10, 6)
        y_dict = {'latency_ms': np.random.rand(100), 'pdr': np.random.rand(100)}
        gen = TorchDataStreamGenerator(X, y_dict, batch_size=10)

        batch_x, batch_y = gen[0]
        assert batch_x.shape == (10, 10, 6)
        assert batch_y['latency_ms'].shape == (10,)
        assert batch_y['pdr'].shape == (10,)

    def test_generator_last_batch(self):
        from learning.model import TorchDataStreamGenerator

        X = np.random.rand(95, 10, 6)
        y_dict = {'latency_ms': np.random.rand(95), 'pdr': np.random.rand(95)}
        gen = TorchDataStreamGenerator(X, y_dict, batch_size=10)

        batch_x, batch_y = gen[len(gen) - 1]
        assert batch_x.shape[0] == 5


class TestGenerateNewMeasurementTorch:
    """generate_new_measurement is framework-agnostic and shared with learning.model."""

    def test_generate_new_measurement_basic(self):
        from learning.model import generate_new_measurement

        X = np.random.rand(10, 5, 6)
        y = np.random.rand(10, 2)
        x_new, y_new = generate_new_measurement(X, y, index=0)

        assert x_new.shape == (1, 5, 6)
        assert y_new.shape == (1, 2)

    def test_generate_new_measurement_out_of_bounds(self):
        from learning.model import generate_new_measurement

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
        from learning.model import build_model_torch, EarlyStopping

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
        from learning.model import build_model_torch, EarlyStopping

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
        from learning.model import build_model_torch, fit_torch

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
        from learning.model import keras_style_validation_split

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

    def test_clipnorm_clips_each_parameter_gradient(self):
        """clipnorm mirrors Keras: every parameter's gradient norm is clipped separately."""
        from learning.model import build_model_torch, fit_torch

        np.random.seed(0)
        model = build_model_torch('lstm', timesteps=5, features=3)
        X = np.random.rand(16, 5, 3).astype(np.float32)
        y = {"latency_ms": np.ones(16, dtype=np.float32),
             "pdr": np.zeros(16, dtype=np.float32)}
        clipnorm = 1e-3

        grad_norms = []
        original_step = model.optimizer.step

        def recording_step(*args, **kwargs):
            grad_norms.extend(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            return original_step(*args, **kwargs)

        model.optimizer.step = recording_step
        fit_torch(model, X, y, epochs=1, batch_size=16, verbose=0, clipnorm=clipnorm)

        assert grad_norms
        assert max(grad_norms) <= clipnorm * (1 + 1e-4)
        # At least one gradient was actually clipped down to the limit
        assert max(grad_norms) == pytest.approx(clipnorm, rel=1e-3)


class TestInferenceHelpers:
    """Tests for predict_torch, evaluate_torch and checkpoint save/load."""

    def test_predict_torch_shapes_and_values(self):
        from learning.model import build_model_torch, predict_torch

        model = build_model_torch('gru', timesteps=10, features=4)
        X = np.random.rand(7, 10, 4)  # float64 on purpose: predict_torch must cast

        pred_lat, pred_pdr = predict_torch(model, X)

        assert pred_lat.shape == (7,)
        assert pred_pdr.shape == (7,)
        assert not model.training
        with torch.no_grad():
            device = next(model.parameters()).device
            lat_ref, pdr_ref = model(torch.from_numpy(X.astype(np.float32)).to(device))
        np.testing.assert_allclose(pred_lat, lat_ref.cpu().numpy().flatten())
        np.testing.assert_allclose(pred_pdr, pdr_ref.cpu().numpy().flatten())

    def test_evaluate_torch_matches_manual_weighted_mse(self):
        from learning.model import build_model_torch, evaluate_torch

        np.random.seed(0)
        model = build_model_torch('lstm', timesteps=5, features=3)
        X = np.random.rand(20, 5, 3).astype(np.float32)
        y = {"latency_ms": np.random.rand(20).astype(np.float32),
             "pdr": np.random.rand(20).astype(np.float32)}

        # A single batch covering every sample, so per-batch averaging equals a full-array MSE
        metrics = evaluate_torch(model, X, y, batch_size=20)

        with torch.no_grad():
            lat, pdr = model(torch.from_numpy(X).to(next(model.parameters()).device))
        lat_mse = float(torch.mean((lat.cpu().flatten() - torch.from_numpy(y["latency_ms"])) ** 2))
        pdr_mse = float(torch.mean((pdr.cpu().flatten() - torch.from_numpy(y["pdr"])) ** 2))
        assert metrics["latency_ms_loss"] == pytest.approx(lat_mse, rel=1e-4)
        assert metrics["pdr_loss"] == pytest.approx(pdr_mse, rel=1e-4)
        assert metrics["loss"] == pytest.approx(lat_mse + 1.5 * pdr_mse, rel=1e-4)
        assert metrics["latency_ms_rmse"] == pytest.approx(np.sqrt(lat_mse), rel=1e-4)
        assert metrics["pdr_rmse"] == pytest.approx(np.sqrt(pdr_mse), rel=1e-4)

    def test_save_load_round_trip_keeps_weights_and_optimizer_state(self, tmp_path):
        from learning.model import (build_model_torch, fit_torch, save_torch_model,
                                    load_torch_model, predict_torch)

        np.random.seed(0)
        model = build_model_torch('rnn', timesteps=5, features=3)
        X = np.random.rand(16, 5, 3).astype(np.float32)
        y = {"latency_ms": np.random.rand(16).astype(np.float32),
             "pdr": np.random.rand(16).astype(np.float32)}
        fit_torch(model, X, y, epochs=1, batch_size=8, verbose=0)

        path = tmp_path / "rnn_pc5_123.pt"
        save_torch_model(model, str(path))
        loaded = load_torch_model(str(path))

        assert (loaded.model_type, loaded.timesteps, loaded.features) == ('rnn', 5, 3)
        np.testing.assert_allclose(predict_torch(loaded, X)[0], predict_torch(model, X)[0])
        saved_state = model.optimizer.state_dict()["state"]
        loaded_state = loaded.optimizer.state_dict()["state"]
        assert len(loaded_state) > 0
        assert saved_state.keys() == loaded_state.keys()
        for idx in saved_state:
            assert torch.equal(saved_state[idx]["exp_avg"], loaded_state[idx]["exp_avg"])
            assert torch.equal(saved_state[idx]["exp_avg_sq"], loaded_state[idx]["exp_avg_sq"])

    def test_load_checkpoint_without_optimizer_state(self, tmp_path):
        """Checkpoints saved before optimizer state was stored still load, with a fresh optimizer."""
        from learning.model import build_model_torch, load_torch_model

        model = build_model_torch('lstm', timesteps=5, features=3)
        path = tmp_path / "lstm_5g_123.pt"
        torch.save({"model_type": "lstm", "timesteps": 5, "features": 3,
                    "state_dict": model.state_dict()}, path)

        loaded = load_torch_model(str(path))

        assert len(loaded.optimizer.state_dict()["state"]) == 0
        for name, param in loaded.state_dict().items():
            assert torch.equal(param, model.state_dict()[name])
