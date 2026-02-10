"""
Tests for learning/model.py - Neural network model definitions and data generators.
"""
import pytest
import numpy as np
import keras
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBuildModel:
    """Tests for the build_model function."""

    def test_build_lstm_model(self):
        """LSTM model should be built with correct architecture."""
        from learning.model import build_model

        model = build_model('lstm', timesteps=10, features=6)

        assert model is not None
        # Check input shape
        assert model.input_shape == (None, 10, 6)
        # Check outputs
        assert len(model.outputs) == 2

    def test_build_gru_model(self):
        """GRU model should be built with correct architecture."""
        from learning.model import build_model

        model = build_model('gru', timesteps=10, features=6)

        assert model is not None
        assert model.input_shape == (None, 10, 6)
        assert len(model.outputs) == 2

    def test_build_rnn_model(self):
        """SimpleRNN model should be built with correct architecture."""
        from learning.model import build_model

        model = build_model('rnn', timesteps=10, features=6)

        assert model is not None
        assert model.input_shape == (None, 10, 6)
        assert len(model.outputs) == 2

    def test_build_model_invalid_type(self):
        """Invalid model type should raise ValueError."""
        from learning.model import build_model

        with pytest.raises(ValueError, match="Unknown model type"):
            build_model('transformer', timesteps=10, features=6)

    def test_model_output_names(self):
        """Model outputs should be named 'latency_ms' and 'pdr'."""
        from learning.model import build_model

        model = build_model('lstm', timesteps=10, features=6)

        # Check output layer names (the Dense layers that produce outputs)
        output_layer_names = [layer.name for layer in model.layers if isinstance(layer, keras.layers.Dense)]
        # Check that latency_ms and pdr output layers exist
        assert any('latency' in name for name in output_layer_names)
        assert any('pdr' in name for name in output_layer_names)

    def test_model_different_features(self):
        """Model should handle different feature counts."""
        from learning.model import build_model

        # PC5 has 4 features
        model_pc5 = build_model('lstm', timesteps=10, features=4)
        assert model_pc5.input_shape == (None, 10, 4)

        # 5G has 6 features
        model_5g = build_model('lstm', timesteps=10, features=6)
        assert model_5g.input_shape == (None, 10, 6)

    def test_model_prediction_shape(self):
        """Model predictions should have correct shape."""
        from learning.model import build_model

        model = build_model('lstm', timesteps=10, features=6)

        # Create dummy input
        dummy_input = np.random.rand(5, 10, 6)
        predictions = model.predict(dummy_input, verbose=0)

        # Should return list of 2 outputs
        assert len(predictions) == 2
        assert predictions[0].shape == (5, 1)  # latency
        assert predictions[1].shape == (5, 1)  # pdr


class TestDataStreamGenerator:
    """Tests for the DataStreamGenerator class."""

    def test_generator_initialization(self):
        """Generator should initialize with correct attributes."""
        from learning.model import DataStreamGenerator

        X = np.random.rand(100, 10, 6)
        y_dict = {
            'latency_ms': np.random.rand(100),
            'pdr': np.random.rand(100)
        }

        gen = DataStreamGenerator(X, y_dict, batch_size=10)

        assert gen.X_data is X
        assert gen.y_data_dict is y_dict
        assert gen.batch_size == 10

    def test_generator_len(self):
        """Generator length should be correct number of batches."""
        from learning.model import DataStreamGenerator

        X = np.random.rand(100, 10, 6)
        y_dict = {
            'latency_ms': np.random.rand(100),
            'pdr': np.random.rand(100)
        }

        # Exact division
        gen = DataStreamGenerator(X, y_dict, batch_size=10)
        assert len(gen) == 10

        # With remainder
        gen2 = DataStreamGenerator(X, y_dict, batch_size=30)
        assert len(gen2) == 4  # ceil(100/30) = 4

    def test_generator_getitem(self):
        """Generator should return correct batch shapes."""
        from learning.model import DataStreamGenerator

        np.random.seed(42)
        X = np.random.rand(100, 10, 6)
        y_dict = {
            'latency_ms': np.random.rand(100),
            'pdr': np.random.rand(100)
        }

        gen = DataStreamGenerator(X, y_dict, batch_size=10)

        batch_x, batch_y = gen[0]

        assert batch_x.shape == (10, 10, 6)
        assert 'latency_ms' in batch_y
        assert 'pdr' in batch_y
        assert batch_y['latency_ms'].shape == (10,)
        assert batch_y['pdr'].shape == (10,)

    def test_generator_last_batch(self):
        """Last batch may have fewer samples."""
        from learning.model import DataStreamGenerator

        X = np.random.rand(95, 10, 6)
        y_dict = {
            'latency_ms': np.random.rand(95),
            'pdr': np.random.rand(95)
        }

        gen = DataStreamGenerator(X, y_dict, batch_size=10)

        # Last batch should have 5 samples
        batch_x, batch_y = gen[len(gen) - 1]
        assert batch_x.shape[0] == 5

    def test_generator_batch_values(self):
        """Generator should return correct values."""
        from learning.model import DataStreamGenerator

        X = np.arange(30).reshape(3, 10, 1)
        y_dict = {
            'latency_ms': np.array([100, 200, 300]),
            'pdr': np.array([0.9, 0.95, 0.99])
        }

        gen = DataStreamGenerator(X, y_dict, batch_size=2)

        batch_x, batch_y = gen[0]

        np.testing.assert_array_equal(batch_x[0], X[0])
        np.testing.assert_array_equal(batch_x[1], X[1])
        assert batch_y['latency_ms'][0] == 100
        assert batch_y['latency_ms'][1] == 200


class TestGenerateNewMeasurement:
    """Tests for the generate_new_measurement function."""

    def test_generate_new_measurement_basic(self):
        """Should extract and reshape single sample correctly."""
        from learning.model import generate_new_measurement

        X = np.random.rand(10, 5, 6)
        y = np.random.rand(10, 2)

        x_new, y_new = generate_new_measurement(X, y, index=0)

        assert x_new.shape == (1, 5, 6)
        assert y_new.shape == (1, 2)
        np.testing.assert_array_equal(x_new[0], X[0])
        np.testing.assert_array_equal(y_new[0], y[0])

    def test_generate_new_measurement_middle_index(self):
        """Should extract correct sample from middle of array."""
        from learning.model import generate_new_measurement

        X = np.arange(30).reshape(3, 10, 1)
        y = np.array([[1, 2], [3, 4], [5, 6]])

        x_new, y_new = generate_new_measurement(X, y, index=1)

        np.testing.assert_array_equal(x_new[0], X[1])
        np.testing.assert_array_equal(y_new[0], y[1])

    def test_generate_new_measurement_out_of_bounds(self):
        """Should raise IndexError for out of bounds index."""
        from learning.model import generate_new_measurement

        X = np.random.rand(5, 10, 6)
        y = np.random.rand(5, 2)

        with pytest.raises(IndexError, match="No more new measurements"):
            generate_new_measurement(X, y, index=5)

        with pytest.raises(IndexError, match="No more new measurements"):
            generate_new_measurement(X, y, index=10)


class TestRmseMetric:
    """Tests for the custom RMSE metric."""

    def test_rmse_identical_values(self):
        """RMSE should be 0 for identical predictions and targets."""
        from learning.model import rmse
        import tensorflow as tf

        y_true = tf.constant([1.0, 2.0, 3.0])
        y_pred = tf.constant([1.0, 2.0, 3.0])

        result = rmse(y_true, y_pred)

        assert float(result) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_known_value(self):
        """RMSE should match expected value for known inputs."""
        from learning.model import rmse
        import tensorflow as tf

        y_true = tf.constant([1.0, 2.0, 3.0, 4.0])
        y_pred = tf.constant([2.0, 3.0, 4.0, 5.0])  # All off by 1

        result = rmse(y_true, y_pred)

        # RMSE = sqrt(mean(1^2)) = 1.0
        assert float(result) == pytest.approx(1.0, abs=1e-6)

    def test_rmse_varied_errors(self):
        """RMSE should be computed correctly with varied errors."""
        from learning.model import rmse
        import tensorflow as tf

        y_true = tf.constant([0.0, 0.0, 0.0, 0.0])
        y_pred = tf.constant([1.0, 2.0, 3.0, 4.0])

        result = rmse(y_true, y_pred)

        # MSE = (1 + 4 + 9 + 16) / 4 = 30/4 = 7.5
        # RMSE = sqrt(7.5) = 2.7386...
        expected = np.sqrt(7.5)
        assert float(result) == pytest.approx(expected, abs=1e-5)
