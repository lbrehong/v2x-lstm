"""
Tests for learning/data_preprocessing.py - PDR computation, sequence generation, and LSTM input preprocessing.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning.data_preprocessing import compute_pdr_rolling, generate_lstm_sequences


class TestComputePdrRolling:
    """Tests for the rolling PDR computation function."""

    def test_pdr_rolling_basic(self):
        """Basic PDR computation with regular packet intervals."""
        # 50 packets/sec for 1 second = 50 packets expected
        # If we receive all packets, PDR = 1.0
        df = pd.DataFrame({
            'tx_timestamp_ms': np.arange(0, 1000, 20)  # 50 packets in 1 sec
        })
        result = compute_pdr_rolling(df, 'tx_timestamp_ms', window_size=1, packet_interval_ms=20)

        assert 'pdr' in result.columns
        assert len(result) == len(df)
        # Last packet should see all 50 packets in window -> PDR = 1.0
        assert result['pdr'].iloc[-1] == 1.0

    def test_pdr_rolling_half_rate(self):
        """PDR computation with half the expected packets."""
        # 25 packets in 1 sec when expecting 50 -> PDR = 0.5
        df = pd.DataFrame({
            'tx_timestamp_ms': np.arange(0, 1000, 40)  # 25 packets in 1 sec
        })
        result = compute_pdr_rolling(df, 'tx_timestamp_ms', window_size=1, packet_interval_ms=20)

        assert 'pdr' in result.columns
        # Should be approximately 0.5 PDR
        final_pdr = result['pdr'].iloc[-1]
        assert 0.45 <= final_pdr <= 0.55, f"Expected ~0.5 PDR, got {final_pdr}"

    def test_pdr_rolling_capped_at_one(self):
        """PDR should be capped at 1.0 even with extra packets."""
        # More packets than expected due to timing
        df = pd.DataFrame({
            'tx_timestamp_ms': np.arange(0, 1000, 10)  # 100 packets in 1 sec
        })
        result = compute_pdr_rolling(df, 'tx_timestamp_ms', window_size=1, packet_interval_ms=20)

        # All PDR values should be <= 1.0
        assert result['pdr'].max() <= 1.0

    def test_pdr_rolling_window_size(self):
        """Different window sizes should affect PDR calculation."""
        df = pd.DataFrame({
            'tx_timestamp_ms': np.arange(0, 5000, 20)  # 5 seconds of data
        })

        # 1-second window
        result_1s = compute_pdr_rolling(df.copy(), 'tx_timestamp_ms', window_size=1)
        # 2-second window
        result_2s = compute_pdr_rolling(df.copy(), 'tx_timestamp_ms', window_size=2)

        # Both should eventually reach PDR = 1.0 for consistent data
        assert result_1s['pdr'].iloc[-1] == 1.0
        assert result_2s['pdr'].iloc[-1] == 1.0

    def test_pdr_rolling_preserves_index(self):
        """PDR computation should preserve DataFrame index."""
        df = pd.DataFrame({
            'tx_timestamp_ms': np.arange(0, 500, 20)
        }, index=range(100, 125))  # Custom index

        result = compute_pdr_rolling(df, 'tx_timestamp_ms')
        assert list(result.index) == list(range(100, 125))

    def test_pdr_rolling_empty_dataframe(self):
        """Empty DataFrame should return empty DataFrame with pdr column."""
        df = pd.DataFrame({'tx_timestamp_ms': []})
        result = compute_pdr_rolling(df, 'tx_timestamp_ms')

        assert 'pdr' in result.columns
        assert len(result) == 0

    def test_pdr_rolling_single_packet(self):
        """Single packet should have low PDR."""
        df = pd.DataFrame({'tx_timestamp_ms': [0]})
        result = compute_pdr_rolling(df, 'tx_timestamp_ms', window_size=1, packet_interval_ms=20)

        assert 'pdr' in result.columns
        assert len(result) == 1
        # 1 packet / 50 expected = 0.02
        assert result['pdr'].iloc[0] == pytest.approx(0.02, rel=0.01)

    def test_pdr_rolling_creates_timestamp_sec_column(self):
        """Function should create tx_timestamp_sec intermediate column."""
        df = pd.DataFrame({
            'tx_timestamp_ms': [0, 20, 40, 60, 80]
        })
        result = compute_pdr_rolling(df, 'tx_timestamp_ms')

        assert 'tx_timestamp_sec' in result.columns
        # Check conversion is correct
        np.testing.assert_array_almost_equal(
            result['tx_timestamp_sec'].values,
            [0.0, 0.02, 0.04, 0.06, 0.08]
        )


class TestGenerateLstmSequences:
    """Tests for LSTM sequence generation."""

    def test_generate_sequences_basic(self):
        """Basic sequence generation with correct shapes."""
        data = pd.DataFrame({
            'feature1': np.arange(20),
            'feature2': np.arange(20, 40),
            'target1': np.arange(100, 120),
            'target2': np.arange(200, 220),
        })

        feature_cols = ['feature1', 'feature2']
        target_cols = ['target1', 'target2']
        seq_length = 5

        # Collect all batches
        all_x, all_y = [], []
        for x_batch, y_batch in generate_lstm_sequences(
            data, feature_cols, target_cols, seq_length, batch_size=10
        ):
            all_x.append(x_batch)
            all_y.append(y_batch)

        x_full = np.concatenate(all_x, axis=0)
        y_full = np.concatenate(all_y, axis=0)

        # Should have 15 sequences (20 - 5 = 15)
        assert x_full.shape == (15, 5, 2)  # (samples, timesteps, features)
        assert y_full.shape == (15, 2)  # (samples, targets)

    def test_generate_sequences_values(self):
        """Verify sequence values are correctly extracted."""
        data = pd.DataFrame({
            'feature': np.arange(10),
            'target': np.arange(100, 110),
        })

        feature_cols = ['feature']
        target_cols = ['target']
        seq_length = 3

        all_x, all_y = [], []
        for x_batch, y_batch in generate_lstm_sequences(
            data, feature_cols, target_cols, seq_length, batch_size=100
        ):
            all_x.append(x_batch)
            all_y.append(y_batch)

        x_full = np.concatenate(all_x, axis=0)
        y_full = np.concatenate(all_y, axis=0)

        # First sequence: [0, 1, 2] -> target at index 3 = 103
        np.testing.assert_array_equal(x_full[0].flatten(), [0, 1, 2])
        assert y_full[0, 0] == 103

        # Second sequence: [1, 2, 3] -> target at index 4 = 104
        np.testing.assert_array_equal(x_full[1].flatten(), [1, 2, 3])
        assert y_full[1, 0] == 104

    def test_generate_sequences_batch_size(self):
        """Different batch sizes should yield same total sequences."""
        data = pd.DataFrame({
            'feature': np.arange(50),
            'target': np.arange(100, 150),
        })

        feature_cols = ['feature']
        target_cols = ['target']
        seq_length = 10

        # Batch size 5
        total_x_5, total_y_5 = [], []
        for x_batch, y_batch in generate_lstm_sequences(
            data, feature_cols, target_cols, seq_length, batch_size=5
        ):
            total_x_5.append(x_batch)
            total_y_5.append(y_batch)

        # Batch size 20
        total_x_20, total_y_20 = [], []
        for x_batch, y_batch in generate_lstm_sequences(
            data, feature_cols, target_cols, seq_length, batch_size=20
        ):
            total_x_20.append(x_batch)
            total_y_20.append(y_batch)

        x_5 = np.concatenate(total_x_5, axis=0)
        x_20 = np.concatenate(total_x_20, axis=0)

        assert x_5.shape == x_20.shape
        np.testing.assert_array_equal(x_5, x_20)

    def test_generate_sequences_seq_length_equals_data(self):
        """When seq_length equals data length, should produce no sequences."""
        data = pd.DataFrame({
            'feature': np.arange(10),
            'target': np.arange(100, 110),
        })

        feature_cols = ['feature']
        target_cols = ['target']
        seq_length = 10  # Same as data length

        batches = list(generate_lstm_sequences(
            data, feature_cols, target_cols, seq_length, batch_size=100
        ))

        # Should produce no sequences
        assert len(batches) == 0

    def test_generate_sequences_multiple_features(self):
        """Sequence generation with multiple features."""
        n_samples = 30
        data = pd.DataFrame({
            'lat': np.linspace(43.55, 43.57, n_samples),
            'lon': np.linspace(1.46, 1.47, n_samples),
            'latency': np.random.uniform(5, 50, n_samples),
            'pdr': np.random.uniform(0.9, 1.0, n_samples),
        })

        feature_cols = ['lat', 'lon', 'latency', 'pdr']
        target_cols = ['latency', 'pdr']
        seq_length = 10

        all_x, all_y = [], []
        for x_batch, y_batch in generate_lstm_sequences(
            data, feature_cols, target_cols, seq_length, batch_size=100
        ):
            all_x.append(x_batch)
            all_y.append(y_batch)

        x_full = np.concatenate(all_x, axis=0)
        y_full = np.concatenate(all_y, axis=0)

        assert x_full.shape == (20, 10, 4)  # 30-10 = 20 sequences
        assert y_full.shape == (20, 2)


class TestPreprocessLstmInput:
    """Tests for the full LSTM preprocessing pipeline."""

    @pytest.fixture
    def valid_5g_df(self):
        """Create valid 5G DataFrame with all required columns."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'tx_latitude': np.linspace(43.556, 43.567, n),
            'tx_longitude': np.linspace(1.465, 1.471, n),
            'latency_ms': np.random.uniform(5, 45, n),
            'sinr': np.random.uniform(-5, 40, n),
            'rsrp': np.random.uniform(-120, -70, n),
            'pdr': np.random.uniform(0.85, 1.0, n),
        })

    @pytest.fixture
    def valid_pc5_df(self):
        """Create valid PC5 DataFrame with all required columns."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'tx_latitude': np.linspace(43.556, 43.567, n),
            'tx_longitude': np.linspace(1.465, 1.471, n),
            'latency_ms': np.random.uniform(5, 35, n),
            'pdr': np.random.uniform(0.9, 1.0, n),
        })

    @pytest.fixture
    def valid_dsrc_df(self):
        """Create valid DSRC DataFrame with all required columns."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'tx_latitude': np.linspace(43.556, 43.567, n),
            'tx_longitude': np.linspace(1.465, 1.471, n),
            'rsrp_1': np.random.uniform(-140, -50, n),
            'rsrp_2': np.random.uniform(-140, -50, n),
            'latency_ms': np.random.uniform(5, 30, n),
            'pdr': np.random.uniform(0.9, 1.0, n),
        })

    def test_preprocess_invalid_rat_raises_error(self, valid_5g_df):
        """Invalid RAT type should raise ValueError."""
        from learning.data_preprocessing import preprocess_lstm_input

        with pytest.raises(ValueError, match="Invalid RAT type"):
            preprocess_lstm_input(valid_5g_df, new=True, rat='invalid',
                                  target_cols=['latency_ms', 'pdr'], seq_length=10)

    def test_preprocess_5g_output_shapes(self, valid_5g_df):
        """5G preprocessing should produce correct output shapes."""
        from learning.data_preprocessing import preprocess_lstm_input

        X, y, scalers = preprocess_lstm_input(
            valid_5g_df, new=True, rat='5g',
            target_cols=['latency_ms', 'pdr'], seq_length=10
        )

        # X shape: (samples, timesteps, features) where features=6 for 5G
        assert X.ndim == 3
        assert X.shape[1] == 10  # timesteps
        assert X.shape[2] == 6  # 5G features

        # y shape: (samples, targets) where targets=2
        assert y.ndim == 2
        assert y.shape[1] == 2

        # X and y should have same number of samples
        assert X.shape[0] == y.shape[0]

    def test_preprocess_pc5_output_shapes(self, valid_pc5_df):
        """PC5 preprocessing should produce correct output shapes."""
        from learning.data_preprocessing import preprocess_lstm_input

        X, y, scalers = preprocess_lstm_input(
            valid_pc5_df, new=True, rat='pc5',
            target_cols=['latency_ms', 'pdr'], seq_length=10
        )

        assert X.shape[1] == 10  # timesteps
        assert X.shape[2] == 4  # PC5 features

    def test_preprocess_dsrc_output_shapes(self, valid_dsrc_df):
        """DSRC preprocessing should produce correct output shapes."""
        from learning.data_preprocessing import preprocess_lstm_input

        X, y, scalers = preprocess_lstm_input(
            valid_dsrc_df, new=True, rat='dsrc',
            target_cols=['latency_ms', 'pdr'], seq_length=10
        )

        assert X.shape[1] == 10  # timesteps
        assert X.shape[2] == 6  # DSRC features

    def test_preprocess_returns_scalers(self, valid_5g_df):
        """Preprocessing should return scalers dictionary."""
        from learning.data_preprocessing import preprocess_lstm_input

        X, y, scalers = preprocess_lstm_input(
            valid_5g_df, new=True, rat='5g',
            target_cols=['latency_ms', 'pdr'], seq_length=10
        )

        assert isinstance(scalers, dict)
        assert 'tx_latitude' in scalers
        assert 'latency_ms' in scalers
        assert 'pdr' in scalers

    def test_preprocess_normalized_values(self, valid_5g_df):
        """Preprocessed features should be normalized to [0, 1] range."""
        from learning.data_preprocessing import preprocess_lstm_input

        X, y, scalers = preprocess_lstm_input(
            valid_5g_df, new=True, rat='5g',
            target_cols=['latency_ms', 'pdr'], seq_length=10
        )

        # All X values should be in [0, 1] range (approximately)
        assert X.min() >= -0.1, "Features should be approximately >= 0"
        assert X.max() <= 1.1, "Features should be approximately <= 1"

    def test_preprocess_filters_invalid_gps(self):
        """Preprocessing should filter out rows with invalid GPS."""
        from learning.data_preprocessing import preprocess_lstm_input
        from config import MIN_LAT, MIN_LON

        df = pd.DataFrame({
            'tx_latitude': [MIN_LAT, 43.56, 43.57, MIN_LAT, 43.58],
            'tx_longitude': [MIN_LON, 1.465, 1.466, MIN_LON, 1.467],
            'latency_ms': [10, 15, 20, 25, 30],
            'sinr': [5, 15, 25, 30, 35],
            'rsrp': [-100, -95, -90, -85, -80],
            'pdr': [0.95, 0.96, 0.97, 0.98, 0.99],
        })

        # With seq_length=2, we need at least 3 valid rows
        X, y, scalers = preprocess_lstm_input(
            df, new=True, rat='5g',
            target_cols=['latency_ms', 'pdr'], seq_length=2
        )

        # Should have filtered out 2 invalid GPS rows, leaving 3
        # 3 rows - 2 seq_length = 1 sequence
        assert X.shape[0] == 1
