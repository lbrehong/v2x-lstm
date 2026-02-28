"""
Tests for config.py - scaler factories and configuration constants.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MIN_LAT, MAX_LAT, MIN_LON, MAX_LON,
    MIN_LATENCY, MAX_LATENCY, MIN_PDR, MAX_PDR,
    MIN_SINR_5G, MAX_SINR_5G, MIN_RSRP_5G, MAX_RSRP_5G,
    MIN_RSRP_DSRC, MAX_RSRP_DSRC,
    FEATURE_COLS, TARGET_COLS, FEATURES_COUNT,
    create_gps_scaler, create_latency_scaler,
    create_pdr_scaler,
    create_throughput_scaler, create_sinr_5g_scaler,
    create_rsrp_5g_scaler, create_rsrp_dsrc_scaler,
    create_all_scalers,
)


class TestConfigConstants:
    """Tests for configuration constants validity."""

    def test_gps_bounds_valid(self):
        """GPS bounds should be valid Toulouse coordinates."""
        assert MIN_LAT < MAX_LAT, "MIN_LAT should be less than MAX_LAT"
        assert MIN_LON < MAX_LON, "MIN_LON should be less than MAX_LON"
        # Toulouse is roughly at 43.6 N, 1.44 E
        assert 43.0 < MIN_LAT < 44.0
        assert 43.0 < MAX_LAT < 44.0
        assert 1.0 < MIN_LON < 2.0
        assert 1.0 < MAX_LON < 2.0

    def test_latency_bounds_valid(self):
        """Latency bounds should cover all RATs (DSRC ~1ms to 5G ~400ms)."""
        assert MIN_LATENCY >= 0, "Minimum latency must be non-negative"
        assert MAX_LATENCY > MIN_LATENCY
        assert MAX_LATENCY <= 1000  # Upper bound covers 5G tail

    def test_pdr_bounds_valid(self):
        """PDR bounds should be [0, 1]."""
        assert MIN_PDR == 0.0
        assert MAX_PDR == 1.0

    def test_feature_cols_structure(self):
        """Feature columns should match expected RAT configurations."""
        assert '5g' in FEATURE_COLS
        assert 'pc5' in FEATURE_COLS
        assert 'dsrc' in FEATURE_COLS

        # All should have GPS coordinates
        for rat in ['5g', 'pc5', 'dsrc']:
            assert 'tx_latitude' in FEATURE_COLS[rat]
            assert 'tx_longitude' in FEATURE_COLS[rat]
            assert 'latency_ms' in FEATURE_COLS[rat]
            assert 'pdr' in FEATURE_COLS[rat]

        # 5G should have SINR and RSRP
        assert 'sinr' in FEATURE_COLS['5g']
        assert 'rsrp' in FEATURE_COLS['5g']

        # DSRC should have two RSRP values
        assert 'rsrp_1' in FEATURE_COLS['dsrc']
        assert 'rsrp_2' in FEATURE_COLS['dsrc']

    def test_features_count_matches_feature_cols(self):
        """FEATURES_COUNT should match length of FEATURE_COLS lists."""
        for rat in ['5g', 'pc5', 'dsrc']:
            assert FEATURES_COUNT[rat] == len(FEATURE_COLS[rat])

    def test_target_cols_structure(self):
        """Target columns should be latency and PDR."""
        assert 'latency_ms' in TARGET_COLS
        assert 'pdr' in TARGET_COLS
        assert len(TARGET_COLS) == 2


class TestGpsScaler:
    """Tests for GPS coordinate scaler."""

    def test_gps_scaler_creation(self):
        """GPS scaler should be created successfully."""
        scaler = create_gps_scaler()
        assert scaler is not None
        assert hasattr(scaler, 'transform')
        assert hasattr(scaler, 'inverse_transform')

    def test_gps_scaler_min_values(self):
        """Minimum GPS values should transform to 0."""
        scaler = create_gps_scaler()
        result = scaler.transform([[MIN_LAT, MIN_LON]])
        np.testing.assert_array_almost_equal(result, [[0.0, 0.0]], decimal=5)

    def test_gps_scaler_max_values(self):
        """Maximum GPS values should transform to 1."""
        scaler = create_gps_scaler()
        result = scaler.transform([[MAX_LAT, MAX_LON]])
        np.testing.assert_array_almost_equal(result, [[1.0, 1.0]], decimal=5)

    def test_gps_scaler_mid_values(self):
        """Mid-range GPS values should transform to 0.5."""
        scaler = create_gps_scaler()
        mid_lat = (MIN_LAT + MAX_LAT) / 2
        mid_lon = (MIN_LON + MAX_LON) / 2
        result = scaler.transform([[mid_lat, mid_lon]])
        np.testing.assert_array_almost_equal(result, [[0.5, 0.5]], decimal=5)

    def test_gps_scaler_inverse_transform(self):
        """Inverse transform should recover original values."""
        scaler = create_gps_scaler()
        original = np.array([[43.560, 1.467]])
        scaled = scaler.transform(original)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(original, recovered, decimal=6)

    def test_gps_scaler_batch_transform(self):
        """Scaler should handle batch transformations."""
        scaler = create_gps_scaler()
        coords = np.array([
            [43.558, 1.462],
            [43.560, 1.467],
            [43.564, 1.471]
        ])
        scaled = scaler.transform(coords)
        assert scaled.shape == coords.shape
        assert np.all(scaled >= 0)  # May exceed 1 for edge cases
        assert np.all(scaled <= 1)


class TestLatencyScaler:
    """Tests for latency scaler."""

    def test_latency_scaler_creation(self):
        """Latency scaler should be created successfully."""
        scaler = create_latency_scaler()
        assert scaler is not None

    def test_latency_scaler_min_value(self):
        """Minimum latency should transform to 0."""
        scaler = create_latency_scaler()
        result = scaler.transform([[MIN_LATENCY]])
        np.testing.assert_almost_equal(result[0][0], 0.0, decimal=5)

    def test_latency_scaler_max_value(self):
        """Maximum latency should transform to 1."""
        scaler = create_latency_scaler()
        result = scaler.transform([[MAX_LATENCY]])
        np.testing.assert_almost_equal(result[0][0], 1.0, decimal=5)

    def test_latency_scaler_inverse_transform(self):
        """Inverse transform should recover original values."""
        scaler = create_latency_scaler()
        original = np.array([[25.0]])
        scaled = scaler.transform(original)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(original, recovered, decimal=6)


class TestPerRatLatencyScaler:
    """Tests for per-RAT latency scalers."""

    def test_per_rat_scaler_creation(self):
        """Per-RAT latency scalers should be created for all RATs."""
        for rat in ("5g", "pc5", "dsrc"):
            scaler = create_latency_scaler(rat)
            assert scaler is not None

    def test_per_rat_scaler_bounds(self):
        """Per-RAT scalers should use tighter bounds than global."""
        global_scaler = create_latency_scaler()
        for rat in ("5g", "pc5", "dsrc"):
            rat_scaler = create_latency_scaler(rat)
            # 50ms should scale higher with per-RAT scaler (tighter bounds)
            global_val = global_scaler.transform([[50.0]])[0][0]
            rat_val = rat_scaler.transform([[50.0]])[0][0]
            assert rat_val >= global_val

    def test_per_rat_scaler_roundtrip(self):
        """Per-RAT scalers should correctly roundtrip values."""
        for rat in ("5g", "pc5", "dsrc"):
            scaler = create_latency_scaler(rat)
            original = np.array([[15.0]])
            scaled = scaler.transform(original)
            recovered = scaler.inverse_transform(scaled)
            np.testing.assert_array_almost_equal(original, recovered, decimal=6)

    def test_unknown_rat_falls_back(self):
        """Unknown RAT should fall back to global bounds."""
        scaler = create_latency_scaler("unknown")
        global_scaler = create_latency_scaler()
        val = scaler.transform([[50.0]])[0][0]
        global_val = global_scaler.transform([[50.0]])[0][0]
        assert abs(val - global_val) < 1e-6


class TestPdrScaler:
    """Tests for PDR scaler."""

    def test_pdr_scaler_creation(self):
        """PDR scaler should be created successfully."""
        scaler = create_pdr_scaler()
        assert scaler is not None

    def test_pdr_scaler_is_identity(self):
        """PDR scaler should be identity since PDR is already [0, 1]."""
        scaler = create_pdr_scaler()
        # Test that 0 maps to 0 and 1 maps to 1
        result_min = scaler.transform([[0.0]])
        result_max = scaler.transform([[1.0]])
        np.testing.assert_almost_equal(result_min[0][0], 0.0, decimal=5)
        np.testing.assert_almost_equal(result_max[0][0], 1.0, decimal=5)

    def test_pdr_scaler_preserves_values(self):
        """PDR values should be preserved since they're already in [0, 1]."""
        scaler = create_pdr_scaler()
        test_values = np.array([[0.5], [0.99], [0.75]])
        scaled = scaler.transform(test_values)
        np.testing.assert_array_almost_equal(test_values, scaled, decimal=6)


class TestSinrScaler:
    """Tests for 5G SINR scaler."""

    def test_sinr_scaler_creation(self):
        """SINR scaler should be created successfully."""
        scaler = create_sinr_5g_scaler()
        assert scaler is not None

    def test_sinr_scaler_bounds(self):
        """SINR bounds should map to [0, 1]."""
        scaler = create_sinr_5g_scaler()
        result_min = scaler.transform([[MIN_SINR_5G]])
        result_max = scaler.transform([[MAX_SINR_5G]])
        np.testing.assert_almost_equal(result_min[0][0], 0.0, decimal=5)
        np.testing.assert_almost_equal(result_max[0][0], 1.0, decimal=5)


class TestRsrpScalers:
    """Tests for RSRP scalers (5G and DSRC)."""

    def test_rsrp_5g_scaler_creation(self):
        """5G RSRP scaler should be created successfully."""
        scaler = create_rsrp_5g_scaler()
        assert scaler is not None

    def test_rsrp_5g_scaler_bounds(self):
        """5G RSRP bounds should map to [0, 1]."""
        scaler = create_rsrp_5g_scaler()
        result_min = scaler.transform([[MIN_RSRP_5G]])
        result_max = scaler.transform([[MAX_RSRP_5G]])
        np.testing.assert_almost_equal(result_min[0][0], 0.0, decimal=5)
        np.testing.assert_almost_equal(result_max[0][0], 1.0, decimal=5)

    def test_rsrp_dsrc_scaler_creation(self):
        """DSRC RSRP scaler should be created successfully."""
        scaler = create_rsrp_dsrc_scaler()
        assert scaler is not None

    def test_rsrp_dsrc_scaler_bounds(self):
        """DSRC RSRP bounds should map to [0, 1]."""
        scaler = create_rsrp_dsrc_scaler()
        result_min = scaler.transform([[MIN_RSRP_DSRC]])
        result_max = scaler.transform([[MAX_RSRP_DSRC]])
        np.testing.assert_almost_equal(result_min[0][0], 0.0, decimal=5)
        np.testing.assert_almost_equal(result_max[0][0], 1.0, decimal=5)


class TestAllScalersFactory:
    """Tests for the create_all_scalers factory function."""

    def test_all_scalers_creation(self):
        """All scalers dictionary should be created successfully."""
        scalers = create_all_scalers()
        assert scalers is not None
        assert isinstance(scalers, dict)

    def test_all_scalers_keys(self):
        """All expected scaler keys should be present."""
        scalers = create_all_scalers()
        expected_keys = [
            'tx_latitude', 'tx_longitude',
            'latency_ms', 'throughput', 'pdr',
            'sinr', 'rsrp', 'rsrp_1', 'rsrp_2'
        ]
        for key in expected_keys:
            assert key in scalers, f"Missing scaler for {key}"

    def test_gps_scalers_share_instance(self):
        """tx_latitude and tx_longitude should share the same scaler."""
        scalers = create_all_scalers()
        assert scalers['tx_latitude'] is scalers['tx_longitude']

    def test_all_scalers_have_transform(self):
        """All scalers should have transform method."""
        scalers = create_all_scalers()
        for name, scaler in scalers.items():
            assert hasattr(scaler, 'transform'), f"Scaler {name} missing transform"
            assert hasattr(scaler, 'inverse_transform'), f"Scaler {name} missing inverse_transform"
