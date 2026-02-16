"""
Tests for selection/file_integration.py - File-based integration helper functions.

Tests cover:
- _safe_float helper function for safe type conversion
- _row_to_network_state for DataFrame row conversion
- Various column naming conventions handling
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selection.file_integration import _safe_float, _row_to_network_state
from api_types import NetworkState


class TestSafeFloat:
    """Tests for _safe_float helper function."""

    def test_safe_float_with_int(self):
        """_safe_float should convert int to float."""
        assert _safe_float(42) == 42.0
        assert isinstance(_safe_float(42), float)

    def test_safe_float_with_float(self):
        """_safe_float should preserve float values."""
        assert _safe_float(3.14) == 3.14
        assert _safe_float(0.0) == 0.0
        assert _safe_float(-10.5) == -10.5

    def test_safe_float_with_string_number(self):
        """_safe_float should convert string numbers to float."""
        assert _safe_float("3.14") == 3.14
        assert _safe_float("42") == 42.0
        assert _safe_float("-10.5") == -10.5

    def test_safe_float_with_none(self):
        """_safe_float should return None for None input."""
        assert _safe_float(None) is None

    def test_safe_float_with_nan(self):
        """_safe_float should return None for NaN values."""
        assert _safe_float(np.nan) is None
        assert _safe_float(float("nan")) is None

    def test_safe_float_with_pandas_na(self):
        """_safe_float should return None for pandas NA values."""
        assert _safe_float(pd.NA) is None
        assert _safe_float(pd.NaT) is None

    def test_safe_float_with_invalid_string(self):
        """_safe_float should return None for non-numeric strings."""
        assert _safe_float("invalid") is None
        assert _safe_float("") is None
        assert _safe_float("abc") is None

    def test_safe_float_with_list(self):
        """_safe_float returns None for list inputs (cannot convert to float)."""
        # Lists cannot be converted to float, so should return None
        assert _safe_float([1, 2, 3]) is None
        # Dict also returns None (caught by TypeError in float conversion)
        assert _safe_float({"a": 1}) is None

    def test_safe_float_with_inf(self):
        """_safe_float should handle infinity values."""
        assert _safe_float(float("inf")) == float("inf")
        assert _safe_float(float("-inf")) == float("-inf")


class TestRowToNetworkState:
    """Tests for _row_to_network_state DataFrame row conversion."""

    def test_basic_conversion(self):
        """Basic row with primary column names should convert correctly."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
            "latency_ms_dsrc": 12.0,
            "pdr_dsrc": 0.98,
            "latency_ms_pc5": 10.0,
            "pdr_pc5": 0.99,
            "latency_ms_5g": 15.0,
            "pdr_5g": 0.995,
        })
        state = _row_to_network_state(row)

        assert state.timestamp_ms == 1699999999000
        assert state.latitude == 43.560
        assert state.longitude == 1.467
        assert state.dsrc_latency_ms == 12.0
        assert state.dsrc_pdr == 0.98
        assert state.pc5_latency_ms == 10.0
        assert state.pc5_pdr == 0.99
        assert state.fiveg_latency_ms == 15.0
        assert state.fiveg_pdr == 0.995

    def test_tx_prefix_gps_columns(self):
        """Row with tx_ prefixed GPS columns should convert correctly."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "tx_latitude": 43.560,
            "tx_longitude": 1.467,
        })
        state = _row_to_network_state(row)

        assert state.latitude == 43.560
        assert state.longitude == 1.467

    def test_alternative_column_names(self):
        """Alternative column naming conventions should be handled."""
        row = pd.Series({
            "timestamp": 1699999999000,  # Alternative timestamp column
            "latitude": 43.560,
            "longitude": 1.467,
            "dsrc_latency_ms": 12.0,  # Alternative naming
            "dsrc_pdr": 0.98,
            "pc5_latency_ms": 10.0,
            "pc5_pdr": 0.99,
            "fiveg_latency_ms": 15.0,  # Alternative naming
            "fiveg_pdr": 0.995,
        })
        state = _row_to_network_state(row)

        assert state.timestamp_ms == 1699999999000
        assert state.dsrc_latency_ms == 12.0
        assert state.pc5_latency_ms == 10.0
        assert state.fiveg_latency_ms == 15.0

    def test_time_ms_column(self):
        """Row with time_ms column should convert correctly."""
        row = pd.Series({
            "time_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
        })
        state = _row_to_network_state(row)
        assert state.timestamp_ms == 1699999999000

    def test_dsrc_rsrp_values(self):
        """DSRC RSRP values should be extracted correctly."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
            "rsrp_1": -85.0,
            "rsrp_2": -90.0,
        })
        state = _row_to_network_state(row)

        assert state.dsrc_rsrp_1 == -85.0
        assert state.dsrc_rsrp_2 == -90.0

    def test_dsrc_rsrp_alternative_names(self):
        """Alternative DSRC RSRP column names should be handled."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
            "dsrc_rsrp_1": -85.0,
            "dsrc_rsrp_2": -90.0,
        })
        state = _row_to_network_state(row)

        assert state.dsrc_rsrp_1 == -85.0
        assert state.dsrc_rsrp_2 == -90.0

    def test_5g_signal_values(self):
        """5G signal quality values should be extracted correctly."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
            "sinr": 25.0,
            "rsrp": -95.0,
        })
        state = _row_to_network_state(row)

        assert state.fiveg_sinr == 25.0
        assert state.fiveg_rsrp == -95.0

    def test_5g_signal_alternative_names(self):
        """Alternative 5G signal column names should be handled."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
            "fiveg_sinr": 25.0,
            "fiveg_rsrp": -95.0,
        })
        state = _row_to_network_state(row)

        assert state.fiveg_sinr == 25.0
        assert state.fiveg_rsrp == -95.0

    def test_5g_latency_fallback(self):
        """5G latency should fallback to generic latency_ms column."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
            "latency_ms": 15.0,  # Generic column, used as 5G fallback
            "pdr": 0.995,  # Generic column, used as 5G fallback
        })
        state = _row_to_network_state(row)

        assert state.fiveg_latency_ms == 15.0
        assert state.fiveg_pdr == 0.995

    def test_missing_optional_fields(self):
        """Missing optional fields should result in None values."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
        })
        state = _row_to_network_state(row)

        assert state.dsrc_latency_ms is None
        assert state.dsrc_pdr is None
        assert state.pc5_latency_ms is None
        assert state.pc5_pdr is None
        assert state.fiveg_latency_ms is None
        assert state.fiveg_pdr is None
        assert state.fiveg_sinr is None
        assert state.fiveg_rsrp is None

    def test_nan_values_converted_to_none(self):
        """NaN values in DataFrame should be converted to None."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
            "latency_ms_dsrc": np.nan,
            "pdr_dsrc": np.nan,
        })
        state = _row_to_network_state(row)

        assert state.dsrc_latency_ms is None
        assert state.dsrc_pdr is None

    def test_default_gps_when_missing(self):
        """GPS should default to 0.0 when columns are missing entirely."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
        })
        state = _row_to_network_state(row)

        assert state.latitude == 0.0
        assert state.longitude == 0.0

    def test_timestamp_default_when_missing(self):
        """Timestamp should default to 0 when all timestamp columns missing."""
        row = pd.Series({
            "latitude": 43.560,
            "longitude": 1.467,
        })
        state = _row_to_network_state(row)
        assert state.timestamp_ms == 0

    def test_complete_row_all_rats(self):
        """Complete row with all RAT data should convert correctly."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "tx_latitude": 43.560123,
            "tx_longitude": 1.467456,
            # DSRC
            "latency_ms_dsrc": 12.5,
            "pdr_dsrc": 0.98,
            "rsrp_1": -85.0,
            "rsrp_2": -90.0,
            # PC5
            "latency_ms_pc5": 10.0,
            "pdr_pc5": 0.99,
            # 5G
            "latency_ms_5g": 15.0,
            "pdr_5g": 0.995,
            "sinr": 25.0,
            "rsrp": -95.0,
        })
        state = _row_to_network_state(row)

        # Verify all fields
        assert state.timestamp_ms == 1699999999000
        assert state.latitude == 43.560123
        assert state.longitude == 1.467456
        assert state.dsrc_latency_ms == 12.5
        assert state.dsrc_pdr == 0.98
        assert state.dsrc_rsrp_1 == -85.0
        assert state.dsrc_rsrp_2 == -90.0
        assert state.pc5_latency_ms == 10.0
        assert state.pc5_pdr == 0.99
        assert state.fiveg_latency_ms == 15.0
        assert state.fiveg_pdr == 0.995
        assert state.fiveg_sinr == 25.0
        assert state.fiveg_rsrp == -95.0

    def test_result_is_network_state(self):
        """_row_to_network_state should return a NetworkState instance."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
        })
        state = _row_to_network_state(row)

        assert isinstance(state, NetworkState)

    def test_string_values_converted(self):
        """String values in numeric fields should be converted to float."""
        row = pd.Series({
            "timestamp_ms": "1699999999000",
            "latitude": "43.560",
            "longitude": "1.467",
            "latency_ms_dsrc": "12.5",
            "pdr_dsrc": "0.98",
        })
        state = _row_to_network_state(row)

        assert state.latitude == 43.560
        assert state.longitude == 1.467
        assert state.dsrc_latency_ms == 12.5
        assert state.dsrc_pdr == 0.98


class TestRowToNetworkStateEdgeCases:
    """Edge case tests for _row_to_network_state."""

    def test_mixed_nan_and_valid_values(self):
        """Row with mixed NaN and valid values should handle both correctly."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
            "latency_ms_dsrc": np.nan,
            "pdr_dsrc": 0.98,  # Valid
            "latency_ms_pc5": 10.0,  # Valid
            "pdr_pc5": np.nan,
        })
        state = _row_to_network_state(row)

        assert state.dsrc_latency_ms is None
        assert state.dsrc_pdr == 0.98
        assert state.pc5_latency_ms == 10.0
        assert state.pc5_pdr is None

    def test_empty_series(self):
        """Empty Series should produce state with defaults."""
        row = pd.Series({})
        state = _row_to_network_state(row)

        assert state.timestamp_ms == 0
        assert state.latitude == 0.0
        assert state.longitude == 0.0

    def test_negative_latency_values(self):
        """Negative latency values should be preserved (for error detection)."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
            "latency_ms_dsrc": -5.0,  # Invalid but should be preserved
        })
        state = _row_to_network_state(row)
        assert state.dsrc_latency_ms == -5.0

    def test_pdr_outside_bounds(self):
        """PDR values outside [0,1] should be preserved (for error detection)."""
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
            "pdr_dsrc": 1.5,  # Invalid but should be preserved
        })
        state = _row_to_network_state(row)
        assert state.dsrc_pdr == 1.5

    def test_very_large_timestamp(self):
        """Very large timestamp values should be handled."""
        row = pd.Series({
            "timestamp_ms": 9999999999999,
            "latitude": 43.560,
            "longitude": 1.467,
        })
        state = _row_to_network_state(row)
        assert state.timestamp_ms == 9999999999999

    def test_zero_timestamp(self):
        """Zero timestamp should be allowed."""
        row = pd.Series({
            "timestamp_ms": 0,
            "latitude": 43.560,
            "longitude": 1.467,
        })
        state = _row_to_network_state(row)
        assert state.timestamp_ms == 0

    def test_dataframe_row_from_iterrows(self):
        """Row obtained from DataFrame.iterrows() should work correctly."""
        df = pd.DataFrame({
            "timestamp_ms": [1699999999000],
            "latitude": [43.560],
            "longitude": [1.467],
            "latency_ms_dsrc": [12.0],
            "pdr_dsrc": [0.98],
        })
        for _, row in df.iterrows():
            state = _row_to_network_state(row)
            assert state.timestamp_ms == 1699999999000
            assert state.dsrc_latency_ms == 12.0

    def test_priority_of_column_names(self):
        """When multiple column names exist, primary should take priority."""
        # tx_latitude should take priority over latitude
        row = pd.Series({
            "timestamp_ms": 1699999999000,
            "tx_latitude": 43.560,
            "latitude": 43.999,  # Should be ignored
            "tx_longitude": 1.467,
            "longitude": 1.999,  # Should be ignored
        })
        state = _row_to_network_state(row)

        assert state.latitude == 43.560
        assert state.longitude == 1.467
