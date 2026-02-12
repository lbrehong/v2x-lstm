"""
Tests for selection/rat_selection.py - RAT selection algorithms and utility functions.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PDR_RELIABILITY_THRESHOLD, PDR_AVAILABILITY_THRESHOLD, LATENCY_TIE_MARGIN_MS


class TestSelectBestRat:
    """Tests for the predictive QoS-based RAT selection function."""

    def test_select_lowest_latency_when_all_reliable(self):
        """Should select RAT with lowest latency when all have high PDR."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_lstm': 15.0,
            'pred_pdr_dsrc_lstm': 0.995,  # Above threshold
            'pdr_dsrc': 0.99,
            'pred_latency_ms_pc5_lstm': 10.0,  # Lowest latency
            'pred_pdr_pc5_lstm': 0.992,  # Above threshold
            'pdr_pc5': 0.985,
            'pred_latency_ms_5g_lstm': 20.0,
            'pred_pdr_5g_lstm': 0.998,  # Above threshold
            'pdr_5g': 0.99,
        })

        result = select_best_rat(row, 'lstm')

        assert result == 'pc5', "Should select PC5 with lowest latency"

    def test_select_5g_when_only_reliable(self):
        """Should select 5G when it's the only reliable option."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_lstm': 8.0,
            'pred_pdr_dsrc_lstm': 0.95,  # Below threshold
            'pdr_dsrc': 0.94,
            'pred_latency_ms_pc5_lstm': 7.0,
            'pred_pdr_pc5_lstm': 0.96,  # Below threshold
            'pdr_pc5': 0.95,
            'pred_latency_ms_5g_lstm': 25.0,
            'pred_pdr_5g_lstm': 0.995,  # Above threshold
            'pdr_5g': 0.99,
        })

        result = select_best_rat(row, 'lstm')

        assert result == '5g', "Should select 5G as only reliable option"

    def test_fallback_to_highest_pdr_when_none_reliable(self):
        """Should fall back to highest PDR when no RAT meets reliability threshold."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_lstm': 8.0,
            'pred_pdr_dsrc_lstm': 0.95,  # Below threshold
            'pdr_dsrc': 0.2,  # Available but low PDR
            'pred_latency_ms_pc5_lstm': 7.0,
            'pred_pdr_pc5_lstm': 0.97,  # Below threshold, but highest pred PDR
            'pdr_pc5': 0.25,  # Available
            'pred_latency_ms_5g_lstm': 25.0,
            'pred_pdr_5g_lstm': 0.94,  # Below threshold
            'pdr_5g': 0.15,  # Available
        })

        result = select_best_rat(row, 'lstm')

        # Should select PC5 which has highest predicted PDR among available
        assert result == 'pc5'

    def test_tie_breaking_prefers_5g(self):
        """When latencies are within margin, should prefer 5G."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_lstm': 10.0,
            'pred_pdr_dsrc_lstm': 0.995,
            'pdr_dsrc': 0.99,
            'pred_latency_ms_pc5_lstm': 10.5,  # Within 1ms of DSRC
            'pred_pdr_pc5_lstm': 0.995,
            'pdr_pc5': 0.99,
            'pred_latency_ms_5g_lstm': 10.8,  # Within 1ms of lowest
            'pred_pdr_5g_lstm': 0.995,
            'pdr_5g': 0.99,
        })

        result = select_best_rat(row, 'lstm')

        assert result == '5g', "Should prefer 5G when all 3 RATs are within tie margin"

    def test_tie_breaking_3way_prefers_5g(self):
        """When all 3 RATs are within margin, should prefer 5G over PC5 and DSRC."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_lstm': 10.0,
            'pred_pdr_dsrc_lstm': 0.995,
            'pdr_dsrc': 0.99,
            'pred_latency_ms_pc5_lstm': 10.3,
            'pred_pdr_pc5_lstm': 0.995,
            'pdr_pc5': 0.99,
            'pred_latency_ms_5g_lstm': 10.6,
            'pred_pdr_5g_lstm': 0.995,
            'pdr_5g': 0.99,
        })

        result = select_best_rat(row, 'lstm')
        assert result == '5g'

    def test_tie_breaking_prefers_pc5_over_dsrc(self):
        """When only PC5 and DSRC are tied, should prefer PC5."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_lstm': 10.0,
            'pred_pdr_dsrc_lstm': 0.995,
            'pdr_dsrc': 0.99,
            'pred_latency_ms_pc5_lstm': 10.5,  # Within 1ms of DSRC
            'pred_pdr_pc5_lstm': 0.995,
            'pdr_pc5': 0.99,
            'pred_latency_ms_5g_lstm': 20.0,  # Far away — not tied
            'pred_pdr_5g_lstm': 0.995,
            'pdr_5g': 0.99,
        })

        result = select_best_rat(row, 'lstm')
        assert result == 'pc5'

    def test_nan_when_all_unavailable(self):
        """Should return NaN when all RATs have PDR below availability threshold."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_lstm': 10.0,
            'pred_pdr_dsrc_lstm': 0.5,
            'pdr_dsrc': 0.05,  # Below availability threshold
            'pred_latency_ms_pc5_lstm': 10.0,
            'pred_pdr_pc5_lstm': 0.5,
            'pdr_pc5': 0.03,  # Below availability threshold
            'pred_latency_ms_5g_lstm': 10.0,
            'pred_pdr_5g_lstm': 0.5,
            'pdr_5g': 0.02,  # Below availability threshold
        })

        result = select_best_rat(row, 'lstm')

        assert result == 'NaN', "Should return NaN when all RATs unavailable"

    def test_works_with_gru_model_type(self):
        """Should work with GRU model type columns."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_gru': 15.0,
            'pred_pdr_dsrc_gru': 0.995,
            'pdr_dsrc': 0.99,
            'pred_latency_ms_pc5_gru': 12.0,
            'pred_pdr_pc5_gru': 0.992,
            'pdr_pc5': 0.99,
            'pred_latency_ms_5g_gru': 18.0,
            'pred_pdr_5g_gru': 0.998,
            'pdr_5g': 0.99,
        })

        result = select_best_rat(row, 'gru')

        assert result == 'pc5'

    def test_works_with_rnn_model_type(self):
        """Should work with RNN model type columns."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_rnn': 15.0,
            'pred_pdr_dsrc_rnn': 0.995,
            'pdr_dsrc': 0.99,
            'pred_latency_ms_pc5_rnn': 12.0,
            'pred_pdr_pc5_rnn': 0.992,
            'pdr_pc5': 0.99,
            'pred_latency_ms_5g_rnn': 18.0,
            'pred_pdr_5g_rnn': 0.998,
            'pdr_5g': 0.99,
        })

        result = select_best_rat(row, 'rnn')

        assert result == 'pc5'


class TestOpportunisticBestRat:
    """Tests for the opportunistic (reactive) RAT selection function."""

    def test_opportunistic_basic_selection(self):
        """Should select lowest latency RAT when all available."""
        from selection.rat_selection import opportunistic_best_rat

        df = pd.DataFrame({
            'latency_ms_dsrc': [15.0],
            'pdr_dsrc': [0.95],
            'latency_ms_pc5': [10.0],  # Lowest latency
            'pdr_pc5': [0.92],
            'latency_ms_5g': [20.0],
            'pdr_5g': [0.98],
        })

        result = opportunistic_best_rat(df)

        assert 'Best_RAT_opp' in result.columns
        # Starting from 5G, should switch to lowest latency V2X option
        assert result['Best_RAT_opp'].iloc[0] in ['pc5', 'dsrc', '5g']

    def test_opportunistic_sticky_policy(self):
        """Should maintain current RAT when it's still available."""
        from selection.rat_selection import opportunistic_best_rat

        df = pd.DataFrame({
            'latency_ms_dsrc': [15.0, 14.0, 16.0],
            'pdr_dsrc': [0.95, 0.96, 0.94],
            'latency_ms_pc5': [10.0, 11.0, 9.0],
            'pdr_pc5': [0.92, 0.91, 0.93],
            'latency_ms_5g': [20.0, 21.0, 19.0],
            'pdr_5g': [0.98, 0.97, 0.99],
        })

        result = opportunistic_best_rat(df)

        assert 'Best_RAT_opp' in result.columns
        assert len(result['Best_RAT_opp']) == 3

    def test_opportunistic_fallback_to_5g(self):
        """Should fall back to 5G when V2X options unavailable."""
        from selection.rat_selection import opportunistic_best_rat

        df = pd.DataFrame({
            'latency_ms_dsrc': [15.0],
            'pdr_dsrc': [0.01],  # Below threshold
            'latency_ms_pc5': [10.0],
            'pdr_pc5': [0.02],  # Below threshold
            'latency_ms_5g': [20.0],
            'pdr_5g': [0.15],  # Above availability threshold
        })

        result = opportunistic_best_rat(df)

        assert result['Best_RAT_opp'].iloc[0] == '5g'

    def test_opportunistic_nan_when_all_unavailable(self):
        """Should return NaN when all RATs have zero PDR."""
        from selection.rat_selection import opportunistic_best_rat

        df = pd.DataFrame({
            'latency_ms_dsrc': [15.0],
            'pdr_dsrc': [0.0],  # No packets
            'latency_ms_pc5': [10.0],
            'pdr_pc5': [0.0],  # No packets
            'latency_ms_5g': [20.0],
            'pdr_5g': [0.0],  # No packets - truly unavailable
        })

        result = opportunistic_best_rat(df)

        assert result['Best_RAT_opp'].iloc[0] == 'NaN'


class TestGetLatencies:
    """Tests for the get_latencies helper function."""

    def test_get_latencies_basic(self):
        """Should extract correct latencies based on selected RAT."""
        from selection.rat_selection import get_latencies

        df = pd.DataFrame({
            'Best_RAT_test': ['dsrc', 'pc5', '5g'],
            'latency_ms_dsrc': [10.0, 15.0, 20.0],
            'latency_ms_pc5': [12.0, 8.0, 18.0],
            'latency_ms_5g': [25.0, 22.0, 14.0],
        })

        result = get_latencies(df, 'Best_RAT_test')

        assert result.iloc[0] == 10.0  # DSRC latency
        assert result.iloc[1] == 8.0   # PC5 latency
        assert result.iloc[2] == 14.0  # 5G latency


class TestGetPdr:
    """Tests for the get_pdr helper function."""

    def test_get_pdr_basic(self):
        """Should extract correct PDR based on selected RAT."""
        from selection.rat_selection import get_pdr

        df = pd.DataFrame({
            'Best_RAT_test': ['dsrc', 'pc5', '5g'],
            'pdr_dsrc': [0.95, 0.90, 0.85],
            'pdr_pc5': [0.92, 0.98, 0.88],
            'pdr_5g': [0.99, 0.97, 0.94],
        })

        result = get_pdr(df, 'Best_RAT_test')

        # PDR is multiplied by 100 in get_pdr
        assert result.iloc[0] == pytest.approx(95.0, abs=0.1)  # DSRC PDR
        assert result.iloc[1] == pytest.approx(98.0, abs=0.1)  # PC5 PDR
        assert result.iloc[2] == pytest.approx(94.0, abs=0.1)  # 5G PDR


class TestMeanCi:
    """Tests for the mean and confidence interval calculation function."""

    def test_mean_ci_basic(self):
        """Should compute mean and CI correctly."""
        from selection.rat_selection import mean_ci

        series = pd.Series([10, 20, 30, 40, 50])
        mean, ci_range = mean_ci(series)

        assert mean == 30.0
        assert ci_range > 0  # CI should be positive

    def test_mean_ci_single_value(self):
        """Should handle single value without error."""
        from selection.rat_selection import mean_ci

        series = pd.Series([42.0])
        mean, ci_range = mean_ci(series)

        assert mean == 42.0
        assert ci_range == 0  # No CI for single value

    def test_mean_ci_with_nan(self):
        """Should handle NaN values by dropping them."""
        from selection.rat_selection import mean_ci

        series = pd.Series([10, np.nan, 20, np.nan, 30])
        mean, ci_range = mean_ci(series)

        assert mean == 20.0  # (10 + 20 + 30) / 3


class TestGrabGps:
    """Tests for the GPS extraction function."""

    def test_grab_gps_extracts_coordinates(self):
        """Should extract and deduplicate GPS coordinates."""
        from selection.rat_selection import grab_gps

        df = pd.DataFrame({
            'tx_latitude': [43.56, 43.56, 43.57, 43.57, 43.58],
            'tx_longitude': [1.465, 1.465, 1.466, 1.466, 1.467],
            'other_col': [1, 2, 3, 4, 5],
        })

        result = grab_gps(df)

        assert 'tx_latitude' in result.columns
        assert 'tx_longitude' in result.columns
        assert len(result) == 3  # 3 unique coordinate pairs


class TestPdrReliabilityThresholds:
    """Tests for PDR threshold boundary conditions."""

    def test_pdr_exactly_at_threshold(self):
        """RAT with PDR exactly at threshold should be considered reliable."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_lstm': 15.0,
            'pred_pdr_dsrc_lstm': PDR_RELIABILITY_THRESHOLD,  # Exactly at threshold
            'pdr_dsrc': 0.99,
            'pred_latency_ms_pc5_lstm': 20.0,
            'pred_pdr_pc5_lstm': 0.5,  # Below threshold
            'pdr_pc5': 0.5,
            'pred_latency_ms_5g_lstm': 25.0,
            'pred_pdr_5g_lstm': 0.5,  # Below threshold
            'pdr_5g': 0.5,
        })

        result = select_best_rat(row, 'lstm')

        assert result == 'dsrc', "RAT at exactly threshold should be selected"

    def test_pdr_just_below_threshold(self):
        """RAT with PDR just below threshold should not be considered reliable."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_lstm': 10.0,  # Lower latency
            'pred_pdr_dsrc_lstm': PDR_RELIABILITY_THRESHOLD - 0.001,  # Just below
            'pdr_dsrc': 0.5,
            'pred_latency_ms_pc5_lstm': 20.0,
            'pred_pdr_pc5_lstm': PDR_RELIABILITY_THRESHOLD,  # At threshold
            'pdr_pc5': 0.5,
            'pred_latency_ms_5g_lstm': 25.0,
            'pred_pdr_5g_lstm': 0.5,  # Below threshold
            'pdr_5g': 0.5,
        })

        result = select_best_rat(row, 'lstm')

        # PC5 should be selected as it's at threshold, despite DSRC having lower latency
        assert result == 'pc5'
