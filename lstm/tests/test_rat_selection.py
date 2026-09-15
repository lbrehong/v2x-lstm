"""
Tests for selection/rat_selection.py - RAT selection algorithms and utility functions.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PDR_RELIABILITY_THRESHOLD, PDR_AVAILABILITY_THRESHOLD, LATENCY_TIE_MARGIN_MS, ALL_RATS


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

    def test_falls_back_to_5g_without_any_prediction(self):
        """No prediction for any RAT (e.g. no full measurement window): choose 5G."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_pc5_lstm': np.nan, 'pred_pdr_pc5_lstm': np.nan, 'pdr_pc5': 0.99,
            'pred_latency_ms_5g_lstm': np.nan, 'pred_pdr_5g_lstm': np.nan, 'pdr_5g': 0.99,
        })

        assert select_best_rat(row, 'lstm') == '5g'
        assert select_best_rat(row, 'lstm', ('pc5',)) == 'NaN'  # 5G not active: no fallback

    def test_no_fallback_when_disabled(self, monkeypatch):
        from selection.rat_selection import select_best_rat

        monkeypatch.setattr('selection.rat_selection.LSTM_FALLBACK_RAT', None)
        assert select_best_rat(pd.Series({'pdr_5g': 0.99, 'pdr_pc5': 0.99}), 'lstm') == 'NaN'

    def test_falls_back_when_no_rat_has_prediction_and_measurement(self):
        """PC5 predicted but never measured, 5G not predicted: the model cannot decide, so 5G."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_pc5_lstm': 12.0, 'pred_pdr_pc5_lstm': 0.999, 'pdr_pc5': np.nan,
            'pred_latency_ms_5g_lstm': np.nan, 'pred_pdr_5g_lstm': np.nan, 'pdr_5g': 0.99,
        })

        assert select_best_rat(row, 'lstm') == '5g'

    def test_falls_back_when_usable_rats_unavailable_and_5g_unpredicted(self):
        """PC5 predicted but unavailable (last PDR ~0), no 5G prediction: 5G by fallback, not NaN."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_pc5_lstm': 12.0, 'pred_pdr_pc5_lstm': 0.5, 'pdr_pc5': 0.01,
            'pred_latency_ms_5g_lstm': np.nan, 'pred_pdr_5g_lstm': np.nan, 'pdr_5g': 1.0,
        })

        assert select_best_rat(row, 'lstm') == '5g'
        assert select_best_rat(row, 'lstm', ('pc5',)) == 'NaN'

    def test_fallback_mask_matches_select_best_rat(self):
        from selection.rat_selection import lstm_fallback_mask, select_best_rat

        df = pd.DataFrame({
            'pred_latency_ms_5g_lstm': [np.nan, 30.0, np.nan, 30.0, np.nan],
            'pred_pdr_5g_lstm': [np.nan, 0.995, np.nan, 0.5, np.nan],
            'pdr_5g': [0.99, 0.99, 0.99, 0.02, 1.0],
            'pred_latency_ms_pc5_lstm': [np.nan, np.nan, 12.0, 12.0, 12.0],
            'pred_pdr_pc5_lstm': [np.nan, np.nan, 0.95, 0.5, 0.5],
            'pdr_pc5': [0.9, 0.9, np.nan, 0.03, 0.01],
        })

        mask = lstm_fallback_mask(df, 'lstm')
        decisions = df.apply(select_best_rat, args=('lstm',), axis=1)

        # Rows 0 and 2: nothing usable -> fallback. Row 3: 5G predicted but unavailable -> NaN.
        # Row 4: PC5 unavailable and no 5G prediction -> fallback
        assert mask.tolist() == [True, False, True, False, True]
        assert decisions.tolist() == ['5g', '5g', '5g', 'NaN', '5g']
        assert not lstm_fallback_mask(df, 'lstm', ('pc5',)).any()

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


    def test_rat_without_columns_is_ignored(self):
        """A RAT with no prediction/PDR columns (e.g. no DSRC data) is treated as unavailable."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_pc5_lstm': 10.0,
            'pred_pdr_pc5_lstm': 0.995,
            'pdr_pc5': 0.99,
            'pred_latency_ms_5g_lstm': 20.0,
            'pred_pdr_5g_lstm': 0.998,
            'pdr_5g': 0.99,
        })

        assert select_best_rat(row, 'lstm') == 'pc5'

    def test_default_rats_ignore_dsrc(self):
        """With the default RAT set, DSRC is never selected even when it is the best."""
        from selection.rat_selection import select_best_rat

        row = pd.Series({
            'pred_latency_ms_dsrc_lstm': 1.0,
            'pred_pdr_dsrc_lstm': 1.0,
            'pdr_dsrc': 1.0,
            'pred_latency_ms_pc5_lstm': 10.0,
            'pred_pdr_pc5_lstm': 0.995,
            'pdr_pc5': 0.99,
            'pred_latency_ms_5g_lstm': 20.0,
            'pred_pdr_5g_lstm': 0.998,
            'pdr_5g': 0.99,
        })

        assert select_best_rat(row, 'lstm') == 'pc5'
        assert select_best_rat(row, 'lstm', ALL_RATS) == 'dsrc'


class RecordingModel:
    """Stand-in model: records its input batch and predicts constant normalized values."""

    def __new__(cls, latency=0.5, pdr=0.97):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))  # predict_torch reads the device from parameters
                self.inputs = None

            def forward(self, x):
                self.inputs = x.numpy().copy()
                batch = x.shape[0]
                return torch.full((batch, 1), latency), torch.full((batch, 1), pdr)

        return _Model()


def _measurements(n=25, seed=0):
    """Super-merged style measurements for 5G and PC5, in time order."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'tx_latitude': 43.56 + np.arange(n) * 1e-4,
        'tx_longitude': 1.466 + np.arange(n) * 1e-4,
        'latency_ms_5g': rng.uniform(30, 200, n),
        'pdr_5g': rng.uniform(0.8, 1.0, n),
        'sinr': rng.uniform(5, 30, n),
        'rsrp': rng.uniform(-120, -80, n),
        'latency_ms_pc5': rng.uniform(5, 30, n),
        'pdr_pc5': rng.uniform(0.7, 1.0, n),
    })


class TestPredictionInputs:
    """Selection-stage model inputs must be built exactly like training inputs."""

    @pytest.mark.parametrize('rat', ['5g', 'pc5'])
    def test_windows_match_training_sequences(self, rat):
        from config import TIMESTEPS, TARGET_COLS, FEATURE_COLS
        from learning.data_preprocessing import preprocess_lstm_input
        from selection.rat_selection import build_prediction_windows

        df = _measurements()
        train_df = df.rename(columns={f'latency_ms_{rat}': 'latency_ms', f'pdr_{rat}': 'pdr'})
        X_train, _, _ = preprocess_lstm_input(train_df, new=True, rat=rat,
                                              target_cols=TARGET_COLS, seq_length=TIMESTEPS)

        windows, valid = build_prediction_windows(rat, df)

        assert not valid[:TIMESTEPS].any()
        assert valid[TIMESTEPS:].all()
        # Training sample j predicts row j + TIMESTEPS, so they must be identical
        np.testing.assert_allclose(windows, X_train)
        # Regression: measured features are real values, not zero padding
        assert windows.shape[2] == len(FEATURE_COLS[rat])
        assert (np.abs(windows[:, :, 2:]).sum(axis=(0, 1)) > 0).all()

    def test_row_is_never_part_of_its_own_window(self):
        from config import TIMESTEPS
        from selection.rat_selection import build_prediction_windows

        df = _measurements()
        changed = df.copy()
        changed.loc[20, 'latency_ms_5g'] = 999.0

        windows, valid = build_prediction_windows('5g', df)
        windows_changed, _ = build_prediction_windows('5g', changed)
        rows = np.flatnonzero(valid)

        row_20 = np.flatnonzero(rows == 20)[0]
        np.testing.assert_array_equal(windows[row_20], windows_changed[row_20])
        assert not np.array_equal(windows[row_20 + 1], windows_changed[row_20 + 1])
        assert rows[0] == TIMESTEPS

    def test_missing_measurements_make_rows_unavailable(self):
        from config import TIMESTEPS
        from selection.rat_selection import get_predictions

        df = _measurements(n=30)
        df.loc[15, ['latency_ms_pc5', 'pdr_pc5']] = np.nan
        model = RecordingModel()

        latency, pdr = get_predictions(model, 'pc5', df)

        unavailable = np.zeros(30, dtype=bool)
        unavailable[:TIMESTEPS] = True
        unavailable[16:16 + TIMESTEPS] = True  # windows that include row 15
        assert np.isnan(latency[unavailable]).all() and np.isnan(pdr[unavailable]).all()
        assert np.isfinite(latency[~unavailable]).all() and np.isfinite(pdr[~unavailable]).all()
        assert model.inputs.shape[0] == (~unavailable).sum()
        assert not np.isnan(model.inputs).any()

    def test_predictions_are_denormalized(self):
        from config import LATENCY_BOUNDS
        from selection.rat_selection import get_predictions

        latency, pdr = get_predictions(RecordingModel(latency=0.5, pdr=0.97), '5g', _measurements())

        lo, hi = LATENCY_BOUNDS['5g']
        valid = ~np.isnan(latency)
        np.testing.assert_allclose(latency[valid], lo + 0.5 * (hi - lo), rtol=1e-6)
        np.testing.assert_allclose(pdr[valid], 0.97, rtol=1e-6)

    def test_missing_feature_column_raises(self):
        from selection.rat_selection import get_predictions

        with pytest.raises(ValueError, match='sinr'):
            get_predictions(RecordingModel(), '5g', _measurements().drop(columns=['sinr']))


class TestLaggedMeasurements:
    """Selection-stage decisions for row i must only see measurements up to row i-1."""

    def test_measurements_shift_by_one_row(self):
        from selection.rat_selection import lagged_measurements

        df = _measurements(n=5)
        df['pred_pdr_5g_lstm'] = [0.9, 0.91, 0.92, 0.93, 0.94]
        lagged = lagged_measurements(df)

        for col in ('latency_ms_5g', 'pdr_5g', 'latency_ms_pc5', 'pdr_pc5'):
            assert np.isnan(lagged[col].iloc[0])
            np.testing.assert_array_equal(lagged[col].iloc[1:].to_numpy(), df[col].iloc[:-1].to_numpy())
        # Predictions and unrelated columns are untouched; input is not modified
        pd.testing.assert_series_equal(lagged['pred_pdr_5g_lstm'], df['pred_pdr_5g_lstm'])
        pd.testing.assert_series_equal(lagged['sinr'], df['sinr'])
        assert not np.isnan(df['pdr_5g'].iloc[0])

    def test_decision_for_a_row_ignores_its_own_measurements(self):
        from selection.rat_selection import lagged_measurements, opportunistic_best_rat

        df = pd.DataFrame({
            'latency_ms_pc5': [10.0, 10.0, 10.0],
            'pdr_pc5': [0.9, 0.9, 0.9],
            'latency_ms_5g': [20.0, 20.0, 20.0],
            'pdr_5g': [0.98, 0.98, 0.98],
        })
        changed = df.copy()
        changed.loc[2, 'pdr_pc5'] = 0.0  # PC5 fails at row 2 itself

        before = opportunistic_best_rat(lagged_measurements(df))['Best_RAT_opp'].tolist()
        after = opportunistic_best_rat(lagged_measurements(changed))['Best_RAT_opp'].tolist()

        assert before == after  # row 2's decision only saw row 1
        assert before[0] == 'NaN'  # nothing observed before the first row


class TestMergeCsvs:
    """Tests for merging matched per-RAT CSVs into super_merged.csv."""

    @staticmethod
    def _write_inputs(tmp_path, with_dsrc=False):
        coords = {'tx_latitude': [43.56, 43.57], 'tx_longitude': [1.466, 1.467]}
        pd.DataFrame({**coords, 'latency_ms_5g': [30.0, 31.0], 'pdr_5g': [1.0, 0.8]}).to_csv(
            tmp_path / 'super.csv', index=False)
        pd.DataFrame({**coords, 'latency_ms': [30.0, 31.0], 'sinr': [10.0, 11.0],
                      'rsrp': [-100.0, -101.0], 'pdr': [1.0, 0.8]}).to_csv(tmp_path / 'matched_5g.csv', index=False)
        pd.DataFrame({**coords, 'latency_ms': [20.0, 21.0], 'pdr': [0.9, 0.95]}).to_csv(
            tmp_path / 'matched_pc5.csv', index=False)
        if with_dsrc:
            pd.DataFrame({**coords, 'latency_ms': [5.0, 6.0], 'rsrp_1': [-80.0, -81.0],
                          'rsrp_2': [-82.0, -83.0], 'pdr': [0.99, 0.98]}).to_csv(
                tmp_path / 'matched_dsrc.csv', index=False)

    def test_default_rats_skip_dsrc(self, tmp_path):
        """By default, DSRC is not merged even if matched_dsrc.csv exists."""
        from selection.rat_selection import merge_csvs

        self._write_inputs(tmp_path, with_dsrc=True)
        df = merge_csvs(str(tmp_path))

        assert not any('dsrc' in col for col in df.columns)
        assert 'rsrp_1' not in df.columns
        assert df['latency_ms_pc5'].tolist() == [20.0, 21.0]

    def test_missing_matched_dsrc_gives_nan_columns(self, tmp_path):
        from selection.rat_selection import merge_csvs

        self._write_inputs(tmp_path)
        df = merge_csvs(str(tmp_path), rats=ALL_RATS)

        assert df['latency_ms_dsrc'].isna().all()
        assert df['pdr_dsrc'].isna().all()
        assert df['latency_ms_pc5'].tolist() == [20.0, 21.0]
        assert df['sinr'].tolist() == [10.0, 11.0]
        assert (tmp_path / 'super_merged.csv').exists()


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

    def test_opportunistic_default_rats_ignore_dsrc(self):
        """Default RAT set never picks DSRC, and works without DSRC columns."""
        from selection.rat_selection import opportunistic_best_rat

        df = pd.DataFrame({
            'latency_ms_dsrc': [1.0, 1.0],  # best latency, but DSRC is not selected
            'pdr_dsrc': [1.0, 1.0],
            'latency_ms_pc5': [10.0, 10.0],
            'pdr_pc5': [0.9, 0.9],
            'latency_ms_5g': [20.0, 20.0],
            'pdr_5g': [0.98, 0.98],
        })

        assert opportunistic_best_rat(df.copy())['Best_RAT_opp'].tolist() == ['pc5', 'pc5']
        assert opportunistic_best_rat(df.copy(), rats=ALL_RATS)['Best_RAT_opp'].tolist() == ['dsrc', 'dsrc']

        no_dsrc = df.drop(columns=['latency_ms_dsrc', 'pdr_dsrc'])
        assert opportunistic_best_rat(no_dsrc)['Best_RAT_opp'].tolist() == ['pc5', 'pc5']


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

        result = select_best_rat(row, 'lstm', ALL_RATS)

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
