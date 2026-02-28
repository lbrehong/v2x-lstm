"""
Tests for scripts/feedback_loop.py - Closed-loop feedback simulation.

Tests cover:
- _actual_latency: ground-truth latency extraction per RAT
- _actual_pdr: ground-truth PDR extraction per RAT
- _simulate_tx: transmission simulation correctness and determinism
- run_feedback_loop: full integration with mocked API/models/queue simulator
- _perturb_state: per-vehicle signal noise
- _simulate_tx_with_contention: TX simulation with RB-based contention
- PHY contention functions: utilization, contention PDR/latency
- run_multi_vehicle_loop: multi-vehicle integration with mocked components

Uses mocking for model-dependent functionality to ensure tests run
without requiring trained models or real data files.
"""
import os
import sys
import random

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_types import RATType, NetworkState, RATDecision, QueueContext, PacketSizeDecision
from config import TIMESTEPS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_state():
    """NetworkState with measurements for all three RATs."""
    return NetworkState(
        timestamp_ms=1700000000000,
        latitude=43.560,
        longitude=1.467,
        dsrc_latency_ms=12.5,
        dsrc_pdr=0.97,
        dsrc_rsrp_1=-85.0,
        dsrc_rsrp_2=-90.0,
        pc5_latency_ms=9.3,
        pc5_pdr=0.995,
        fiveg_latency_ms=18.0,
        fiveg_pdr=0.99,
        fiveg_sinr=25.0,
        fiveg_rsrp=-95.0,
    )


@pytest.fixture
def partial_state():
    """NetworkState with only PC5 measurements (others None)."""
    return NetworkState(
        timestamp_ms=1700000000000,
        latitude=43.560,
        longitude=1.467,
        pc5_latency_ms=10.0,
        pc5_pdr=0.98,
    )


@pytest.fixture
def super_merged_csv(tmp_path):
    """Create a small CSV file in super_merged format with 25 rows."""
    n_rows = 25
    rng = np.random.RandomState(42)

    data = {
        "tx_latitude": rng.uniform(43.558, 43.564, n_rows),
        "tx_longitude": rng.uniform(1.462, 1.472, n_rows),
        "latency_ms_5g": rng.uniform(10, 30, n_rows),
        "sinr": rng.uniform(-5, 40, n_rows),
        "rsrp": rng.uniform(-127, -67, n_rows),
        "pdr_5g": rng.uniform(0.9, 1.0, n_rows),
        "latency_ms_pc5": rng.uniform(5, 15, n_rows),
        "pdr_pc5": rng.uniform(0.9, 1.0, n_rows),
        "rsrp_1": rng.uniform(-150, -45, n_rows),
        "rsrp_2": rng.uniform(-150, -45, n_rows),
        "latency_ms_dsrc": rng.uniform(3, 12, n_rows),
        "pdr_dsrc": rng.uniform(0.9, 1.0, n_rows),
    }
    csv_path = tmp_path / "super_merged_test.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return str(csv_path)


# ---------------------------------------------------------------------------
# Tests for _actual_latency
# ---------------------------------------------------------------------------

class TestActualLatency:
    """Tests for _actual_latency helper."""

    def test_dsrc_latency(self, sample_state):
        """DSRC should return dsrc_latency_ms from the state."""
        from scripts.feedback_loop import _actual_latency
        result = _actual_latency(sample_state, RATType.DSRC)
        assert result == 12.5

    def test_pc5_latency(self, sample_state):
        """PC5 should return pc5_latency_ms from the state."""
        from scripts.feedback_loop import _actual_latency
        result = _actual_latency(sample_state, RATType.PC5)
        assert result == 9.3

    def test_fiveg_latency(self, sample_state):
        """FiveG should return fiveg_latency_ms from the state."""
        from scripts.feedback_loop import _actual_latency
        result = _actual_latency(sample_state, RATType.FiveG)
        assert result == 18.0

    def test_unavailable_returns_none(self, sample_state):
        """UNAVAILABLE RAT should return None."""
        from scripts.feedback_loop import _actual_latency
        result = _actual_latency(sample_state, RATType.UNAVAILABLE)
        assert result is None

    def test_missing_measurement_returns_none(self, partial_state):
        """RAT with no measurement in state should return None."""
        from scripts.feedback_loop import _actual_latency
        assert _actual_latency(partial_state, RATType.DSRC) is None
        assert _actual_latency(partial_state, RATType.FiveG) is None

    def test_present_measurement_in_partial_state(self, partial_state):
        """RAT that does have a measurement should still return it."""
        from scripts.feedback_loop import _actual_latency
        assert _actual_latency(partial_state, RATType.PC5) == 10.0


# ---------------------------------------------------------------------------
# Tests for _actual_pdr
# ---------------------------------------------------------------------------

class TestActualPdr:
    """Tests for _actual_pdr helper."""

    def test_dsrc_pdr(self, sample_state):
        """DSRC should return dsrc_pdr from the state."""
        from scripts.feedback_loop import _actual_pdr
        result = _actual_pdr(sample_state, RATType.DSRC)
        assert result == 0.97

    def test_pc5_pdr(self, sample_state):
        """PC5 should return pc5_pdr from the state."""
        from scripts.feedback_loop import _actual_pdr
        result = _actual_pdr(sample_state, RATType.PC5)
        assert result == 0.995

    def test_fiveg_pdr(self, sample_state):
        """FiveG should return fiveg_pdr from the state."""
        from scripts.feedback_loop import _actual_pdr
        result = _actual_pdr(sample_state, RATType.FiveG)
        assert result == 0.99

    def test_unavailable_returns_none(self, sample_state):
        """UNAVAILABLE RAT should return None."""
        from scripts.feedback_loop import _actual_pdr
        result = _actual_pdr(sample_state, RATType.UNAVAILABLE)
        assert result is None

    def test_missing_measurement_returns_none(self, partial_state):
        """RAT with no PDR in state should return None."""
        from scripts.feedback_loop import _actual_pdr
        assert _actual_pdr(partial_state, RATType.DSRC) is None
        assert _actual_pdr(partial_state, RATType.FiveG) is None

    def test_present_measurement_in_partial_state(self, partial_state):
        """RAT that does have a PDR should still return it."""
        from scripts.feedback_loop import _actual_pdr
        assert _actual_pdr(partial_state, RATType.PC5) == 0.98


# ---------------------------------------------------------------------------
# Tests for _simulate_tx
# ---------------------------------------------------------------------------

class TestSimulateTx:
    """Tests for _simulate_tx transmission simulation."""

    def test_returns_three_values(self):
        """_simulate_tx should return a (bool, float, float) tuple."""
        from scripts.feedback_loop import _simulate_tx

        random.seed(42)
        delivered, latency, corrected_pdr = _simulate_tx(
            rat=RATType.PC5,
            packet_size=1000,
            predicted_pdr=0.95,
            base_packet_size=1000,
            correction_exponent=1.0,
        )
        assert isinstance(delivered, bool)
        assert isinstance(latency, float)
        assert isinstance(corrected_pdr, float)

    def test_delivered_is_bool(self):
        """delivered field should always be a boolean."""
        from scripts.feedback_loop import _simulate_tx

        random.seed(99)
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            delivered, _, _ = _simulate_tx(rat, 1024, 0.95, 1000, 0.8)
            assert isinstance(delivered, bool)

    def test_latency_is_positive(self):
        """Simulated latency should always be positive."""
        from scripts.feedback_loop import _simulate_tx

        random.seed(0)
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            for size in [512, 1024, 2048, 4096]:
                _, latency, _ = _simulate_tx(rat, size, 0.99, 1000, 0.8)
                assert latency > 0, f"Latency should be positive for {rat.value}, size={size}"

    def test_corrected_pdr_bounded_zero_one(self):
        """Corrected PDR should be in [0, 1]."""
        from scripts.feedback_loop import _simulate_tx

        random.seed(7)
        for pdr in [0.0, 0.5, 0.95, 0.99, 1.0]:
            for size in [500, 1000, 2000, 4096]:
                _, _, corrected_pdr = _simulate_tx(
                    RATType.FiveG, size, pdr, 1000, 0.8,
                )
                assert 0.0 <= corrected_pdr <= 1.0, (
                    f"Corrected PDR out of range for pdr={pdr}, size={size}"
                )

    def test_deterministic_with_seed(self):
        """Same seed should produce the same results."""
        from scripts.feedback_loop import _simulate_tx

        results = []
        for _ in range(2):
            random.seed(123)
            result = _simulate_tx(RATType.PC5, 1024, 0.95, 1000, 0.8)
            results.append(result)

        assert results[0] == results[1], "Results should be identical with same seed"

    def test_different_seeds_differ(self):
        """Different seeds should (very likely) produce different results."""
        from scripts.feedback_loop import _simulate_tx

        random.seed(1)
        r1 = _simulate_tx(RATType.PC5, 1024, 0.95, 1000, 0.8)
        random.seed(999)
        r2 = _simulate_tx(RATType.PC5, 1024, 0.95, 1000, 0.8)

        # At minimum the latency should differ since it includes random jitter
        assert r1[1] != r2[1], "Different seeds should produce different latencies"

    def test_dsrc_vs_non_dsrc_jitter(self):
        """DSRC and non-DSRC should use different jitter ranges."""
        from scripts.feedback_loop import _simulate_tx

        # Run many samples with the same seed and collect latencies
        dsrc_latencies = []
        fiveg_latencies = []
        for i in range(50):
            random.seed(i)
            _, lat_dsrc, _ = _simulate_tx(RATType.DSRC, 1024, 0.99, 1000, 0.8)
            random.seed(i)
            _, lat_5g, _ = _simulate_tx(RATType.FiveG, 1024, 0.99, 1000, 0.8)
            dsrc_latencies.append(lat_dsrc)
            fiveg_latencies.append(lat_5g)

        # They should differ because base latencies and jitter ranges differ
        assert np.mean(dsrc_latencies) != pytest.approx(np.mean(fiveg_latencies), abs=0.01)

    def test_larger_packet_increases_latency(self):
        """Larger packets should generally produce higher latency due to size_factor."""
        from scripts.feedback_loop import _simulate_tx

        random.seed(42)
        _, lat_small, _ = _simulate_tx(RATType.PC5, 500, 0.99, 1000, 0.8)
        random.seed(42)
        _, lat_large, _ = _simulate_tx(RATType.PC5, 4000, 0.99, 1000, 0.8)

        # lat_large should be higher because of both tx_time and size_factor
        assert lat_large > lat_small

    def test_zero_pdr_never_delivers(self):
        """A predicted PDR of 0 should result in corrected_pdr of 0 and never deliver."""
        from scripts.feedback_loop import _simulate_tx

        for i in range(20):
            random.seed(i)
            delivered, _, corrected_pdr = _simulate_tx(
                RATType.FiveG, 1000, 0.0, 1000, 0.8,
            )
            assert corrected_pdr == 0.0
            assert delivered is False


# ---------------------------------------------------------------------------
# Tests for _retrain_model
# ---------------------------------------------------------------------------

class TestRetrainModel:
    """Tests for _retrain_model helper."""

    def _make_retrain_mock(self, rat, n_samples, n_features, model_type="lstm",
                           eval_return=None):
        """Build mock model + api with all fields needed by _retrain_model."""
        from config import create_latency_scaler

        eval_return = eval_return or [0.05, 0.02, 0.03, 0.01, 0.01]
        mock_model = Mock()
        mock_model.evaluate.return_value = eval_return
        mock_history = Mock()
        mock_history.history = {
            "loss": [0.04],
            "latency_ms_loss": [0.015],
            "pdr_loss": [0.025],
            "latency_ms_rmse": [0.008],
            "pdr_rmse": [0.009],
            "val_loss": [0.06],
            "val_latency_ms_loss": [0.025],
            "val_pdr_loss": [0.035],
            "val_latency_ms_rmse": [0.012],
            "val_pdr_rmse": [0.015],
        }
        mock_model.fit.return_value = mock_history
        mock_model.save = Mock()
        mock_model.optimizer = Mock()
        mock_model.optimizer.learning_rate = 0.001
        mock_model.predict.return_value = [
            np.full((n_samples, 1), 0.1, dtype=np.float32),
            np.full((n_samples, 1), 0.8, dtype=np.float32),
        ]

        mock_api = Mock()
        mock_api.models = {rat: mock_model}
        mock_api.model_type = model_type
        mock_api.latency_scaler = create_latency_scaler()

        x = np.random.rand(n_samples, TIMESTEPS, n_features).astype(np.float32)
        y_lat = np.random.rand(n_samples).astype(np.float32)
        y_pdr = np.random.rand(n_samples).astype(np.float32)
        return mock_api, mock_model, x, y_lat, y_pdr

    def test_retrain_appends_to_log(self):
        """_retrain_model should append one entry to retrain_log with all metrics."""
        from scripts.feedback_loop import _retrain_model

        mock_api, _, x, y_lat, y_pdr = self._make_retrain_mock("pc5", 10, 4)
        retrain_log = []

        with patch("scripts.feedback_loop.MODEL_DIR", "/tmp/test_models"):
            _retrain_model(mock_api, "pc5", x, y_lat, y_pdr, cycle=1, retrain_log=retrain_log)

        assert len(retrain_log) == 1
        entry = retrain_log[0]
        assert entry["cycle"] == 1
        assert entry["rat"] == "pc5"
        assert entry["n_samples"] == 10
        assert entry["loss_before"] == pytest.approx(0.05)
        assert entry["loss_after"] == pytest.approx(0.04)
        assert "pdr_mae" in entry
        # Timing and learning rate
        assert entry["train_time_s"] >= 0
        assert entry["learning_rate"] == pytest.approx(0.001)
        # Per-head before metrics
        assert entry["latency_loss_before"] == pytest.approx(0.02)
        assert entry["pdr_loss_before"] == pytest.approx(0.03)
        assert entry["latency_rmse_before"] == pytest.approx(0.01)
        assert entry["pdr_rmse_before"] == pytest.approx(0.01)
        # Per-head after metrics
        assert entry["latency_loss_after"] == pytest.approx(0.015)
        assert entry["pdr_loss_after"] == pytest.approx(0.025)
        # Validation metrics
        assert entry["val_loss"] == pytest.approx(0.06)
        assert entry["val_latency_loss"] == pytest.approx(0.025)
        assert entry["val_pdr_loss"] == pytest.approx(0.035)
        # Prediction accuracy snapshot
        assert entry["latency_mae_ms"] >= 0
        assert entry["pdr_mae"] >= 0

    def test_retrain_calls_fit_not_save(self):
        """_retrain_model should call model.fit but not save (save happens at end of loop)."""
        from scripts.feedback_loop import _retrain_model

        mock_api, mock_model, x, y_lat, y_pdr = self._make_retrain_mock(
            "dsrc", 5, 6, model_type="gru")

        _retrain_model(mock_api, "dsrc", x, y_lat, y_pdr, cycle=2, retrain_log=[])

        mock_model.fit.assert_called_once()
        mock_model.save.assert_not_called()

    def test_retrain_evaluate_scalar(self):
        """_retrain_model should handle evaluate returning a scalar."""
        from scripts.feedback_loop import _retrain_model

        mock_api, _, x, y_lat, y_pdr = self._make_retrain_mock(
            "5g", 3, 6, eval_return=0.07)  # scalar, not list
        retrain_log = []

        with patch("scripts.feedback_loop.MODEL_DIR", "/tmp/test_models"):
            _retrain_model(mock_api, "5g", x, y_lat, y_pdr, cycle=1, retrain_log=retrain_log)

        assert retrain_log[0]["loss_before"] == pytest.approx(0.07)
        # Per-head before metrics should be None for scalar evaluate
        assert retrain_log[0]["latency_loss_before"] is None


# ---------------------------------------------------------------------------
# Tests for run_feedback_loop (integration, heavily mocked)
# ---------------------------------------------------------------------------

class TestRunFeedbackLoop:
    """Integration tests for run_feedback_loop with mocked components."""

    def _make_mock_model(self, n_features):
        """Create a mock Keras model for a given feature count."""
        mock_model = Mock()
        # predict returns [latency_array, pdr_array]
        mock_model.predict.return_value = [
            np.array([[0.3]]),   # normalized latency
            np.array([[0.96]]),  # PDR
        ]
        mock_history = Mock()
        mock_history.history = {"loss": [0.05]}
        mock_model.fit.return_value = mock_history
        mock_model.evaluate.return_value = [0.06, 0.03, 0.03, 0.01, 0.01]
        mock_model.save = Mock()
        return mock_model

    def _make_mock_api(self):
        """Create a fully mocked RATSelectionAPI."""
        mock_api = Mock()
        mock_api.model_type = "lstm"

        # select_rat returns a RATDecision
        mock_api.select_rat.return_value = RATDecision(
            selected_rat=RATType.PC5,
            confidence=0.85,
            predicted_latency_ms=10.0,
            predicted_pdr=0.96,
            all_predictions={RATType.PC5: (10.0, 0.96)},
            model_type="lstm",
        )

        # _history has enough entries for sequence building
        mock_api._history = {
            "dsrc": [np.zeros(6).tolist() for _ in range(TIMESTEPS)],
            "pc5": [np.zeros(4).tolist() for _ in range(TIMESTEPS)],
            "5g": [np.zeros(6).tolist() for _ in range(TIMESTEPS)],
        }

        # models dict
        mock_api.models = {
            "dsrc": self._make_mock_model(6),
            "pc5": self._make_mock_model(4),
            "5g": self._make_mock_model(6),
        }

        return mock_api

    def _make_mock_qsim(self):
        """Create a mock QueueSimulator."""
        mock_qsim = Mock()

        mock_qsim.get_queue_context.return_value = QueueContext(
            queue_depth=0,
            avg_packet_size_bytes=1024,
            urgency_level=0.0,
            recent_pdr_trend=0.0,
            target_latency_ms=20.0,
            target_pdr=0.99,
        )

        mock_qsim.decide_packet_size.return_value = PacketSizeDecision(
            packet_size_bytes=1024,
            fragment_count=1,
            priority_level=4,
            send_rate_hz=50.0,
        )

        # DTMC sizers per RAT with mock get_window_pdr and get_statistics
        mock_dtmc = Mock()
        mock_dtmc.current_state = 0
        mock_dtmc.current_size = 1024
        mock_dtmc.get_window_pdr.return_value = 0.97
        mock_dtmc.get_statistics.return_value = {
            "mean_state": 0.5, "stay_count": 10,
        }
        mock_dtmc.history = [(0, 0, 0.97, "stay"), (1, 1, 0.99, "increase")]
        mock_dtmc.size_levels = np.array([512, 1024, 2048, 4096])

        mock_qsim.dtmc_sizers = {
            RATType.DSRC: mock_dtmc,
            RATType.PC5: mock_dtmc,
            RATType.FiveG: mock_dtmc,
        }

        mock_qsim.queue_depth = 0
        mock_qsim.metrics = Mock()
        mock_qsim.metrics.total_packets = 0
        mock_qsim.metrics.successful_packets = 0

        mock_qsim.update_pdr_estimate = Mock()

        return mock_qsim

    def test_returns_summary_dict(self, super_merged_csv, tmp_path):
        """run_feedback_loop should return a summary dict with expected keys."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,  # large so no retrain in 25 rows
            )

        assert isinstance(summary, dict)
        expected_keys = [
            "input_file", "model_type", "seed", "total_rows",
            "processed", "successful", "achieved_pdr",
            "mean_sim_latency", "mean_pred_latency",
            "successful_bytes", "throughput_bytes_per_s", "mean_packet_size",
            "rat_switches", "retrain_cycles_total",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_creates_output_files(self, super_merged_csv, tmp_path):
        """run_feedback_loop should create feedback_loop_log.csv and summary CSV."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
            )

        log_path = os.path.join(output_dir, "feedback_loop_log.csv")
        summary_path = os.path.join(output_dir, "feedback_loop_summary.csv")

        assert os.path.exists(log_path), "feedback_loop_log.csv should be created"
        assert os.path.exists(summary_path), "feedback_loop_summary.csv should be created"

        # Verify log has content
        log_df = pd.read_csv(log_path)
        assert len(log_df) == 25, "Log should have one row per input row"

    def test_creates_dtmc_transitions_csv(self, super_merged_csv, tmp_path):
        """run_feedback_loop should create feedback_dtmc_transitions.csv."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
            )

        dtmc_path = os.path.join(output_dir, "feedback_dtmc_transitions.csv")
        assert os.path.exists(dtmc_path), "feedback_dtmc_transitions.csv should be created"
        dtmc_df = pd.read_csv(dtmc_path)
        expected_cols = ["rat", "step", "state", "packet_size", "pdr", "action"]
        for col in expected_cols:
            assert col in dtmc_df.columns, f"Missing column: {col}"
        # Mock has 2 transitions per RAT x 3 RATs = 6 rows
        assert len(dtmc_df) == 6

    def test_throughput_fields_in_summary(self, super_merged_csv, tmp_path):
        """Summary should contain throughput fields with correct types."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
            )

        assert summary["successful_bytes"] >= 0
        assert summary["throughput_bytes_per_s"] >= 0
        assert summary["mean_packet_size"] > 0

    def test_total_rows_matches_input(self, super_merged_csv, tmp_path):
        """Summary total_rows should match the number of CSV rows."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
            )

        assert summary["total_rows"] == 25

    def test_seed_produces_deterministic_results(self, super_merged_csv, tmp_path):
        """Running with the same seed should produce the same summary."""
        from scripts.feedback_loop import run_feedback_loop

        summaries = []
        for run_idx in range(2):
            mock_api = self._make_mock_api()
            mock_qsim = self._make_mock_qsim()

            output_dir = str(tmp_path / f"output_{run_idx}")
            model_dir = str(tmp_path / f"models_{run_idx}")

            with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
                 patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
                 patch("config.OUTPUT_DIR", output_dir), \
                 patch("scripts.feedback_loop.MODEL_DIR", model_dir):

                s = run_feedback_loop(
                    input_csv=super_merged_csv,
                    model_type="lstm",
                    seed=42,
                    retrain_interval=9999,
                )
                summaries.append(s)

        assert summaries[0]["achieved_pdr"] == summaries[1]["achieved_pdr"]
        assert summaries[0]["mean_sim_latency"] == summaries[1]["mean_sim_latency"]

    def test_unavailable_rat_skipped(self, super_merged_csv, tmp_path):
        """When API returns UNAVAILABLE, that row should be skipped gracefully."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()
        # Make the first 5 calls return UNAVAILABLE, rest return PC5
        call_count = {"n": 0}
        def side_effect_select_rat(state, queue_ctx=None):
            call_count["n"] += 1
            if call_count["n"] <= 5:
                return RATDecision(
                    selected_rat=RATType.UNAVAILABLE,
                    confidence=0.0,
                    predicted_latency_ms=0.0,
                    predicted_pdr=0.0,
                    all_predictions={},
                    model_type="lstm",
                )
            return RATDecision(
                selected_rat=RATType.PC5,
                confidence=0.85,
                predicted_latency_ms=10.0,
                predicted_pdr=0.96,
                all_predictions={RATType.PC5: (10.0, 0.96)},
                model_type="lstm",
            )

        mock_api.select_rat.side_effect = side_effect_select_rat
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
            )

        # 5 rows were UNAVAILABLE, so processed count should be 20
        assert summary["processed"] == 20

    def test_no_rat_switches_when_same_rat(self, super_merged_csv, tmp_path):
        """If API always selects the same RAT, rat_switches should be zero."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()
        # Always returns PC5 (default fixture behavior)
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
            )

        assert summary["rat_switches"] == 0

    def test_rat_switches_counted(self, super_merged_csv, tmp_path):
        """RAT switches should be counted when API alternates between RATs."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()

        call_idx = {"n": 0}
        def alternating_select_rat(state, queue_ctx=None):
            call_idx["n"] += 1
            rat = RATType.PC5 if call_idx["n"] % 2 == 0 else RATType.DSRC
            return RATDecision(
                selected_rat=rat,
                confidence=0.85,
                predicted_latency_ms=10.0,
                predicted_pdr=0.96,
                all_predictions={rat: (10.0, 0.96)},
                model_type="lstm",
            )

        mock_api.select_rat.side_effect = alternating_select_rat
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
            )

        # 25 rows, alternating DSRC/PC5 -> 24 switches
        assert summary["rat_switches"] == 24

    def test_model_type_passed_through(self, super_merged_csv, tmp_path):
        """Summary model_type should match the argument passed in."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="gru",
                seed=0,
                retrain_interval=9999,
            )

        assert summary["model_type"] == "gru"

    def test_log_csv_columns(self, super_merged_csv, tmp_path):
        """The row-level log CSV should contain all expected columns."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
            )

        log_df = pd.read_csv(os.path.join(output_dir, "feedback_loop_log.csv"))
        expected_cols = [
            "idx", "sim_time", "latitude", "longitude",
            "selected_rat", "pred_latency", "pred_pdr",
            "actual_latency", "actual_pdr",
            "sim_latency", "delivered", "packet_size",
            "corrected_pdr", "dtmc_state", "dtmc_size",
            "confidence", "queue_depth",
        ]
        for col in expected_cols:
            assert col in log_df.columns, f"Missing column: {col}"

    def test_sim_time_increments(self, super_merged_csv, tmp_path):
        """Simulated time should increment by 0.1s per row."""
        from scripts.feedback_loop import run_feedback_loop

        mock_api = self._make_mock_api()
        mock_qsim = self._make_mock_qsim()

        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", return_value=mock_qsim), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            run_feedback_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
            )

        log_df = pd.read_csv(os.path.join(output_dir, "feedback_loop_log.csv"))
        sim_times = log_df["sim_time"].values
        # Each step increments by 0.1; first row starts at 0.0
        expected_first = 0.0
        expected_last = 0.1 * 24  # 25 rows, 0-indexed: last sim_time = 2.4
        assert sim_times[0] == pytest.approx(expected_first, abs=1e-3)
        assert sim_times[-1] == pytest.approx(expected_last, abs=1e-3)


# ---------------------------------------------------------------------------
# Tests for RAT_STR mapping
# ---------------------------------------------------------------------------

class TestRATStrMapping:
    """Tests for the RAT_STR constant."""

    def test_all_active_rats_present(self):
        """RAT_STR should map DSRC, PC5, and FiveG."""
        from scripts.feedback_loop import RAT_STR

        assert RATType.DSRC in RAT_STR
        assert RATType.PC5 in RAT_STR
        assert RATType.FiveG in RAT_STR

    def test_values_are_lowercase_strings(self):
        """RAT_STR values should be the lowercase config keys."""
        from scripts.feedback_loop import RAT_STR

        assert RAT_STR[RATType.DSRC] == "dsrc"
        assert RAT_STR[RATType.PC5] == "pc5"
        assert RAT_STR[RATType.FiveG] == "5g"

    def test_unavailable_not_in_mapping(self):
        """UNAVAILABLE should not appear in RAT_STR."""
        from scripts.feedback_loop import RAT_STR

        assert RATType.UNAVAILABLE not in RAT_STR


# ---------------------------------------------------------------------------
# Tests for _perturb_state
# ---------------------------------------------------------------------------

class TestPerturbState:
    """Tests for _perturb_state helper."""

    def test_returns_network_state(self, sample_state):
        """Should return a NetworkState object."""
        from scripts.feedback_loop import _perturb_state
        rng = np.random.RandomState(42)
        result = _perturb_state(sample_state, rng)
        assert isinstance(result, NetworkState)

    def test_perturbed_differs_from_original(self, sample_state):
        """Perturbed state should differ from original."""
        from scripts.feedback_loop import _perturb_state
        rng = np.random.RandomState(42)
        result = _perturb_state(sample_state, rng)
        # At least one field should differ
        assert (
            result.dsrc_latency_ms != sample_state.dsrc_latency_ms
            or result.pc5_latency_ms != sample_state.pc5_latency_ms
            or result.fiveg_pdr != sample_state.fiveg_pdr
        )

    def test_deterministic_with_seed(self, sample_state):
        """Same RNG seed should produce same perturbation."""
        from scripts.feedback_loop import _perturb_state
        r1 = _perturb_state(sample_state, np.random.RandomState(42))
        r2 = _perturb_state(sample_state, np.random.RandomState(42))
        assert r1.dsrc_latency_ms == r2.dsrc_latency_ms
        assert r1.fiveg_pdr == r2.fiveg_pdr

    def test_pdr_clamped_to_zero_one(self, sample_state):
        """Perturbed PDR should always be in [0, 1]."""
        from scripts.feedback_loop import _perturb_state
        for seed in range(50):
            result = _perturb_state(sample_state, np.random.RandomState(seed))
            for pdr in [result.dsrc_pdr, result.pc5_pdr, result.fiveg_pdr]:
                if pdr is not None:
                    assert 0.0 <= pdr <= 1.0, f"PDR out of bounds: {pdr}"

    def test_latency_stays_positive(self, sample_state):
        """Perturbed latency should always be positive."""
        from scripts.feedback_loop import _perturb_state
        for seed in range(50):
            result = _perturb_state(sample_state, np.random.RandomState(seed))
            for lat in [result.dsrc_latency_ms, result.pc5_latency_ms, result.fiveg_latency_ms]:
                if lat is not None:
                    assert lat > 0, f"Latency should be positive, got {lat}"

    def test_none_fields_stay_none(self, partial_state):
        """Fields that are None in original should stay None."""
        from scripts.feedback_loop import _perturb_state
        result = _perturb_state(partial_state, np.random.RandomState(42))
        assert result.dsrc_latency_ms is None
        assert result.fiveg_latency_ms is None
        assert result.dsrc_pdr is None

    def test_coordinates_unchanged(self, sample_state):
        """GPS coordinates should not be perturbed."""
        from scripts.feedback_loop import _perturb_state
        result = _perturb_state(sample_state, np.random.RandomState(42))
        assert result.latitude == sample_state.latitude
        assert result.longitude == sample_state.longitude


# ---------------------------------------------------------------------------
# Tests for _simulate_tx_with_contention
# ---------------------------------------------------------------------------

class TestSimulateTxWithContention:
    """Tests for _simulate_tx_with_contention (RB-based)."""

    def test_returns_three_values(self):
        """Should return (bool, float, float)."""
        from scripts.feedback_loop import _simulate_tx_with_contention
        random.seed(42)
        delivered, latency, pdr = _simulate_tx_with_contention(
            RATType.PC5, 1000, 0.95, 1000, 0.8, 5, 0.5,
        )
        assert isinstance(delivered, bool)
        assert isinstance(latency, float)
        assert isinstance(pdr, float)

    def test_single_vehicle_matches_base(self):
        """With 1 vehicle, should behave like _simulate_tx (no contention effect)."""
        from scripts.feedback_loop import _simulate_tx, _simulate_tx_with_contention
        random.seed(42)
        r1 = _simulate_tx(RATType.PC5, 1000, 0.95, 1000, 0.8)
        random.seed(42)
        r2 = _simulate_tx_with_contention(RATType.PC5, 1000, 0.95, 1000, 0.8, 1, 0.0)
        assert r1[1] == pytest.approx(r2[1], abs=1e-6)
        assert r1[2] == pytest.approx(r2[2], abs=1e-6)

    def test_more_vehicles_higher_latency(self):
        """More vehicles should increase latency."""
        from scripts.feedback_loop import _simulate_tx_with_contention
        random.seed(42)
        _, lat_2, _ = _simulate_tx_with_contention(RATType.DSRC, 1000, 0.99, 1000, 0.8, 2, 0.1)
        random.seed(42)
        _, lat_10, _ = _simulate_tx_with_contention(RATType.DSRC, 1000, 0.99, 1000, 0.8, 10, 0.5)
        assert lat_10 > lat_2

    def test_more_vehicles_lower_pdr(self):
        """More vehicles should reduce corrected PDR on DSRC."""
        from scripts.feedback_loop import _simulate_tx_with_contention
        random.seed(42)
        _, _, pdr_2 = _simulate_tx_with_contention(RATType.DSRC, 1000, 0.99, 1000, 0.8, 2, 0.1)
        random.seed(42)
        _, _, pdr_10 = _simulate_tx_with_contention(RATType.DSRC, 1000, 0.99, 1000, 0.8, 10, 0.5)
        assert pdr_10 < pdr_2

    def test_5g_pdr_unchanged_with_contention(self):
        """5G PDR should not change with more vehicles (scheduled access)."""
        from scripts.feedback_loop import _simulate_tx_with_contention
        random.seed(42)
        _, _, pdr_1 = _simulate_tx_with_contention(RATType.FiveG, 1000, 0.99, 1000, 0.8, 1, 0.0)
        random.seed(42)
        _, _, pdr_20 = _simulate_tx_with_contention(RATType.FiveG, 1000, 0.99, 1000, 0.8, 20, 0.001)
        assert pdr_1 == pytest.approx(pdr_20, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests for run_multi_vehicle_loop (integration, heavily mocked)
# ---------------------------------------------------------------------------

class TestRunMultiVehicleLoop:
    """Integration tests for run_multi_vehicle_loop."""

    def _make_mock_model(self, n_features):
        mock_model = Mock()
        mock_model.predict.return_value = [
            np.array([[0.3]]),
            np.array([[0.96]]),
        ]
        mock_history = Mock()
        mock_history.history = {"loss": [0.05]}
        mock_model.fit.return_value = mock_history
        mock_model.evaluate.return_value = [0.06, 0.03, 0.03, 0.01, 0.01]
        mock_model.save = Mock()
        return mock_model

    def _make_mock_api(self):
        mock_api = Mock()
        mock_api.model_type = "lstm"

        mock_api.select_rat.return_value = RATDecision(
            selected_rat=RATType.PC5,
            confidence=0.85,
            predicted_latency_ms=10.0,
            predicted_pdr=0.96,
            all_predictions={RATType.PC5: (10.0, 0.96)},
            model_type="lstm",
        )

        mock_api._history = {
            "dsrc": [np.zeros(6).tolist() for _ in range(TIMESTEPS)],
            "pc5": [np.zeros(4).tolist() for _ in range(TIMESTEPS)],
            "5g": [np.zeros(6).tolist() for _ in range(TIMESTEPS)],
        }

        mock_api.models = {
            "dsrc": self._make_mock_model(6),
            "pc5": self._make_mock_model(4),
            "5g": self._make_mock_model(6),
        }

        return mock_api

    def _make_mock_qsim(self):
        mock_qsim = Mock()
        mock_qsim.get_queue_context.return_value = QueueContext(
            queue_depth=0,
            avg_packet_size_bytes=1024,
            urgency_level=0.0,
            recent_pdr_trend=0.0,
            target_latency_ms=20.0,
            target_pdr=0.99,
        )
        mock_qsim.decide_packet_size.return_value = PacketSizeDecision(
            packet_size_bytes=1024,
            fragment_count=1,
            priority_level=4,
            send_rate_hz=50.0,
        )
        mock_dtmc = Mock()
        mock_dtmc.current_state = 0
        mock_dtmc.current_size = 1024
        mock_dtmc.get_window_pdr.return_value = 0.97
        mock_dtmc.get_statistics.return_value = {
            "mean_state": 0.5, "stay_count": 10,
        }
        mock_dtmc.history = [(0, 0, 0.97, "stay"), (1, 1, 0.99, "increase")]
        mock_dtmc.size_levels = np.array([512, 1024, 2048, 4096])
        mock_qsim.dtmc_sizers = {
            RATType.DSRC: mock_dtmc,
            RATType.PC5: mock_dtmc,
            RATType.FiveG: mock_dtmc,
        }
        mock_qsim.queue_depth = 0
        mock_qsim.metrics = Mock()
        mock_qsim.metrics.total_packets = 0
        mock_qsim.metrics.successful_packets = 0
        mock_qsim.update_pdr_estimate = Mock()
        return mock_qsim

    def test_returns_summary_dict(self, super_merged_csv, tmp_path):
        """run_multi_vehicle_loop should return a summary dict."""
        from scripts.feedback_loop import run_multi_vehicle_loop

        mock_api = self._make_mock_api()
        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", side_effect=lambda **kw: self._make_mock_qsim()), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_multi_vehicle_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
                num_vehicles=3,
            )

        assert isinstance(summary, dict)
        assert summary["num_vehicles"] == 3
        assert "total_packets" in summary
        assert "achieved_pdr" in summary

    def test_creates_output_files(self, super_merged_csv, tmp_path):
        """Should create log, contention, and summary CSVs."""
        from scripts.feedback_loop import run_multi_vehicle_loop

        mock_api = self._make_mock_api()
        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", side_effect=lambda **kw: self._make_mock_qsim()), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            run_multi_vehicle_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
                num_vehicles=3,
            )

        assert os.path.exists(os.path.join(output_dir, "feedback_multi_log.csv"))
        assert os.path.exists(os.path.join(output_dir, "feedback_multi_summary.csv"))
        assert os.path.exists(os.path.join(output_dir, "feedback_multi_contention.csv"))
        assert os.path.exists(os.path.join(output_dir, "feedback_multi_per_vehicle.csv"))

    def test_creates_dtmc_transitions_csv(self, super_merged_csv, tmp_path):
        """Should create feedback_multi_dtmc_transitions.csv with vehicle_id column."""
        from scripts.feedback_loop import run_multi_vehicle_loop

        mock_api = self._make_mock_api()
        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", side_effect=lambda **kw: self._make_mock_qsim()), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            run_multi_vehicle_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
                num_vehicles=3,
            )

        dtmc_path = os.path.join(output_dir, "feedback_multi_dtmc_transitions.csv")
        assert os.path.exists(dtmc_path), "feedback_multi_dtmc_transitions.csv should be created"
        dtmc_df = pd.read_csv(dtmc_path)
        expected_cols = ["vehicle_id", "rat", "step", "state", "packet_size", "pdr", "action"]
        for col in expected_cols:
            assert col in dtmc_df.columns, f"Missing column: {col}"
        # 3 vehicles x 3 RATs x 2 transitions = 18 rows
        assert len(dtmc_df) == 18

    def test_throughput_in_summary_and_per_vehicle(self, super_merged_csv, tmp_path):
        """Summary and per-vehicle CSVs should contain throughput fields."""
        from scripts.feedback_loop import run_multi_vehicle_loop

        mock_api = self._make_mock_api()
        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", side_effect=lambda **kw: self._make_mock_qsim()), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_multi_vehicle_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
                num_vehicles=3,
            )

        assert "successful_bytes" in summary
        assert "throughput_bytes_per_s" in summary
        assert summary["successful_bytes"] >= 0
        assert summary["throughput_bytes_per_s"] >= 0

        # Per-vehicle CSV should have throughput columns
        pv_df = pd.read_csv(os.path.join(output_dir, "feedback_multi_per_vehicle.csv"))
        assert "successful_bytes" in pv_df.columns
        assert "throughput_bytes_per_s" in pv_df.columns

    def test_log_has_vehicle_id_column(self, super_merged_csv, tmp_path):
        """Row log should contain vehicle_id, contention_n_vehicles, contention_utilization."""
        from scripts.feedback_loop import run_multi_vehicle_loop

        mock_api = self._make_mock_api()
        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", side_effect=lambda **kw: self._make_mock_qsim()), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            run_multi_vehicle_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
                num_vehicles=3,
            )

        log_df = pd.read_csv(os.path.join(output_dir, "feedback_multi_log.csv"))
        assert "vehicle_id" in log_df.columns
        assert "contention_n_vehicles" in log_df.columns
        assert "contention_utilization" in log_df.columns
        # 25 rows x 3 vehicles = 75 log entries
        assert len(log_df) == 75

    def test_total_packets_is_vehicles_times_rows(self, super_merged_csv, tmp_path):
        """Total packets should be num_vehicles * num_rows."""
        from scripts.feedback_loop import run_multi_vehicle_loop

        mock_api = self._make_mock_api()
        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", side_effect=lambda **kw: self._make_mock_qsim()), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_multi_vehicle_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
                num_vehicles=5,
            )

        assert summary["total_packets"] == 25 * 5

    def test_contention_log_has_all_timesteps(self, super_merged_csv, tmp_path):
        """Contention log should have one row per time step."""
        from scripts.feedback_loop import run_multi_vehicle_loop

        mock_api = self._make_mock_api()
        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", side_effect=lambda **kw: self._make_mock_qsim()), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            run_multi_vehicle_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
                num_vehicles=3,
            )

        ct_df = pd.read_csv(os.path.join(output_dir, "feedback_multi_contention.csv"))
        assert len(ct_df) == 25
        assert "n_dsrc" in ct_df.columns
        assert "n_pc5" in ct_df.columns
        assert "n_5g" in ct_df.columns
        assert "utilization_dsrc" in ct_df.columns
        assert "utilization_pc5" in ct_df.columns
        assert "utilization_5g" in ct_df.columns

    def test_seed_deterministic(self, super_merged_csv, tmp_path):
        """Same seed should produce same results."""
        from scripts.feedback_loop import run_multi_vehicle_loop

        summaries = []
        for run_idx in range(2):
            mock_api = self._make_mock_api()
            output_dir = str(tmp_path / f"output_{run_idx}")
            model_dir = str(tmp_path / f"models_{run_idx}")

            with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
                 patch("scripts.feedback_loop.QueueSimulator", side_effect=lambda **kw: self._make_mock_qsim()), \
                 patch("config.OUTPUT_DIR", output_dir), \
                 patch("scripts.feedback_loop.MODEL_DIR", model_dir):

                s = run_multi_vehicle_loop(
                    input_csv=super_merged_csv,
                    model_type="lstm",
                    seed=42,
                    retrain_interval=9999,
                    num_vehicles=3,
                )
                summaries.append(s)

        assert summaries[0]["achieved_pdr"] == summaries[1]["achieved_pdr"]
        assert summaries[0]["mean_sim_latency"] == summaries[1]["mean_sim_latency"]

    def test_unavailable_vehicles_skipped(self, super_merged_csv, tmp_path):
        """When API returns UNAVAILABLE for some calls, those are skipped."""
        from scripts.feedback_loop import run_multi_vehicle_loop

        mock_api = self._make_mock_api()
        call_count = {"n": 0}
        def side_effect_select_rat(state, queue_ctx=None):
            call_count["n"] += 1
            # Make every 3rd call UNAVAILABLE
            if call_count["n"] % 3 == 0:
                return RATDecision(
                    selected_rat=RATType.UNAVAILABLE,
                    confidence=0.0,
                    predicted_latency_ms=0.0,
                    predicted_pdr=0.0,
                    all_predictions={},
                    model_type="lstm",
                )
            return RATDecision(
                selected_rat=RATType.PC5,
                confidence=0.85,
                predicted_latency_ms=10.0,
                predicted_pdr=0.96,
                all_predictions={RATType.PC5: (10.0, 0.96)},
                model_type="lstm",
            )

        mock_api.select_rat.side_effect = side_effect_select_rat
        output_dir = str(tmp_path / "output")
        model_dir = str(tmp_path / "models")

        with patch("scripts.feedback_loop.RATSelectionAPI", return_value=mock_api), \
             patch("scripts.feedback_loop.QueueSimulator", side_effect=lambda **kw: self._make_mock_qsim()), \
             patch("config.OUTPUT_DIR", output_dir), \
             patch("scripts.feedback_loop.MODEL_DIR", model_dir):

            summary = run_multi_vehicle_loop(
                input_csv=super_merged_csv,
                model_type="lstm",
                seed=42,
                retrain_interval=9999,
                num_vehicles=3,
            )

        # 25 rows * 3 vehicles = 75 calls, every 3rd skipped = 50 processed
        assert summary["total_packets"] == 50
