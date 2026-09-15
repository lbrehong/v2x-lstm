"""
Tests for selection/api.py - RATSelectionAPI and JointController.

Tests cover:
- RATSelectionAPI initialization and configuration
- State to features conversion
- Sequence building with history management
- RAT selection logic and thresholds
- Packet size recommendations
- Confidence computation
- JointController orchestration

Uses mocking for model-dependent functionality to ensure tests run
without requiring trained models.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_types import (
    RATType, NetworkState, QueueContext, RATDecision,
    PacketSizeDecision, TransmissionOutcome,
)
from config import (
    PDR_RELIABILITY_THRESHOLD, PDR_AVAILABILITY_THRESHOLD,
    LATENCY_TIE_MARGIN_MS, TIMESTEPS, PACKET_SIZE_BOUNDS,
)
import torch
import torch.nn as nn


class ConstantModel(nn.Module):
    """Stand-in for a RATPredictor that always predicts the same normalized latency and PDR."""

    def __init__(self, latency, pdr):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # predict_torch reads the device from parameters
        self.latency = latency
        self.pdr = pdr

    def forward(self, x):
        batch = x.shape[0]
        return torch.full((batch, 1), self.latency), torch.full((batch, 1), self.pdr)


def _observe(api, state, n=TIMESTEPS):
    """Record the same measured state n times, filling the model input window."""
    for _ in range(n):
        api.observe(state)


class TestRATSelectionAPIInitialization:
    """Tests for RATSelectionAPI initialization."""

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_torch_model')
    def test_api_initialization(self, mock_load_model, mock_get_latest_model):
        """API should initialize with model type and load models."""
        mock_get_latest_model.return_value = None  # No models available

        from selection.api import RATSelectionAPI
        api = RATSelectionAPI(model_type="lstm")

        assert api.model_type == "lstm"
        assert api.pdr_threshold == PDR_RELIABILITY_THRESHOLD
        assert api.pdr_availability == PDR_AVAILABILITY_THRESHOLD
        assert api.latency_margin == LATENCY_TIE_MARGIN_MS
        assert api.retrain_interval == 500

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_torch_model')
    def test_api_initialization_with_gru(self, mock_load_model, mock_get_latest_model):
        """API should support gru model type."""
        mock_get_latest_model.return_value = None

        from selection.api import RATSelectionAPI
        api = RATSelectionAPI(model_type="gru")

        assert api.model_type == "gru"

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_torch_model')
    def test_api_history_initialized_empty(self, mock_load_model, mock_get_latest_model):
        """API history should be initialized as empty for the default RATs (no DSRC)."""
        mock_get_latest_model.return_value = None

        from selection.api import RATSelectionAPI
        api = RATSelectionAPI(model_type="lstm")

        assert set(api._history) == {"pc5", "5g"}
        assert all(len(h) == 0 for h in api._history.values())

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_torch_model')
    def test_api_history_includes_dsrc_when_selected(self, mock_load_model, mock_get_latest_model):
        """Passing all RATs should restore a DSRC history."""
        mock_get_latest_model.return_value = None

        from config import ALL_RATS
        from selection.api import RATSelectionAPI
        api = RATSelectionAPI(model_type="lstm", rats=ALL_RATS)

        assert set(api._history) == {"dsrc", "pc5", "5g"}

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_torch_model')
    def test_dsrc_model_not_loaded_by_default(self, mock_load_model, mock_get_latest_model):
        """An existing DSRC model must not be loaded when DSRC is not selected."""
        mock_get_latest_model.side_effect = lambda model_type, rat, model_dir: f"/fake/{model_type}_{rat}.pt"
        mock_load_model.side_effect = lambda path: path

        from selection.api import RATSelectionAPI
        api = RATSelectionAPI(model_type="lstm")

        assert set(api.models) == {"pc5", "5g"}
        requested_rats = {call.args[1] for call in mock_get_latest_model.call_args_list}
        assert "dsrc" not in requested_rats

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_torch_model')
    def test_invalid_rat_rejected(self, mock_load_model, mock_get_latest_model):
        """Unknown RAT identifiers should raise."""
        mock_get_latest_model.return_value = None

        from selection.api import RATSelectionAPI
        with pytest.raises(ValueError):
            RATSelectionAPI(model_type="lstm", rats=["5g", "wifi"])


class TestRATSelectionAPIThresholds:
    """Tests for threshold configuration."""

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_torch_model')
    def test_set_thresholds(self, mock_load_model, mock_get_latest_model):
        """set_thresholds should update threshold values."""
        mock_get_latest_model.return_value = None

        from selection.api import RATSelectionAPI
        api = RATSelectionAPI(model_type="lstm")

        api.set_thresholds(
            pdr_reliability=0.95,
            pdr_availability=0.2,
            latency_tie_margin_ms=2.0
        )

        assert api.pdr_threshold == 0.95
        assert api.pdr_availability == 0.2
        assert api.latency_margin == 2.0

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_torch_model')
    def test_set_thresholds_partial(self, mock_load_model, mock_get_latest_model):
        """set_thresholds should only update provided values."""
        mock_get_latest_model.return_value = None

        from selection.api import RATSelectionAPI
        api = RATSelectionAPI(model_type="lstm")

        original_pdr_availability = api.pdr_availability
        api.set_thresholds(pdr_reliability=0.95)

        assert api.pdr_threshold == 0.95
        assert api.pdr_availability == original_pdr_availability  # Unchanged


class TestStateToFeatures:
    """Tests for _state_to_features conversion."""

    @pytest.fixture
    def api_instance(self):
        """Create API instance with mocked model loading."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_torch_model') as mock_load:
            mock_get.return_value = None
            from selection.api import RATSelectionAPI
            return RATSelectionAPI(model_type="lstm")

    @pytest.fixture
    def sample_state(self):
        """Create sample NetworkState."""
        return NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
            dsrc_latency_ms=12.0,
            dsrc_pdr=0.98,
            dsrc_rsrp_1=-85.0,
            dsrc_rsrp_2=-90.0,
            pc5_latency_ms=10.0,
            pc5_pdr=0.99,
            fiveg_latency_ms=15.0,
            fiveg_pdr=0.995,
            fiveg_sinr=25.0,
            fiveg_rsrp=-95.0,
        )

    def test_state_to_features_5g(self, api_instance, sample_state):
        """5G features should have 6 elements: lat, lon, latency, sinr, rsrp, pdr."""
        features = api_instance._state_to_features(sample_state, "5g")

        assert len(features) == 6
        assert isinstance(features, np.ndarray)
        # Check PDR is last element
        assert features[5] == 0.995

    def test_state_to_features_pc5(self, api_instance, sample_state):
        """PC5 features should have 4 elements: lat, lon, latency, pdr."""
        features = api_instance._state_to_features(sample_state, "pc5")

        assert len(features) == 4
        # Check PDR is last element
        assert features[3] == 0.99

    def test_state_to_features_dsrc(self, api_instance, sample_state):
        """DSRC features should have 6 elements: lat, lon, rsrp_1, rsrp_2, latency, pdr."""
        features = api_instance._state_to_features(sample_state, "dsrc")

        assert len(features) == 6
        # RSRP values (normalized via DSRC RSRP scaler: [-150, -45])
        assert features[2] == pytest.approx((-85.0 - (-150)) / (-45 - (-150)), abs=1e-6)
        assert features[3] == pytest.approx((-90.0 - (-150)) / (-45 - (-150)), abs=1e-6)
        # PDR is last element
        assert features[5] == 0.98

    def test_state_to_features_invalid_rat(self, api_instance, sample_state):
        """Invalid RAT should raise ValueError."""
        with pytest.raises(ValueError):
            api_instance._state_to_features(sample_state, "wifi")

    def test_state_to_features_none_values(self, api_instance):
        """Missing measurements should be NaN, never zero-filled."""
        state = NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
            # All RAT-specific values are None
        )
        features = api_instance._state_to_features(state, "5g")

        assert np.isfinite(features[:2]).all()  # GPS is present
        assert np.isnan(features[2:]).all()

    @pytest.mark.parametrize("rat", ["5g", "pc5", "dsrc"])
    def test_state_to_features_matches_training_scaling(self, sample_state, rat):
        """Per-step features must equal the scaling used to build training data."""
        import pandas as pd
        from config import ALL_RATS
        from learning.data_preprocessing import normalize_features
        from selection.api import RATSelectionAPI

        with patch('selection.api.get_latest_model', return_value=None):
            api = RATSelectionAPI(model_type="lstm", rats=ALL_RATS)

        raw = {
            "5g": dict(latency_ms=15.0, sinr=25.0, rsrp=-95.0, pdr=0.995),
            "pc5": dict(latency_ms=10.0, pdr=0.99),
            "dsrc": dict(rsrp_1=-85.0, rsrp_2=-90.0, latency_ms=12.0, pdr=0.98),
        }[rat]
        row = pd.DataFrame([{"tx_latitude": 43.560, "tx_longitude": 1.467, **raw}])

        expected = normalize_features(row, rat).to_numpy()[0]
        np.testing.assert_allclose(api._state_to_features(sample_state, rat), expected, rtol=1e-9)


class TestSequenceBuilding:
    """Tests for sequence building with history management."""

    @pytest.fixture
    def api_instance(self):
        """Create API instance with mocked model loading."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_torch_model') as mock_load:
            mock_get.return_value = None
            from selection.api import RATSelectionAPI
            return RATSelectionAPI(model_type="lstm")

    @pytest.fixture
    def sample_state(self):
        """Create sample NetworkState."""
        return NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
            pc5_latency_ms=10.0,
            pc5_pdr=0.99,
        )

    def test_input_window_none_until_full_history(self, api_instance, sample_state):
        """No zero padding: a window exists only after TIMESTEPS observations."""
        _observe(api_instance, sample_state, TIMESTEPS - 1)
        assert api_instance.input_window("pc5") is None

        api_instance.observe(sample_state)
        window = api_instance.input_window("pc5")
        assert window.shape == (1, TIMESTEPS, 4)  # PC5 has 4 features
        assert np.isfinite(window).all()

    def test_observe_caps_history(self, api_instance, sample_state):
        """History keeps only the last TIMESTEPS observations."""
        _observe(api_instance, sample_state, TIMESTEPS + 5)

        assert len(api_instance._history["pc5"]) == TIMESTEPS
        assert api_instance._last_state is sample_state

    def test_input_window_none_with_missing_measurements(self, api_instance, sample_state):
        """A window containing a missing measurement gives no model input."""
        _observe(api_instance, sample_state)  # sample_state has no 5G measurements

        assert api_instance.input_window("pc5") is not None
        assert api_instance.input_window("5g") is None

    def test_deciding_does_not_record_the_decided_state(self, api_instance, sample_state):
        """select_rat must not use or record the state it decides for (no lookahead)."""
        api_instance.models = {"pc5": ConstantModel(latency=0.1, pdr=0.995)}
        _observe(api_instance, sample_state)
        window_before = api_instance.input_window("pc5").copy()

        decided_state = NetworkState(
            timestamp_ms=1700000000000, latitude=43.570, longitude=1.480,
            pc5_latency_ms=90.0, pc5_pdr=0.0,
        )
        decision = api_instance.select_rat(decided_state)
        api_instance.select_rat_opportunistic(decided_state)

        np.testing.assert_array_equal(api_instance.input_window("pc5"), window_before)
        assert api_instance._last_state is sample_state
        assert decision.selected_rat == RATType.PC5

        api_instance.observe(decided_state)
        assert not np.array_equal(api_instance.input_window("pc5"), window_before)

    def test_opportunistic_uses_last_observed_state(self, api_instance, sample_state):
        """Opportunistic is UNAVAILABLE before any observation, then uses the last one."""
        assert api_instance.select_rat_opportunistic(sample_state).selected_rat == RATType.UNAVAILABLE

        api_instance.observe(sample_state)  # PC5 good, no 5G
        pc5_down_now = NetworkState(
            timestamp_ms=1700000000000, latitude=43.560, longitude=1.467,
            pc5_latency_ms=10.0, pc5_pdr=0.0,
        )
        decision = api_instance.select_rat_opportunistic(pc5_down_now)

        assert decision.selected_rat == RATType.PC5

    def test_select_rat_falls_back_to_5g_without_predictions(self, api_instance, sample_state):
        """No full measurement window for any RAT: 5G is chosen and flagged as a fallback."""
        api_instance.models = {"pc5": ConstantModel(latency=0.1, pdr=0.995)}

        decision = api_instance.select_rat(sample_state)  # nothing observed yet

        assert decision.selected_rat == RATType.FiveG
        assert decision.fallback is True
        assert decision.confidence == 0.0
        assert decision.all_predictions == {}
        assert np.isnan(decision.predicted_pdr) and np.isnan(decision.predicted_latency_ms)

        _observe(api_instance, sample_state)
        decision = api_instance.select_rat(sample_state)
        assert decision.fallback is False and decision.selected_rat == RATType.PC5

    def test_no_fallback_when_5g_not_active(self, sample_state):
        from selection.api import RATSelectionAPI

        with patch('selection.api.get_latest_model', return_value=None):
            api = RATSelectionAPI(model_type="lstm", rats=("pc5",))

        decision = api.select_rat(sample_state)

        assert decision.selected_rat == RATType.UNAVAILABLE
        assert decision.fallback is False

    def test_no_fallback_when_disabled(self, api_instance, sample_state, monkeypatch):
        monkeypatch.setattr("selection.api.LSTM_FALLBACK_RAT", None)

        assert api_instance.select_rat(sample_state).selected_rat == RATType.UNAVAILABLE

    def test_select_rat_falls_back_when_predicted_rats_unavailable_and_5g_unpredicted(self, api_instance):
        """PC5 predicted but unreliable and last seen down, 5G not predicted: 5G fallback."""
        api_instance.models = {"pc5": ConstantModel(latency=0.1, pdr=0.5)}
        pc5_down = NetworkState(
            timestamp_ms=1699999999000, latitude=43.560, longitude=1.467,
            pc5_latency_ms=10.0, pc5_pdr=0.01,
        )
        _observe(api_instance, pc5_down)

        decision = api_instance.select_rat(pc5_down)

        assert decision.selected_rat == RATType.FiveG
        assert decision.fallback is True
        assert RATType.PC5 in decision.all_predictions  # the PC5 prediction is still reported

    def test_context_round_trip(self, api_instance, sample_state):
        """get_context/set_context restore history, last state and sticky RAT without aliasing."""
        _observe(api_instance, sample_state, 3)
        api_instance._previous_rat = RATType.PC5
        context = api_instance.get_context()

        _observe(api_instance, sample_state, 5)
        api_instance._previous_rat = RATType.FiveG
        api_instance.set_context(context)

        assert len(api_instance._history["pc5"]) == 3
        assert api_instance._previous_rat == RATType.PC5
        api_instance.observe(sample_state)
        assert len(context["history"]["pc5"]) == 3

    def test_reset_history(self, api_instance, sample_state):
        """reset_history should clear all history."""
        _observe(api_instance, sample_state, 5)

        api_instance.reset_history()

        assert len(api_instance._history["pc5"]) == 0
        assert len(api_instance._history["5g"]) == 0


class TestPacketSizeRecommendation:
    """Tests for packet size recommendation logic."""

    @pytest.fixture
    def api_instance(self):
        """Create API instance with mocked model loading."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_torch_model') as mock_load:
            mock_get.return_value = None
            from selection.api import RATSelectionAPI
            return RATSelectionAPI(model_type="lstm")

    def test_recommend_packet_size_high_pdr(self, api_instance):
        """High PDR (>=0.99) should recommend maximum packet size."""
        size = api_instance._recommend_packet_size(RATType.PC5, 0.99)
        assert size == PACKET_SIZE_BOUNDS["pc5"]["max"]

        size = api_instance._recommend_packet_size(RATType.PC5, 1.0)
        assert size == PACKET_SIZE_BOUNDS["pc5"]["max"]

    def test_recommend_packet_size_low_pdr(self, api_instance):
        """Low PDR (<=0.85) should recommend minimum packet size."""
        size = api_instance._recommend_packet_size(RATType.PC5, 0.85)
        assert size == PACKET_SIZE_BOUNDS["pc5"]["min"]

        size = api_instance._recommend_packet_size(RATType.PC5, 0.5)
        assert size == PACKET_SIZE_BOUNDS["pc5"]["min"]

    def test_recommend_packet_size_mid_pdr(self, api_instance):
        """Mid-range PDR should interpolate between min and max."""
        size = api_instance._recommend_packet_size(RATType.PC5, 0.92)
        min_size = PACKET_SIZE_BOUNDS["pc5"]["min"]
        max_size = PACKET_SIZE_BOUNDS["pc5"]["max"]

        assert min_size < size < max_size

    def test_recommend_packet_size_unavailable_rat(self, api_instance):
        """UNAVAILABLE RAT should return None."""
        size = api_instance._recommend_packet_size(RATType.UNAVAILABLE, 0.99)
        assert size is None

    def test_recommend_packet_size_each_rat(self, api_instance):
        """Each RAT should have different size bounds."""
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            size = api_instance._recommend_packet_size(rat, 0.99)
            assert size == PACKET_SIZE_BOUNDS[rat.value]["max"]


class TestRATSelectionWithMockedModels:
    """Tests for RAT selection logic with mocked models."""

    @pytest.fixture
    def api_with_mock_models(self, sample_state):
        """Create API with mocked models that return predictable values."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_torch_model') as mock_load:

            # Model returns (normalized latency, pdr)
            mock_model = ConstantModel(latency=0.3, pdr=0.995)

            mock_get.return_value = "/fake/model/path.pt"
            mock_load.return_value = mock_model

            from config import ALL_RATS
            from selection.api import RATSelectionAPI
            api = RATSelectionAPI(model_type="lstm", rats=ALL_RATS)

            # Set models manually since they're loaded via mocks
            api.models = {
                "dsrc": mock_model,
                "pc5": mock_model,
                "5g": mock_model,
            }
            _observe(api, sample_state)

            return api

    @pytest.fixture
    def sample_state(self):
        """Create sample NetworkState with all RATs available."""
        return NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
            dsrc_latency_ms=12.0,
            dsrc_pdr=0.98,
            dsrc_rsrp_1=-85.0,
            dsrc_rsrp_2=-90.0,
            pc5_latency_ms=10.0,
            pc5_pdr=0.99,
            fiveg_latency_ms=15.0,
            fiveg_pdr=0.995,
            fiveg_sinr=25.0,
            fiveg_rsrp=-95.0,
        )

    def test_select_rat_returns_decision(self, api_with_mock_models, sample_state):
        """select_rat should return a RATDecision object."""
        decision = api_with_mock_models.select_rat(sample_state)

        assert isinstance(decision, RATDecision)
        assert decision.model_type == "lstm"

    def test_select_rat_with_queue_context(self, api_with_mock_models, sample_state):
        """select_rat should accept queue context."""
        queue_ctx = QueueContext(
            queue_depth=10,
            avg_packet_size_bytes=800,
            urgency_level=0.5,
            recent_pdr_trend=0.0,
            target_latency_ms=50.0,
            target_pdr=0.99,
        )

        decision = api_with_mock_models.select_rat(sample_state, queue_ctx)
        assert isinstance(decision, RATDecision)

    def test_select_rat_with_contention_context(self, api_with_mock_models, sample_state):
        """select_rat should accept contention context and adjust PDR filtering."""
        # Contention context: PC5 is heavily overloaded (5 vehicles, util=3.0)
        contention_ctx = {
            RATType.DSRC: (1, 0.5),
            RATType.PC5: (5, 3.0),
            RATType.FiveG: (1, 0.05),
        }
        decision = api_with_mock_models.select_rat(
            sample_state, contention_context=contention_ctx,
        )
        assert isinstance(decision, RATDecision)
        # With heavy PC5 contention, PC5 should not be selected
        # (contention-corrected PDR will be very low)
        assert decision.selected_rat != RATType.PC5

    def test_select_rat_without_contention_context(self, api_with_mock_models, sample_state):
        """select_rat without contention context should use raw pred_pdr."""
        decision = api_with_mock_models.select_rat(sample_state)
        assert isinstance(decision, RATDecision)
        # Without contention, all RATs are candidates (raw pred_pdr is high)

    def test_fallback_with_contention_prefers_5g(self, sample_state):
        """When contention context is provided and no RAT passes threshold, 5G is selected.

        Even if 5G has low predicted PDR, the contention-aware fallback
        should unconditionally prefer 5G (scheduled access, contention-free)
        because low predictions early in simulation are model warm-up artifacts.
        """
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_torch_model') as mock_load:
            mock_get.return_value = "/fake/model/path.pt"

            # All models predict low PDR (below threshold of 0.99)
            # PDR well below 0.99 threshold
            mock_model_low = ConstantModel(latency=0.1, pdr=0.50)

            mock_load.return_value = mock_model_low

            from config import ALL_RATS
            from selection.api import RATSelectionAPI
            api = RATSelectionAPI(model_type="lstm", rats=ALL_RATS)
            api.models = {"dsrc": mock_model_low, "pc5": mock_model_low, "5g": mock_model_low}
            _observe(api, sample_state)

            contention_ctx = {
                RATType.DSRC: (3, 0.8),
                RATType.PC5: (3, 0.8),
                RATType.FiveG: (3, 0.1),
            }

            decision = api.select_rat(sample_state, contention_context=contention_ctx)

            assert decision.selected_rat == RATType.FiveG

    def test_fallback_without_contention_uses_max_effective_pdr(self, sample_state):
        """Without contention context, fallback picks RAT with highest effective PDR.

        When no RAT passes the PDR threshold and contention_context is None,
        the original fallback logic applies: select the RAT with the highest
        effective_pdr among those whose actual_pdr >= pdr_availability.
        """
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_torch_model') as mock_load:
            mock_get.return_value = "/fake/model/path.pt"

            # DSRC predicts highest PDR (but still below threshold)
            mock_dsrc = ConstantModel(latency=0.1, pdr=0.95)  # highest among the three, still < 0.99
            mock_pc5 = ConstantModel(latency=0.1, pdr=0.90)
            mock_5g = ConstantModel(latency=0.1, pdr=0.85)  # lowest

            mock_load.return_value = mock_dsrc  # default for loading

            from config import ALL_RATS
            from selection.api import RATSelectionAPI
            api = RATSelectionAPI(model_type="lstm", rats=ALL_RATS)
            api.models = {"dsrc": mock_dsrc, "pc5": mock_pc5, "5g": mock_5g}
            _observe(api, sample_state)

            # No contention context — original fallback
            decision = api.select_rat(sample_state, contention_context=None)

            # DSRC has highest effective_pdr (0.95) and actual_pdr >= availability
            assert decision.selected_rat == RATType.DSRC

    def test_default_rats_never_select_dsrc(self, sample_state):
        """With the default RAT set, DSRC is ignored even when its model is the best."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_torch_model'):
            mock_get.return_value = None

            from selection.api import RATSelectionAPI
            api = RATSelectionAPI(model_type="lstm")
            api.models = {
                "dsrc": ConstantModel(latency=0.01, pdr=1.0),  # best on every metric
                "pc5": ConstantModel(latency=0.3, pdr=0.995),
                "5g": ConstantModel(latency=0.5, pdr=0.995),
            }
            _observe(api, sample_state)

            decision = api.select_rat(sample_state)
            opp_decision = api.select_rat_opportunistic(sample_state)

            assert decision.selected_rat == RATType.PC5
            assert RATType.DSRC not in decision.all_predictions
            assert opp_decision.selected_rat != RATType.DSRC
            assert RATType.DSRC not in opp_decision.all_predictions

    def test_fallback_with_contention_no_5g_model(self, sample_state):
        """With contention context but no 5G model, falls through to generic fallback.

        If contention_context is provided but no 5G model is loaded,
        the 5G-preferred path cannot be taken, so the code falls through
        to the generic max(effective_pdr) fallback among available RATs.
        """
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_torch_model') as mock_load:
            mock_get.return_value = "/fake/model/path.pt"

            # Only DSRC and PC5 models, no 5G
            mock_dsrc = ConstantModel(latency=0.1, pdr=0.90)  # below threshold
            mock_pc5 = ConstantModel(latency=0.1, pdr=0.93)  # below threshold, but highest

            mock_load.return_value = mock_dsrc

            from config import ALL_RATS
            from selection.api import RATSelectionAPI
            api = RATSelectionAPI(model_type="lstm", rats=ALL_RATS)
            # No 5G model loaded
            api.models = {"dsrc": mock_dsrc, "pc5": mock_pc5}
            _observe(api, sample_state)

            contention_ctx = {
                RATType.DSRC: (3, 0.5),
                RATType.PC5: (3, 0.5),
            }

            decision = api.select_rat(sample_state, contention_context=contention_ctx)

            # 5G is not available, so fallback picks max effective_pdr
            assert decision.selected_rat != RATType.FiveG
            assert decision.selected_rat in (RATType.DSRC, RATType.PC5)


class TestOutcomeReporting:
    """Tests for transmission outcome reporting."""

    @pytest.fixture
    def api_instance(self):
        """Create API instance with mocked model loading."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_torch_model') as mock_load:
            mock_get.return_value = None
            from selection.api import RATSelectionAPI
            return RATSelectionAPI(model_type="lstm")

    def test_report_outcome_buffers(self, api_instance):
        """report_outcome should buffer outcomes."""
        state = NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
        )
        outcome = TransmissionOutcome(
            timestamp_ms=1699999999100,
            rat_used=RATType.PC5,
            packet_size_bytes=800,
            actual_latency_ms=11.0,
            delivered=True,
            network_state=state,
        )

        api_instance.report_outcome(outcome)

        assert len(api_instance.outcome_buffer) == 1
        assert api_instance.outcome_buffer[0] == outcome

    def test_report_outcome_triggers_retrain(self, api_instance):
        """report_outcome should trigger retraining when buffer is full."""
        state = NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
        )

        # Fill buffer to retrain threshold
        api_instance.retrain_interval = 3  # Small for testing

        for i in range(3):
            outcome = TransmissionOutcome(
                timestamp_ms=1699999999000 + i * 100,
                rat_used=RATType.PC5,
                packet_size_bytes=800,
                actual_latency_ms=11.0,
                delivered=True,
                network_state=state,
            )
            api_instance.report_outcome(outcome)

        # Buffer should be cleared after retrain
        assert len(api_instance.outcome_buffer) == 0


class TestJointController:
    """Tests for JointController orchestration."""

    @pytest.fixture
    def mock_rat_selector(self):
        """Create a mock RAT selector."""
        mock = Mock()
        mock.select_rat.return_value = RATDecision(
            selected_rat=RATType.PC5,
            confidence=0.9,
            predicted_latency_ms=10.0,
            predicted_pdr=0.99,
            all_predictions={RATType.PC5: (10.0, 0.99)},
            model_type="lstm",
        )
        return mock

    @pytest.fixture
    def mock_queue_sim(self):
        """Create a mock queue simulator."""
        mock = Mock()
        mock.get_queue_context.return_value = QueueContext(
            queue_depth=10,
            avg_packet_size_bytes=800,
            urgency_level=0.5,
            recent_pdr_trend=0.0,
            target_latency_ms=50.0,
            target_pdr=0.99,
        )
        mock.decide_packet_size.return_value = PacketSizeDecision(
            packet_size_bytes=1000,
            fragment_count=1,
            priority_level=4,
            send_rate_hz=50.0,
        )
        return mock

    def test_joint_controller_creation(self, mock_rat_selector):
        """JointController should be created with RAT selector."""
        from selection.api import JointController
        controller = JointController(mock_rat_selector)

        assert controller.rat_selector == mock_rat_selector
        assert controller.queue_sim is None

    def test_joint_controller_with_queue_sim(self, mock_rat_selector, mock_queue_sim):
        """JointController should accept queue simulator."""
        from selection.api import JointController
        controller = JointController(mock_rat_selector, mock_queue_sim)

        assert controller.queue_sim == mock_queue_sim

    def test_process_transmission_without_queue_sim(self, mock_rat_selector):
        """process_transmission should work without queue simulator."""
        from selection.api import JointController
        controller = JointController(mock_rat_selector)

        state = NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
        )

        rat_decision, packet_decision = controller.process_transmission(state)

        assert rat_decision is not None
        assert packet_decision is None
        mock_rat_selector.select_rat.assert_called_once()

    def test_process_transmission_with_queue_sim(self, mock_rat_selector, mock_queue_sim):
        """process_transmission should coordinate with queue simulator."""
        from selection.api import JointController
        controller = JointController(mock_rat_selector, mock_queue_sim)

        state = NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
        )

        rat_decision, packet_decision = controller.process_transmission(state)

        assert rat_decision is not None
        assert packet_decision is not None
        mock_queue_sim.get_queue_context.assert_called_once()
        mock_queue_sim.decide_packet_size.assert_called_once()

    def test_report_outcome_to_both(self, mock_rat_selector, mock_queue_sim):
        """report_outcome should report to both RAT selector and queue sim."""
        from selection.api import JointController
        controller = JointController(mock_rat_selector, mock_queue_sim)

        state = NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
        )
        outcome = TransmissionOutcome(
            timestamp_ms=1699999999100,
            rat_used=RATType.PC5,
            packet_size_bytes=800,
            actual_latency_ms=11.0,
            delivered=True,
            network_state=state,
        )

        controller.report_outcome(outcome)

        mock_rat_selector.report_outcome.assert_called_once_with(outcome)
        mock_queue_sim.update_pdr_estimate.assert_called_once_with(outcome)


class TestConfidenceComputation:
    """Tests for confidence score computation."""

    @pytest.fixture
    def api_instance(self):
        """Create API instance with mocked model loading."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_torch_model') as mock_load:
            mock_get.return_value = None
            from selection.api import RATSelectionAPI
            return RATSelectionAPI(model_type="lstm")

    def test_confidence_unavailable_rat(self, api_instance):
        """UNAVAILABLE RAT should have zero confidence."""
        state = NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
        )

        confidence = api_instance._compute_confidence(
            RATType.UNAVAILABLE,
            {},
            state
        )

        assert confidence == 0.0

    def test_confidence_with_high_pdr(self, api_instance):
        """High PDR predictions should increase confidence."""
        state = NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
            pc5_pdr=0.995,
        )

        all_predictions = {
            RATType.PC5: (10.0, 0.999),  # High PDR
        }

        confidence = api_instance._compute_confidence(
            RATType.PC5,
            all_predictions,
            state
        )

        # Confidence should be relatively high
        assert confidence > 0.5

    def test_confidence_bounded_0_to_1(self, api_instance):
        """Confidence should always be between 0 and 1."""
        state = NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
            pc5_pdr=0.5,  # Low actual PDR
        )

        # Various prediction scenarios
        for pdr in [0.0, 0.5, 0.99, 1.0]:
            all_predictions = {RATType.PC5: (10.0, pdr)}
            confidence = api_instance._compute_confidence(
                RATType.PC5,
                all_predictions,
                state
            )
            assert 0.0 <= confidence <= 1.0
