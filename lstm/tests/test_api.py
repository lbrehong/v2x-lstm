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


class TestRATSelectionAPIInitialization:
    """Tests for RATSelectionAPI initialization."""

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_model')
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
    @patch('selection.api.load_model')
    def test_api_initialization_with_gru(self, mock_load_model, mock_get_latest_model):
        """API should support gru model type."""
        mock_get_latest_model.return_value = None

        from selection.api import RATSelectionAPI
        api = RATSelectionAPI(model_type="gru")

        assert api.model_type == "gru"

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_model')
    def test_api_history_initialized_empty(self, mock_load_model, mock_get_latest_model):
        """API history should be initialized as empty for all RATs."""
        mock_get_latest_model.return_value = None

        from selection.api import RATSelectionAPI
        api = RATSelectionAPI(model_type="lstm")

        assert "dsrc" in api._history
        assert "pc5" in api._history
        assert "5g" in api._history
        assert len(api._history["dsrc"]) == 0
        assert len(api._history["pc5"]) == 0
        assert len(api._history["5g"]) == 0


class TestRATSelectionAPIThresholds:
    """Tests for threshold configuration."""

    @patch('selection.api.get_latest_model')
    @patch('selection.api.load_model')
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
    @patch('selection.api.load_model')
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
             patch('selection.api.load_model') as mock_load:
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
            fiveg_sinr=320.0,
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
        """None values should be converted to 0.0."""
        state = NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560,
            longitude=1.467,
            # All RAT-specific values are None
        )
        features = api_instance._state_to_features(state, "5g")

        # Latency should be normalized 0
        # PDR should be 0
        assert features[5] == 0.0  # PDR


class TestSequenceBuilding:
    """Tests for sequence building with history management."""

    @pytest.fixture
    def api_instance(self):
        """Create API instance with mocked model loading."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_model') as mock_load:
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

    def test_build_sequence_initial(self, api_instance, sample_state):
        """First call should build sequence with padding."""
        sequence = api_instance._build_sequence(sample_state, "pc5")

        assert sequence.shape == (1, TIMESTEPS, 4)  # PC5 has 4 features
        # First TIMESTEPS-1 should be zeros (padding)
        assert np.all(sequence[0, :TIMESTEPS-1, :] == 0)

    def test_build_sequence_accumulates_history(self, api_instance, sample_state):
        """Multiple calls should accumulate history."""
        # Build 5 sequences
        for i in range(5):
            api_instance._build_sequence(sample_state, "pc5")

        assert len(api_instance._history["pc5"]) == 5

    def test_build_sequence_max_history(self, api_instance, sample_state):
        """History should be capped at TIMESTEPS."""
        # Build more than TIMESTEPS sequences
        for i in range(TIMESTEPS + 5):
            api_instance._build_sequence(sample_state, "pc5")

        assert len(api_instance._history["pc5"]) == TIMESTEPS

    def test_build_sequence_shape_consistency(self, api_instance, sample_state):
        """Sequence shape should be consistent regardless of history length."""
        for i in range(TIMESTEPS + 3):
            sequence = api_instance._build_sequence(sample_state, "pc5")
            assert sequence.shape == (1, TIMESTEPS, 4)

    def test_reset_history(self, api_instance, sample_state):
        """reset_history should clear all history."""
        # Build some history
        for i in range(5):
            api_instance._build_sequence(sample_state, "pc5")
            api_instance._build_sequence(sample_state, "dsrc")

        api_instance.reset_history()

        assert len(api_instance._history["pc5"]) == 0
        assert len(api_instance._history["dsrc"]) == 0
        assert len(api_instance._history["5g"]) == 0


class TestPacketSizeRecommendation:
    """Tests for packet size recommendation logic."""

    @pytest.fixture
    def api_instance(self):
        """Create API instance with mocked model loading."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_model') as mock_load:
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
    def api_with_mock_models(self):
        """Create API with mocked models that return predictable values."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_model') as mock_load:

            # Create mock models
            mock_model = Mock()
            # Model returns [latency_output, pdr_output]
            mock_model.predict.return_value = [
                np.array([[0.3]]),  # normalized latency
                np.array([[0.995]]),  # pdr
            ]

            mock_get.return_value = "/fake/model/path.keras"
            mock_load.return_value = mock_model

            from selection.api import RATSelectionAPI
            api = RATSelectionAPI(model_type="lstm")

            # Set models manually since they're loaded via mocks
            api.models = {
                "dsrc": mock_model,
                "pc5": mock_model,
                "5g": mock_model,
            }

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
            fiveg_sinr=320.0,
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


class TestOutcomeReporting:
    """Tests for transmission outcome reporting."""

    @pytest.fixture
    def api_instance(self):
        """Create API instance with mocked model loading."""
        with patch('selection.api.get_latest_model') as mock_get, \
             patch('selection.api.load_model') as mock_load:
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
             patch('selection.api.load_model') as mock_load:
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
