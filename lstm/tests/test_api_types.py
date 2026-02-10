"""
Tests for api_types.py - Data structures for RAT Selection + Queue Simulator Integration.

Tests cover:
- RATType enum and from_string conversion
- NetworkState to_dict/from_dict roundtrip
- QueueContext to_dict/from_dict roundtrip
- RATDecision to_dict/from_dict roundtrip
- PacketSizeDecision to_dict/from_dict roundtrip
- TransmissionOutcome to_dict/from_dict roundtrip with nested NetworkState
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_types import (
    RATType, NetworkState, QueueContext, RATDecision,
    PacketSizeDecision, TransmissionOutcome,
)


class TestRATType:
    """Tests for RATType enum."""

    def test_rat_type_values(self):
        """RATType enum should have expected values."""
        assert RATType.DSRC.value == "dsrc"
        assert RATType.PC5.value == "pc5"
        assert RATType.FiveG.value == "5g"
        assert RATType.UNAVAILABLE.value == "NaN"

    def test_from_string_dsrc(self):
        """from_string should correctly parse 'dsrc'."""
        assert RATType.from_string("dsrc") == RATType.DSRC
        assert RATType.from_string("DSRC") == RATType.DSRC
        assert RATType.from_string("Dsrc") == RATType.DSRC

    def test_from_string_pc5(self):
        """from_string should correctly parse 'pc5'."""
        assert RATType.from_string("pc5") == RATType.PC5
        assert RATType.from_string("PC5") == RATType.PC5

    def test_from_string_5g(self):
        """from_string should correctly parse '5g' and 'fiveg'."""
        assert RATType.from_string("5g") == RATType.FiveG
        assert RATType.from_string("5G") == RATType.FiveG
        assert RATType.from_string("fiveg") == RATType.FiveG
        assert RATType.from_string("FiveG") == RATType.FiveG

    def test_from_string_nan(self):
        """from_string should correctly parse 'nan'."""
        assert RATType.from_string("nan") == RATType.UNAVAILABLE
        assert RATType.from_string("NaN") == RATType.UNAVAILABLE
        assert RATType.from_string("NAN") == RATType.UNAVAILABLE

    def test_from_string_invalid(self):
        """from_string should raise ValueError for invalid input."""
        with pytest.raises(ValueError):
            RATType.from_string("invalid")
        with pytest.raises(ValueError):
            RATType.from_string("")
        with pytest.raises(ValueError):
            RATType.from_string("wifi")


class TestNetworkState:
    """Tests for NetworkState dataclass."""

    @pytest.fixture
    def sample_network_state(self):
        """Create a sample NetworkState with all fields populated."""
        return NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560123,
            longitude=1.467456,
            dsrc_latency_ms=12.5,
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

    @pytest.fixture
    def minimal_network_state(self):
        """Create a minimal NetworkState with only required fields."""
        return NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560123,
            longitude=1.467456,
        )

    def test_network_state_creation(self, sample_network_state):
        """NetworkState should be created with correct values."""
        state = sample_network_state
        assert state.timestamp_ms == 1699999999000
        assert state.latitude == 43.560123
        assert state.longitude == 1.467456
        assert state.dsrc_latency_ms == 12.5
        assert state.fiveg_pdr == 0.995

    def test_network_state_optional_fields_default_none(self, minimal_network_state):
        """Optional fields should default to None."""
        state = minimal_network_state
        assert state.dsrc_latency_ms is None
        assert state.dsrc_pdr is None
        assert state.pc5_latency_ms is None
        assert state.fiveg_sinr is None

    def test_network_state_to_dict(self, sample_network_state):
        """to_dict should return all fields."""
        d = sample_network_state.to_dict()
        assert d["timestamp_ms"] == 1699999999000
        assert d["latitude"] == 43.560123
        assert d["longitude"] == 1.467456
        assert d["dsrc_latency_ms"] == 12.5
        assert d["fiveg_pdr"] == 0.995
        assert "dsrc_rsrp_1" in d
        assert "fiveg_sinr" in d

    def test_network_state_from_dict(self, sample_network_state):
        """from_dict should recreate NetworkState correctly."""
        original = sample_network_state
        d = original.to_dict()
        restored = NetworkState.from_dict(d)

        assert restored.timestamp_ms == original.timestamp_ms
        assert restored.latitude == original.latitude
        assert restored.longitude == original.longitude
        assert restored.dsrc_latency_ms == original.dsrc_latency_ms
        assert restored.fiveg_pdr == original.fiveg_pdr

    def test_network_state_roundtrip(self, sample_network_state):
        """to_dict -> from_dict should preserve all values."""
        original = sample_network_state
        restored = NetworkState.from_dict(original.to_dict())

        # Check all fields
        assert restored.timestamp_ms == original.timestamp_ms
        assert restored.latitude == original.latitude
        assert restored.longitude == original.longitude
        assert restored.dsrc_latency_ms == original.dsrc_latency_ms
        assert restored.dsrc_pdr == original.dsrc_pdr
        assert restored.dsrc_rsrp_1 == original.dsrc_rsrp_1
        assert restored.dsrc_rsrp_2 == original.dsrc_rsrp_2
        assert restored.pc5_latency_ms == original.pc5_latency_ms
        assert restored.pc5_pdr == original.pc5_pdr
        assert restored.fiveg_latency_ms == original.fiveg_latency_ms
        assert restored.fiveg_pdr == original.fiveg_pdr
        assert restored.fiveg_sinr == original.fiveg_sinr
        assert restored.fiveg_rsrp == original.fiveg_rsrp

    def test_network_state_from_dict_missing_optional_fields(self):
        """from_dict should handle missing optional fields gracefully."""
        d = {
            "timestamp_ms": 1699999999000,
            "latitude": 43.560,
            "longitude": 1.467,
        }
        state = NetworkState.from_dict(d)
        assert state.timestamp_ms == 1699999999000
        assert state.latitude == 43.560
        assert state.dsrc_latency_ms is None
        assert state.fiveg_pdr is None

    def test_network_state_from_dict_missing_timestamp(self):
        """from_dict should default timestamp to 0 if missing."""
        d = {"latitude": 43.560, "longitude": 1.467}
        state = NetworkState.from_dict(d)
        assert state.timestamp_ms == 0


class TestQueueContext:
    """Tests for QueueContext dataclass."""

    @pytest.fixture
    def sample_queue_context(self):
        """Create a sample QueueContext."""
        return QueueContext(
            queue_depth=15,
            avg_packet_size_bytes=800,
            urgency_level=0.7,
            recent_pdr_trend=-0.2,
            target_latency_ms=50.0,
            target_pdr=0.99,
        )

    def test_queue_context_creation(self, sample_queue_context):
        """QueueContext should be created with correct values."""
        ctx = sample_queue_context
        assert ctx.queue_depth == 15
        assert ctx.avg_packet_size_bytes == 800
        assert ctx.urgency_level == 0.7
        assert ctx.recent_pdr_trend == -0.2
        assert ctx.target_latency_ms == 50.0
        assert ctx.target_pdr == 0.99

    def test_queue_context_to_dict(self, sample_queue_context):
        """to_dict should return all fields."""
        d = sample_queue_context.to_dict()
        assert d["queue_depth"] == 15
        assert d["avg_packet_size_bytes"] == 800
        assert d["urgency_level"] == 0.7
        assert d["recent_pdr_trend"] == -0.2
        assert d["target_latency_ms"] == 50.0
        assert d["target_pdr"] == 0.99

    def test_queue_context_roundtrip(self, sample_queue_context):
        """to_dict -> from_dict should preserve all values."""
        original = sample_queue_context
        restored = QueueContext.from_dict(original.to_dict())

        assert restored.queue_depth == original.queue_depth
        assert restored.avg_packet_size_bytes == original.avg_packet_size_bytes
        assert restored.urgency_level == original.urgency_level
        assert restored.recent_pdr_trend == original.recent_pdr_trend
        assert restored.target_latency_ms == original.target_latency_ms
        assert restored.target_pdr == original.target_pdr

    def test_queue_context_boundary_values(self):
        """QueueContext should handle boundary values correctly."""
        # Minimum urgency
        ctx_min = QueueContext(
            queue_depth=0,
            avg_packet_size_bytes=100,
            urgency_level=0.0,
            recent_pdr_trend=-1.0,
            target_latency_ms=1.0,
            target_pdr=0.0,
        )
        assert ctx_min.urgency_level == 0.0
        assert ctx_min.recent_pdr_trend == -1.0

        # Maximum urgency
        ctx_max = QueueContext(
            queue_depth=1000,
            avg_packet_size_bytes=1500,
            urgency_level=1.0,
            recent_pdr_trend=1.0,
            target_latency_ms=100.0,
            target_pdr=1.0,
        )
        assert ctx_max.urgency_level == 1.0
        assert ctx_max.recent_pdr_trend == 1.0


class TestRATDecision:
    """Tests for RATDecision dataclass."""

    @pytest.fixture
    def sample_rat_decision(self):
        """Create a sample RATDecision with predictions for all RATs."""
        return RATDecision(
            selected_rat=RATType.PC5,
            confidence=0.85,
            predicted_latency_ms=12.0,
            predicted_pdr=0.992,
            all_predictions={
                RATType.DSRC: (15.0, 0.985),
                RATType.PC5: (12.0, 0.992),
                RATType.FiveG: (18.0, 0.998),
            },
            recommended_max_packet_size=1200,
            model_type="lstm",
        )

    def test_rat_decision_creation(self, sample_rat_decision):
        """RATDecision should be created with correct values."""
        decision = sample_rat_decision
        assert decision.selected_rat == RATType.PC5
        assert decision.confidence == 0.85
        assert decision.predicted_latency_ms == 12.0
        assert decision.predicted_pdr == 0.992
        assert decision.recommended_max_packet_size == 1200
        assert decision.model_type == "lstm"

    def test_rat_decision_all_predictions(self, sample_rat_decision):
        """all_predictions should contain correct values for all RATs."""
        preds = sample_rat_decision.all_predictions
        assert RATType.DSRC in preds
        assert RATType.PC5 in preds
        assert RATType.FiveG in preds

        dsrc_lat, dsrc_pdr = preds[RATType.DSRC]
        assert dsrc_lat == 15.0
        assert dsrc_pdr == 0.985

    def test_rat_decision_to_dict(self, sample_rat_decision):
        """to_dict should return all fields including per-RAT predictions."""
        d = sample_rat_decision.to_dict()
        assert d["selected_rat"] == "pc5"
        assert d["confidence"] == 0.85
        assert d["pred_latency"] == 12.0
        assert d["pred_pdr"] == 0.992
        assert d["model_type"] == "lstm"
        assert d["recommended_max_packet_size"] == 1200

        # Per-RAT predictions
        assert d["pred_latency_dsrc"] == 15.0
        assert d["pred_pdr_dsrc"] == 0.985
        assert d["pred_latency_pc5"] == 12.0
        assert d["pred_pdr_pc5"] == 0.992
        assert d["pred_latency_5g"] == 18.0
        assert d["pred_pdr_5g"] == 0.998

    def test_rat_decision_from_dict(self, sample_rat_decision):
        """from_dict should recreate RATDecision correctly."""
        original = sample_rat_decision
        d = original.to_dict()
        restored = RATDecision.from_dict(d)

        assert restored.selected_rat == original.selected_rat
        assert restored.confidence == original.confidence
        assert restored.predicted_latency_ms == original.predicted_latency_ms
        assert restored.predicted_pdr == original.predicted_pdr
        assert restored.model_type == original.model_type

    def test_rat_decision_roundtrip(self, sample_rat_decision):
        """to_dict -> from_dict should preserve all values."""
        original = sample_rat_decision
        restored = RATDecision.from_dict(original.to_dict())

        assert restored.selected_rat == original.selected_rat
        assert restored.confidence == original.confidence
        assert restored.predicted_latency_ms == original.predicted_latency_ms
        assert restored.predicted_pdr == original.predicted_pdr

        # Check all_predictions
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            assert rat in restored.all_predictions
            orig_lat, orig_pdr = original.all_predictions[rat]
            rest_lat, rest_pdr = restored.all_predictions[rat]
            assert rest_lat == orig_lat
            assert rest_pdr == orig_pdr

    def test_rat_decision_without_recommended_size(self):
        """RATDecision should work without recommended_max_packet_size."""
        decision = RATDecision(
            selected_rat=RATType.FiveG,
            confidence=0.9,
            predicted_latency_ms=15.0,
            predicted_pdr=0.998,
            all_predictions={RATType.FiveG: (15.0, 0.998)},
        )
        d = decision.to_dict()
        assert "recommended_max_packet_size" not in d

        # Round-trip
        restored = RATDecision.from_dict(d)
        assert restored.recommended_max_packet_size is None

    def test_rat_decision_unavailable_rat(self):
        """RATDecision should handle UNAVAILABLE selected RAT."""
        decision = RATDecision(
            selected_rat=RATType.UNAVAILABLE,
            confidence=0.0,
            predicted_latency_ms=float("inf"),
            predicted_pdr=0.0,
            all_predictions={},
        )
        d = decision.to_dict()
        assert d["selected_rat"] == "NaN"

        restored = RATDecision.from_dict(d)
        assert restored.selected_rat == RATType.UNAVAILABLE


class TestPacketSizeDecision:
    """Tests for PacketSizeDecision dataclass."""

    @pytest.fixture
    def sample_packet_decision(self):
        """Create a sample PacketSizeDecision."""
        return PacketSizeDecision(
            packet_size_bytes=1200,
            fragment_count=1,
            priority_level=4,
            send_rate_hz=50.0,
        )

    def test_packet_decision_creation(self, sample_packet_decision):
        """PacketSizeDecision should be created with correct values."""
        pd = sample_packet_decision
        assert pd.packet_size_bytes == 1200
        assert pd.fragment_count == 1
        assert pd.priority_level == 4
        assert pd.send_rate_hz == 50.0

    def test_packet_decision_to_dict(self, sample_packet_decision):
        """to_dict should return all fields."""
        d = sample_packet_decision.to_dict()
        assert d["packet_size_bytes"] == 1200
        assert d["fragment_count"] == 1
        assert d["priority_level"] == 4
        assert d["send_rate_hz"] == 50.0

    def test_packet_decision_roundtrip(self, sample_packet_decision):
        """to_dict -> from_dict should preserve all values."""
        original = sample_packet_decision
        restored = PacketSizeDecision.from_dict(original.to_dict())

        assert restored.packet_size_bytes == original.packet_size_bytes
        assert restored.fragment_count == original.fragment_count
        assert restored.priority_level == original.priority_level
        assert restored.send_rate_hz == original.send_rate_hz

    def test_packet_decision_default_values(self):
        """from_dict should use default values for missing optional fields."""
        d = {"packet_size_bytes": 800}
        pd = PacketSizeDecision.from_dict(d)
        assert pd.packet_size_bytes == 800
        assert pd.fragment_count == 1  # default
        assert pd.priority_level == 0  # default
        assert pd.send_rate_hz == 50.0  # default

    def test_packet_decision_fragmentation(self):
        """PacketSizeDecision should support fragmented packets."""
        pd = PacketSizeDecision(
            packet_size_bytes=2000,
            fragment_count=3,
            priority_level=7,
            send_rate_hz=100.0,
        )
        assert pd.fragment_count == 3
        d = pd.to_dict()
        assert d["fragment_count"] == 3


class TestTransmissionOutcome:
    """Tests for TransmissionOutcome dataclass."""

    @pytest.fixture
    def sample_network_state(self):
        """Create a sample NetworkState for embedding in outcome."""
        return NetworkState(
            timestamp_ms=1699999999000,
            latitude=43.560123,
            longitude=1.467456,
            pc5_latency_ms=10.0,
            pc5_pdr=0.99,
        )

    @pytest.fixture
    def sample_transmission_outcome(self, sample_network_state):
        """Create a sample TransmissionOutcome."""
        return TransmissionOutcome(
            timestamp_ms=1699999999100,
            rat_used=RATType.PC5,
            packet_size_bytes=800,
            actual_latency_ms=11.5,
            delivered=True,
            network_state=sample_network_state,
        )

    def test_outcome_creation(self, sample_transmission_outcome):
        """TransmissionOutcome should be created with correct values."""
        outcome = sample_transmission_outcome
        assert outcome.timestamp_ms == 1699999999100
        assert outcome.rat_used == RATType.PC5
        assert outcome.packet_size_bytes == 800
        assert outcome.actual_latency_ms == 11.5
        assert outcome.delivered is True
        assert outcome.network_state is not None

    def test_outcome_to_dict_flattens_network_state(self, sample_transmission_outcome):
        """to_dict should flatten network state with ns_ prefix."""
        d = sample_transmission_outcome.to_dict()

        # Direct fields
        assert d["timestamp_ms"] == 1699999999100
        assert d["rat_used"] == "pc5"
        assert d["packet_size_bytes"] == 800
        assert d["actual_latency_ms"] == 11.5
        assert d["delivered"] is True

        # Flattened network state
        assert d["ns_timestamp_ms"] == 1699999999000
        assert d["ns_latitude"] == 43.560123
        assert d["ns_longitude"] == 1.467456
        assert d["ns_pc5_latency_ms"] == 10.0
        assert d["ns_pc5_pdr"] == 0.99

    def test_outcome_from_dict_extracts_network_state(self, sample_transmission_outcome):
        """from_dict should extract network state from ns_ prefixed keys."""
        d = sample_transmission_outcome.to_dict()
        restored = TransmissionOutcome.from_dict(d)

        assert restored.timestamp_ms == sample_transmission_outcome.timestamp_ms
        assert restored.rat_used == sample_transmission_outcome.rat_used
        assert restored.network_state.latitude == 43.560123
        assert restored.network_state.pc5_pdr == 0.99

    def test_outcome_roundtrip(self, sample_transmission_outcome):
        """to_dict -> from_dict should preserve all values."""
        original = sample_transmission_outcome
        restored = TransmissionOutcome.from_dict(original.to_dict())

        assert restored.timestamp_ms == original.timestamp_ms
        assert restored.rat_used == original.rat_used
        assert restored.packet_size_bytes == original.packet_size_bytes
        assert restored.actual_latency_ms == original.actual_latency_ms
        assert restored.delivered == original.delivered

        # Network state
        assert restored.network_state.timestamp_ms == original.network_state.timestamp_ms
        assert restored.network_state.latitude == original.network_state.latitude
        assert restored.network_state.longitude == original.network_state.longitude

    def test_outcome_failed_delivery(self, sample_network_state):
        """TransmissionOutcome should handle failed delivery."""
        outcome = TransmissionOutcome(
            timestamp_ms=1699999999200,
            rat_used=RATType.DSRC,
            packet_size_bytes=1500,
            actual_latency_ms=500.0,  # High latency due to timeout
            delivered=False,
            network_state=sample_network_state,
        )
        assert outcome.delivered is False

        d = outcome.to_dict()
        assert d["delivered"] is False

        restored = TransmissionOutcome.from_dict(d)
        assert restored.delivered is False
