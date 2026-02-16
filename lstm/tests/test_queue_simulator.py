"""Tests for queuesim/queue_simulator.py - SimPy-based Queue Simulator."""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if simpy is available
try:
    import simpy
    SIMPY_AVAILABLE = True
except ImportError:
    SIMPY_AVAILABLE = False

from api_types import RATType, NetworkState, RATDecision
from config import RAT_PHY_CONFIG, DTMC_PACKET_SIZES

# Skip all tests if simpy not available
pytestmark = pytest.mark.skipif(
    not SIMPY_AVAILABLE,
    reason="SimPy not installed. Install with: sudo apt install python3-simpy3"
)


# Only import queue_simulator modules if simpy is available
if SIMPY_AVAILABLE:
    from queuesim.queue_simulator import (
        QueueSimulator,
        IntegratedQueueSimulator,
    )
    from queuesim.dtmc_sizer import DTMCPacketSizer, correct_pdr_for_packet_size
    from queuesim.sim_types import SimulationMetrics, TransmissionRecord
    from queuesim.phy_layer import (
        get_phy_config,
        can_transmit,
        get_queue_capacity_packets,
        get_max_packet_size,
        calculate_tx_time_ms,
        get_base_latency_ms,
        calculate_subframe_capacity_bits,
        calculate_tx_capacity_bytes,
        calculate_queue_capacity_bytes,
    )
    from config import get_tx_interval_ms


class TestPDRCorrection:
    """Tests for packet size PDR correction function."""

    def test_same_size_no_correction(self):
        """Same packet size should return same PDR."""
        pdr = correct_pdr_for_packet_size(0.95, 1000, 1000, 0.8)
        assert pdr == 0.95

    def test_larger_packet_lower_pdr(self):
        """Larger packets should have lower PDR."""
        base_pdr = 0.95
        pdr_large = correct_pdr_for_packet_size(base_pdr, 1000, 1500, 0.8)
        assert pdr_large < base_pdr

    def test_smaller_packet_higher_pdr(self):
        """Smaller packets should have higher or equal PDR."""
        base_pdr = 0.90
        pdr_small = correct_pdr_for_packet_size(base_pdr, 1000, 500, 0.8)
        assert pdr_small >= base_pdr

    def test_pdr_bounded_zero_one(self):
        """Corrected PDR should always be in [0, 1]."""
        # Test with extreme values
        pdr = correct_pdr_for_packet_size(0.99, 100, 10000, 1.5)
        assert 0.0 <= pdr <= 1.0

        pdr = correct_pdr_for_packet_size(0.5, 2000, 100, 0.5)
        assert 0.0 <= pdr <= 1.0

    def test_zero_pdr_stays_zero(self):
        """Zero PDR should stay zero regardless of size."""
        pdr = correct_pdr_for_packet_size(0.0, 1000, 500, 0.8)
        assert pdr == 0.0

    def test_invalid_sizes_return_base_pdr(self):
        """Invalid sizes should return base PDR."""
        pdr = correct_pdr_for_packet_size(0.95, 0, 1000, 0.8)
        assert pdr == 0.95

        pdr = correct_pdr_for_packet_size(0.95, 1000, 0, 0.8)
        assert pdr == 0.95


class TestDTMCPacketSizer:
    """Tests for DTMC packet sizing logic."""

    def test_initialization(self):
        """DTMC should initialize with correct parameters."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        assert dtmc.current_state == 0  # Start at minimum
        assert len(dtmc.size_levels) == 4  # 1024, 2048, 3072, 4096
        assert dtmc.min_size == 1024
        assert dtmc.max_size == 4096

    def test_packet_sizes_are_1024_increments(self):
        """Packet sizes should be 1024, 2048, 3072, 4096."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        assert list(dtmc.size_levels) == DTMC_PACKET_SIZES
        assert list(dtmc.size_levels) == [1024, 2048, 3072, 4096]

    def test_all_rats_same_sizes(self):
        """All RATs should use the same packet size levels."""
        dtmc_5g = DTMCPacketSizer(RATType.FiveG)
        dtmc_dsrc = DTMCPacketSizer(RATType.DSRC)
        dtmc_pc5 = DTMCPacketSizer(RATType.PC5)

        assert list(dtmc_5g.size_levels) == [1024, 2048, 3072, 4096]
        assert list(dtmc_dsrc.size_levels) == [1024, 2048, 3072, 4096]
        assert list(dtmc_pc5.size_levels) == [1024, 2048, 3072, 4096]

    def test_high_pdr_increases(self):
        """PDR > 0.99 should increase packet size."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        assert dtmc.current_state == 0
        assert dtmc.current_size == 1024

        dtmc.transition(0.995, step=1)  # PDR > 0.99
        assert dtmc.current_state == 1
        assert dtmc.current_size == 2048

    def test_low_pdr_decreases(self):
        """PDR < 0.95 should decrease packet size."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.current_state = 2  # Start at 3072

        dtmc.transition(0.94, step=1)  # PDR < 0.95
        assert dtmc.current_state == 1
        assert dtmc.current_size == 2048

    def test_mid_pdr_stays(self):
        """PDR between 0.95 and 0.99 should maintain size."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.current_state = 1  # 2048 bytes

        dtmc.transition(0.97, step=1)  # 0.95 <= PDR <= 0.99
        assert dtmc.current_state == 1
        assert dtmc.current_size == 2048

    def test_cannot_go_below_minimum(self):
        """State should not go below 0."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.current_state = 0  # Already at minimum

        dtmc.transition(0.50, step=1)  # Very low PDR
        assert dtmc.current_state == 0
        assert dtmc.current_size == 1024

    def test_cannot_go_above_maximum(self):
        """State should not exceed max."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.current_state = 3  # Maximum (4096)

        dtmc.transition(0.999, step=1)  # Very high PDR
        assert dtmc.current_state == 3
        assert dtmc.current_size == 4096

    def test_history_tracking(self):
        """Transitions should be recorded in history."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.transition(0.999, step=1)  # increase
        dtmc.transition(0.94, step=2)   # decrease
        dtmc.transition(0.97, step=3)   # stay

        assert len(dtmc.history) == 3
        assert dtmc.history[0][0] == 1  # step
        assert dtmc.history[2][0] == 3

    def test_reset_clears_state(self):
        """Reset should return to initial state (minimum)."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.current_state = 3
        dtmc.transition(0.97, step=1)

        dtmc.reset()
        assert dtmc.current_state == 0  # Back to minimum
        assert dtmc.current_size == 1024
        assert len(dtmc.history) == 0

    def test_statistics(self):
        """Statistics should compute correctly."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.transition(0.999, step=1)  # increase
        dtmc.transition(0.94, step=2)   # decrease
        dtmc.transition(0.97, step=3)   # stay

        stats = dtmc.get_statistics()
        assert "mean_state" in stats
        assert stats["increase_count"] == 1
        assert stats["decrease_count"] == 1
        assert stats["stay_count"] == 1

    def test_moving_window_pdr(self):
        """Should calculate PDR from moving window."""
        dtmc = DTMCPacketSizer(RATType.FiveG, window_seconds=1.0, tx_rate_hz=10)

        # Record some outcomes
        for i in range(10):
            dtmc.record_outcome(i * 0.1, True)  # 10 successes

        pdr = dtmc.get_window_pdr(1.0)
        assert pdr == 1.0  # All successful

        # Add some failures
        for i in range(5):
            dtmc.record_outcome(1.0 + i * 0.1, False)

        pdr = dtmc.get_window_pdr(1.5)
        assert pdr is not None
        assert pdr < 1.0  # Some failures now

    def test_window_pdr_insufficient_data(self):
        """Should return None with insufficient data."""
        dtmc = DTMCPacketSizer(RATType.FiveG)

        # No data
        assert dtmc.get_window_pdr(0.0) is None

        # Too few samples
        dtmc.record_outcome(0.1, True)
        dtmc.record_outcome(0.2, True)
        assert dtmc.get_window_pdr(0.3) is None  # < 5 samples


class TestQueueSimulator:
    """Tests for QueueSimulator class."""

    def test_initialization(self):
        """QueueSimulator should initialize correctly."""
        sim = QueueSimulator()
        assert sim.base_packet_size == 1000
        assert len(sim.dtmc_sizers) == 3
        assert sim.queue_depth == 0

    def test_get_queue_context(self):
        """Should return valid QueueContext."""
        sim = QueueSimulator()
        sim._queue_depth[RATType.FiveG] = 5

        context = sim.get_queue_context()
        assert context.queue_depth == 5
        assert 0.0 <= context.urgency_level <= 1.0
        assert -1.0 <= context.recent_pdr_trend <= 1.0
        assert RATType.FiveG in context.per_rat_queue_depth

    def test_decide_packet_size(self):
        """Should return PacketSizeDecision."""
        sim = QueueSimulator()
        decision = RATDecision(
            selected_rat=RATType.FiveG,
            confidence=0.9,
            predicted_latency_ms=15.0,
            predicted_pdr=0.95,
            all_predictions={},
        )
        state = NetworkState(
            timestamp_ms=1000,
            latitude=43.56,
            longitude=1.47,
        )

        packet_decision = sim.decide_packet_size(decision, state)
        assert packet_decision.packet_size_bytes > 0
        assert packet_decision.fragment_count == 1

    def test_unavailable_rat_returns_base_size(self):
        """UNAVAILABLE RAT should return base packet size."""
        sim = QueueSimulator()
        decision = RATDecision(
            selected_rat=RATType.UNAVAILABLE,
            confidence=0.0,
            predicted_latency_ms=float("inf"),
            predicted_pdr=0.0,
            all_predictions={},
        )
        state = NetworkState(timestamp_ms=1000, latitude=43.56, longitude=1.47)

        packet_decision = sim.decide_packet_size(decision, state)
        assert packet_decision.packet_size_bytes == sim.base_packet_size

    def test_pdr_trend_calculation(self):
        """PDR trend should update based on outcomes."""
        sim = QueueSimulator()

        # Add some successful outcomes
        for _ in range(10):
            sim.recent_pdrs.append(1.0)

        context = sim.get_queue_context()
        assert context.recent_pdr_trend >= 0

    def test_reset(self):
        """Reset should clear all state."""
        sim = QueueSimulator()
        sim._queue_depth[RATType.FiveG] = 10
        sim.recent_pdrs = [0.9, 0.8, 0.7]

        sim.reset()
        assert sim.queue_depth == 0
        assert len(sim.recent_pdrs) == 0


class TestSimulationMetrics:
    """Tests for SimulationMetrics class."""

    def test_pdr_calculation(self):
        """PDR should be calculated correctly."""
        metrics = SimulationMetrics(
            total_packets=100,
            successful_packets=95,
            failed_packets=5,
        )
        assert metrics.pdr == 0.95

    def test_pdr_zero_packets(self):
        """PDR should be 0 with no packets."""
        metrics = SimulationMetrics()
        assert metrics.pdr == 0.0

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        metrics = SimulationMetrics(
            total_packets=100,
            successful_packets=95,
        )
        d = metrics.to_dict()
        assert "total_packets" in d
        assert "pdr" in d
        assert d["total_packets"] == 100


class TestIntegratedQueueSimulator:
    """Tests for IntegratedQueueSimulator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample network data for testing."""
        return pd.DataFrame({
            "latitude": [43.56] * 10,
            "longitude": [1.47] * 10,
            "selected_rat": ["5g"] * 10,
            "pred_pdr": [0.95] * 10,
            "pred_latency": [15.0] * 10,
            "confidence": [0.9] * 10,
        })

    def test_initialization(self, sample_data):
        """Should initialize correctly."""
        sim = IntegratedQueueSimulator(network_data=sample_data)
        # Default arrival rate is 1000/DEFAULT_TX_INTERVAL_MS Hz
        from config import DEFAULT_TX_INTERVAL_MS
        assert sim.arrival_rate_hz == 1000 / DEFAULT_TX_INTERVAL_MS
        assert sim.base_packet_size == 1000

    def test_run_returns_metrics(self, sample_data):
        """Run should return SimulationMetrics."""
        sim = IntegratedQueueSimulator(
            network_data=sample_data,
            seed=42,
        )
        metrics = sim.run(max_packets=10)

        assert isinstance(metrics, SimulationMetrics)
        assert metrics.total_packets > 0

    def test_results_dataframe(self, sample_data):
        """Should produce results DataFrame."""
        sim = IntegratedQueueSimulator(
            network_data=sample_data,
            seed=42,
        )
        sim.run(max_packets=10)

        df = sim.get_results_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "rat" in df.columns
        assert "packet_size" in df.columns

    def test_deterministic_with_seed(self, sample_data):
        """Same seed should produce same results."""
        sim1 = IntegratedQueueSimulator(network_data=sample_data.copy(), seed=42)
        metrics1 = sim1.run(max_packets=10)

        sim2 = IntegratedQueueSimulator(network_data=sample_data.copy(), seed=42)
        metrics2 = sim2.run(max_packets=10)

        assert metrics1.total_packets == metrics2.total_packets
        assert metrics1.successful_packets == metrics2.successful_packets

    def test_rat_usage_tracking(self, sample_data):
        """Should track RAT usage."""
        sim = IntegratedQueueSimulator(network_data=sample_data, seed=42)
        metrics = sim.run(max_packets=10)

        assert len(metrics.rat_usage) > 0
        total_usage = sum(metrics.rat_usage.values())
        assert total_usage == metrics.total_packets


class TestTransmissionRecord:
    """Tests for TransmissionRecord dataclass."""

    def test_creation(self):
        """Should create TransmissionRecord correctly."""
        record = TransmissionRecord(
            timestamp=1.0,
            rat=RATType.FiveG,
            packet_size=1000,
            predicted_pdr=0.95,
            corrected_pdr=0.93,
            success=True,
            latency_ms=15.0,
            queue_depth=5,
            dtmc_state=4,
        )
        assert record.timestamp == 1.0
        assert record.rat == RATType.FiveG
        assert record.success is True


# =============================================================================
# PHY-Layer Tests
# =============================================================================

class TestPHYConfig:
    """Tests for PHY-layer configuration."""

    def test_config_exists_for_all_rats(self):
        """PHY config should exist for all RAT types."""
        for rat_key in ["5g", "pc5", "dsrc"]:
            assert rat_key in RAT_PHY_CONFIG
            config = RAT_PHY_CONFIG[rat_key]
            # All RATs should have base latency and queue multiplier
            assert "base_latency_ms" in config
            assert "queue_multiplier" in config

    def test_5g_has_rb_formula_fields(self):
        """5G config should have RB formula fields."""
        config = RAT_PHY_CONFIG["5g"]
        assert "n_subcarriers" in config
        assert "n_symbols" in config
        assert "n_rbs" in config
        assert "modulation_order" in config
        assert "coding_rate" in config
        assert "slot_duration_ms" in config

    def test_dsrc_has_ofdm_and_contention_fields(self):
        """DSRC config should have OFDM capacity and contention window fields."""
        config = RAT_PHY_CONFIG["dsrc"]
        assert "n_data_subcarriers" in config
        assert "n_symbols" in config
        assert "modulation_order" in config
        assert "coding_rate" in config
        assert "contention_window_min" in config
        assert "contention_window_max" in config

    def test_pc5_has_subchannel_fields(self):
        """PC5 config should have subchannel-based RB fields."""
        config = RAT_PHY_CONFIG["pc5"]
        assert "n_subcarriers" in config
        assert "n_symbols" in config
        assert "n_rbs_per_subchannel" in config
        assert "n_subchannels" in config
        assert "modulation_order" in config
        assert "coding_rate" in config
        assert "subframe_duration_ms" in config

    def test_capacity_ordering(self):
        """5G should have highest capacity, PC5 lowest (via RB formula)."""
        tx_5g = calculate_tx_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        tx_dsrc = calculate_tx_capacity_bytes(RATType.DSRC, get_tx_interval_ms(RATType.DSRC))
        tx_pc5 = calculate_tx_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))

        assert tx_5g > tx_dsrc
        assert tx_dsrc > tx_pc5


class TestPHYHelperFunctions:
    """Tests for PHY-layer helper functions."""

    def test_get_phy_config_valid_rat(self):
        """Should return config for valid RAT."""
        config = get_phy_config(RATType.FiveG)
        assert config == RAT_PHY_CONFIG["5g"]

        config = get_phy_config(RATType.DSRC)
        assert config == RAT_PHY_CONFIG["dsrc"]

        config = get_phy_config(RATType.PC5)
        assert config == RAT_PHY_CONFIG["pc5"]

    def test_get_phy_config_unavailable_rat(self):
        """Should return default config for UNAVAILABLE RAT."""
        config = get_phy_config(RATType.UNAVAILABLE)
        assert config == RAT_PHY_CONFIG["5g"]  # Default to 5G

    def test_can_transmit_within_limit(self):
        """Should return True for packets within TX limit."""
        assert can_transmit(1000, RATType.FiveG)
        assert can_transmit(1000, RATType.DSRC)
        assert can_transmit(1000, RATType.PC5)

    def test_can_transmit_at_limit(self):
        """Should return True for packets at exact TX limit."""
        max_5g = calculate_tx_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        assert can_transmit(max_5g, RATType.FiveG)

    def test_can_transmit_over_limit(self):
        """Should return False for packets over TX limit."""
        max_5g = calculate_tx_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        assert not can_transmit(max_5g + 1, RATType.FiveG)

        max_pc5 = calculate_tx_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))
        assert not can_transmit(max_pc5 + 1, RATType.PC5)

    def test_get_queue_capacity_packets(self):
        """Should calculate queue capacity in packets."""
        # Queue capacity is based on RB formula
        queue_bytes = calculate_queue_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        capacity = get_queue_capacity_packets(RATType.FiveG, 1000)
        expected = queue_bytes // 1000
        assert capacity == expected

    def test_get_queue_capacity_small_packets(self):
        """Smaller packets should give higher packet capacity."""
        capacity_small = get_queue_capacity_packets(RATType.FiveG, 100)
        capacity_large = get_queue_capacity_packets(RATType.FiveG, 1000)
        assert capacity_small > capacity_large

    def test_get_queue_capacity_zero_packet(self):
        """Zero packet size should return default capacity."""
        capacity = get_queue_capacity_packets(RATType.FiveG, 0)
        assert capacity == 100  # Default

    def test_get_max_packet_size(self):
        """Should return max packet size (TX capacity) for each RAT."""
        assert get_max_packet_size(RATType.FiveG) == calculate_tx_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        assert get_max_packet_size(RATType.PC5) == calculate_tx_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))
        assert get_max_packet_size(RATType.DSRC) == calculate_tx_capacity_bytes(RATType.DSRC, get_tx_interval_ms(RATType.DSRC))

    def test_calculate_tx_time_positive(self):
        """TX time should always be positive."""
        for rat in [RATType.FiveG, RATType.PC5, RATType.DSRC]:
            tx_time = calculate_tx_time_ms(1000, rat)
            assert tx_time > 0

    def test_calculate_tx_time_proportional_to_size(self):
        """TX time should generally increase with packet size."""
        small_time = calculate_tx_time_ms(100, RATType.FiveG)
        large_time = calculate_tx_time_ms(10000, RATType.FiveG)
        assert large_time >= small_time

    def test_get_base_latency_ms(self):
        """Should return base latency for each RAT."""
        assert get_base_latency_ms(RATType.FiveG) == RAT_PHY_CONFIG["5g"]["base_latency_ms"]
        assert get_base_latency_ms(RATType.DSRC) == RAT_PHY_CONFIG["dsrc"]["base_latency_ms"]
        assert get_base_latency_ms(RATType.PC5) == RAT_PHY_CONFIG["pc5"]["base_latency_ms"]

    def test_dsrc_lowest_latency(self):
        """DSRC should have lowest base latency (contention-based)."""
        assert get_base_latency_ms(RATType.DSRC) < get_base_latency_ms(RATType.PC5)
        assert get_base_latency_ms(RATType.PC5) < get_base_latency_ms(RATType.FiveG)


class TestQueueSimulatorPHY:
    """Tests for QueueSimulator PHY-layer features."""

    def test_enforce_phy_limits_default(self):
        """PHY limits should be enforced by default."""
        sim = QueueSimulator()
        assert sim.enforce_phy_limits is True

    def test_enforce_phy_limits_disabled(self):
        """PHY limits can be disabled."""
        sim = QueueSimulator(enforce_phy_limits=False)
        assert sim.enforce_phy_limits is False

    def test_queue_bytes_tracking(self):
        """Queue bytes should be tracked."""
        sim = QueueSimulator()
        assert sim.queue_bytes == 0

        # Enqueue a packet
        sim.enqueue(1000, RATType.FiveG)
        assert sim.queue_bytes == 1000
        assert sim.queue_depth == 1

        # Dequeue
        sim.dequeue(1000, RATType.FiveG)
        assert sim.queue_bytes == 0
        assert sim.queue_depth == 0

    def test_can_enqueue_within_capacity(self):
        """Should allow enqueue within capacity."""
        sim = QueueSimulator()
        assert sim.can_enqueue(1000, RATType.FiveG)

    def test_can_enqueue_at_capacity(self):
        """Should check capacity correctly using RB-based formula."""
        sim = QueueSimulator()
        capacity = calculate_queue_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))

        # Fill queue to near capacity
        sim._queue_bytes[RATType.FiveG] = capacity - 1000
        assert sim.can_enqueue(1000, RATType.FiveG)  # Should fit

        sim._queue_bytes[RATType.FiveG] = capacity
        assert not sim.can_enqueue(1, RATType.FiveG)  # Should not fit

    def test_enqueue_drops_on_overflow(self):
        """Enqueue should return False and track dropped packets on overflow."""
        sim = QueueSimulator()
        capacity = calculate_queue_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))

        # Fill queue to capacity
        sim._queue_bytes[RATType.FiveG] = capacity

        # Try to enqueue more
        result = sim.enqueue(1000, RATType.FiveG)
        assert result is False
        assert sim.metrics.dropped_packets == 1
        assert sim.metrics.per_rat_dropped_packets[RATType.FiveG] == 1

    def test_enqueue_without_phy_limits(self):
        """Enqueue should always succeed without PHY limits."""
        sim = QueueSimulator(enforce_phy_limits=False)

        # Simulate very full queue
        sim._queue_bytes[RATType.FiveG] = 1_000_000_000  # 1GB

        # Should still allow enqueue
        assert sim.can_enqueue(1000, RATType.FiveG)

    def test_max_queue_depth_tracking(self):
        """Max queue depth should be tracked."""
        sim = QueueSimulator()

        sim.enqueue(1000, RATType.FiveG)
        sim.enqueue(1000, RATType.FiveG)
        assert sim.metrics.max_queue_depth == 2

        sim.dequeue(1000, RATType.FiveG)
        assert sim.metrics.max_queue_depth == 2  # Max stays

    def test_get_queue_capacity(self):
        """Should return queue capacity for current RAT using RB formula."""
        sim = QueueSimulator()
        sim.current_rat = RATType.FiveG

        # Capacity is based on DTMC's current size (starts at 1024)
        dtmc = sim.dtmc_sizers[RATType.FiveG]
        assert dtmc.current_size == 1024  # Initial size

        capacity = sim.get_queue_capacity()
        queue_bytes = calculate_queue_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        expected = queue_bytes // 1024
        assert capacity == expected

    def test_packet_size_capped_at_max(self):
        """Packet size should be capped at RAT's PHY max."""
        sim = QueueSimulator()

        # Create a decision
        decision = RATDecision(
            selected_rat=RATType.PC5,
            confidence=0.9,
            predicted_latency_ms=10.0,
            predicted_pdr=0.99,
            all_predictions={},
        )

        # Force DTMC to max state (4096 bytes)
        dtmc = sim.dtmc_sizers[RATType.PC5]
        dtmc.current_state = dtmc.num_levels - 1
        assert dtmc.current_size == 4096

        # Get packet size decision
        packet_decision = sim.decide_packet_size(decision, current_time=1.0)

        # Should not exceed PC5 PHY max (calculated from RB formula)
        # Since 4096 < RB-calculated max, should return 4096
        max_phy = calculate_tx_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))
        assert packet_decision.packet_size_bytes <= max_phy
        assert packet_decision.packet_size_bytes == 4096  # DTMC max

    def test_reset_clears_queue_bytes(self):
        """Reset should clear queue bytes."""
        sim = QueueSimulator()
        sim._queue_bytes[RATType.FiveG] = 5000
        sim._queue_depth[RATType.FiveG] = 5

        sim.reset()
        assert sim.queue_bytes == 0
        assert sim.queue_depth == 0


class TestIntegratedQueueSimulatorPHY:
    """Tests for IntegratedQueueSimulator PHY-layer features."""

    @pytest.fixture
    def sample_data(self):
        """Create sample network data for testing."""
        return pd.DataFrame({
            "latitude": [43.56] * 20,
            "longitude": [1.47] * 20,
            "selected_rat": ["5g"] * 20,
            "pred_pdr": [0.95] * 20,
            "pred_latency": [15.0] * 20,
            "confidence": [0.9] * 20,
        })

    def test_enforce_phy_limits_parameter(self, sample_data):
        """Should pass PHY limits parameter to inner simulator."""
        sim = IntegratedQueueSimulator(
            network_data=sample_data,
            enforce_phy_limits=True,
        )
        assert sim.enforce_phy_limits is True
        assert sim.queue_sim.enforce_phy_limits is True

        sim = IntegratedQueueSimulator(
            network_data=sample_data,
            enforce_phy_limits=False,
        )
        assert sim.enforce_phy_limits is False
        assert sim.queue_sim.enforce_phy_limits is False

    def test_simulation_tracks_tx_time(self, sample_data):
        """Simulation should track mean TX time."""
        sim = IntegratedQueueSimulator(
            network_data=sample_data,
            seed=42,
        )
        metrics = sim.run(max_packets=10)

        # Mean TX time should be calculated
        assert metrics.mean_tx_time_ms >= 0

    def test_results_include_tx_time(self, sample_data):
        """Results DataFrame should include TX time."""
        sim = IntegratedQueueSimulator(
            network_data=sample_data,
            seed=42,
        )
        sim.run(max_packets=10)
        df = sim.get_results_dataframe()

        assert "tx_time_ms" in df.columns
        assert df["tx_time_ms"].notna().all()

    def test_dropped_packets_tracked(self, sample_data):
        """Dropped packets should be tracked in metrics."""
        sim = IntegratedQueueSimulator(
            network_data=sample_data,
            seed=42,
        )
        metrics = sim.run(max_packets=10)

        # Dropped packets should be a valid integer >= 0
        assert isinstance(metrics.dropped_packets, int)
        assert metrics.dropped_packets >= 0

    def test_max_queue_depth_tracked(self, sample_data):
        """Max queue depth should be tracked in metrics."""
        sim = IntegratedQueueSimulator(
            network_data=sample_data,
            seed=42,
        )
        metrics = sim.run(max_packets=10)

        # Max queue depth should be tracked
        assert isinstance(metrics.max_queue_depth, int)
        assert metrics.max_queue_depth >= 0


class TestSimulationMetricsPHY:
    """Tests for SimulationMetrics PHY-layer fields."""

    def test_new_phy_fields_exist(self):
        """SimulationMetrics should have new PHY fields."""
        metrics = SimulationMetrics()

        assert hasattr(metrics, "dropped_packets")
        assert hasattr(metrics, "max_queue_depth")
        assert hasattr(metrics, "mean_tx_time_ms")
        assert hasattr(metrics, "capacity_limited_count")

    def test_to_dict_includes_phy_fields(self):
        """to_dict should include PHY fields."""
        metrics = SimulationMetrics(
            dropped_packets=5,
            max_queue_depth=10,
            mean_tx_time_ms=2.5,
            capacity_limited_count=3,
        )
        d = metrics.to_dict()

        assert "dropped_packets" in d
        assert d["dropped_packets"] == 5
        assert "max_queue_depth" in d
        assert d["max_queue_depth"] == 10
        assert "mean_tx_time_ms" in d
        assert d["mean_tx_time_ms"] == 2.5
        assert "capacity_limited_count" in d
        assert d["capacity_limited_count"] == 3


# =============================================================================
# RB-Based Capacity Model Tests
# =============================================================================

class TestRBCapacityModel:
    """Tests for RB-based PHY capacity calculation."""

    def test_pc5_subframe_capacity_formula(self):
        """PC5 capacity should follow dSF = NSC × Nsym × NRB × Rmod × CR."""
        # PC5 with QPSK CR=0.5, 1 subchannel (10 RBs):
        # dSF = 12 × 14 × 10 × 2 × 0.5 = 1,680 bits/subframe
        bits = calculate_subframe_capacity_bits(RATType.PC5)

        # Formula: 12 * 14 * 10 * 2 * 0.5 = 1680
        expected = 12 * 14 * 10 * 2 * 0.5
        assert bits == int(expected)
        assert bits == 1680

    def test_5g_subframe_capacity_formula(self):
        """5G capacity should follow dSF = NSC × Nsym × NRB × Rmod × CR."""
        # 5G with 64QAM CR=0.66, 106 RBs:
        # dSF = 12 × 14 × 106 × 6 × 0.66 = 70,519 bits/slot
        bits = calculate_subframe_capacity_bits(RATType.FiveG)

        # Formula: 12 * 14 * 106 * 6 * 0.66 = 70518.72 -> 70518
        expected = 12 * 14 * 106 * 6 * 0.66
        assert bits == int(expected)

    def test_dsrc_uses_ofdm_capacity_model(self):
        """DSRC should use OFDM formula: 52 subcarriers × 125 sym/ms × QPSK × 1/2 = 6500 bits/ms."""
        bits = calculate_subframe_capacity_bits(RATType.DSRC)

        # 52 data subcarriers × 125 symbols/ms × 2 (QPSK) × 0.5 (CR) = 6500 bits/ms
        assert bits == 6500

    def test_tx_capacity_bytes_pc5(self):
        """PC5 TX capacity for 100ms interval."""
        # 1,680 bits/ms * 100ms = 168,000 bits = 21,000 bytes
        bytes_100ms = calculate_tx_capacity_bytes(RATType.PC5, 100)

        expected = (1680 * 100) // 8
        assert bytes_100ms == expected
        assert bytes_100ms == 21000

    def test_tx_capacity_bytes_5g(self):
        """5G TX capacity for 100ms interval."""
        bits_per_ms = calculate_subframe_capacity_bits(RATType.FiveG)
        bytes_100ms = calculate_tx_capacity_bytes(RATType.FiveG, 100)

        expected = (bits_per_ms * 100) // 8
        assert bytes_100ms == expected

    def test_tx_capacity_bytes_dsrc(self):
        """DSRC TX capacity for 100ms interval."""
        # 6,500 bits/ms * 100ms = 650,000 bits = 81,250 bytes
        bytes_100ms = calculate_tx_capacity_bytes(RATType.DSRC, 100)

        expected = (6500 * 100) // 8
        assert bytes_100ms == expected
        assert bytes_100ms == 81250

    def test_queue_capacity_is_multiple_of_tx(self):
        """Queue capacity should be a multiple of TX capacity."""
        for rat in [RATType.PC5, RATType.FiveG, RATType.DSRC]:
            tx_capacity = calculate_tx_capacity_bytes(rat, get_tx_interval_ms(rat))
            queue_capacity = calculate_queue_capacity_bytes(rat, get_tx_interval_ms(rat))
            config = get_phy_config(rat)
            multiplier = config.get("queue_multiplier", 2)

            assert queue_capacity == tx_capacity * multiplier

    def test_capacity_ordering_matches_expectations(self):
        """5G should have highest capacity, PC5 lowest for same interval."""
        tx_5g = calculate_tx_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        tx_dsrc = calculate_tx_capacity_bytes(RATType.DSRC, get_tx_interval_ms(RATType.DSRC))
        tx_pc5 = calculate_tx_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))

        assert tx_5g > tx_dsrc
        assert tx_dsrc > tx_pc5

    def test_get_max_packet_size_uses_rb_formula(self):
        """get_max_packet_size should use RB-based capacity."""
        for rat in [RATType.PC5, RATType.FiveG, RATType.DSRC]:
            max_size = get_max_packet_size(rat)
            expected = calculate_tx_capacity_bytes(rat, get_tx_interval_ms(rat))
            assert max_size == expected

    def test_can_transmit_uses_rb_formula(self):
        """can_transmit should use RB-based capacity."""
        # PC5: 21,000 bytes per 100ms
        max_pc5 = calculate_tx_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))

        assert can_transmit(max_pc5, RATType.PC5)
        assert not can_transmit(max_pc5 + 1, RATType.PC5)

    def test_get_queue_capacity_packets_uses_rb_formula(self):
        """get_queue_capacity_packets should use RB-based queue capacity."""
        packet_size = 1000
        queue_bytes = calculate_queue_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        expected_packets = queue_bytes // packet_size

        actual = get_queue_capacity_packets(RATType.FiveG, packet_size)
        assert actual == expected_packets

    def test_pc5_subchannel_scaling(self):
        """PC5 capacity should scale with number of subchannels."""
        # This test validates the formula handles subchannel config
        # Default is 1 subchannel with 10 RBs
        bits_1ch = calculate_subframe_capacity_bits(RATType.PC5)

        # Formula with 1 subchannel: 12 * 14 * 10 * 2 * 0.5 = 1680
        assert bits_1ch == 1680

    def test_unavailable_rat_uses_5g_defaults(self):
        """UNAVAILABLE RAT should use 5G defaults for capacity."""
        unavailable_bits = calculate_subframe_capacity_bits(RATType.UNAVAILABLE)
        fiveg_bits = calculate_subframe_capacity_bits(RATType.FiveG)

        assert unavailable_bits == fiveg_bits


class TestRBCapacityIntegration:
    """Integration tests for RB-based capacity with simulator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample network data for testing."""
        return pd.DataFrame({
            "latitude": [43.56] * 10,
            "longitude": [1.47] * 10,
            "selected_rat": ["pc5"] * 10,  # Use PC5 for lower capacity testing
            "pred_pdr": [0.95] * 10,
            "pred_latency": [8.0] * 10,
            "confidence": [0.9] * 10,
        })

    def test_queue_simulator_uses_rb_capacity(self):
        """QueueSimulator should use RB-based capacity for queue limits."""
        sim = QueueSimulator()

        # Get expected capacity from RB formula
        expected_capacity = calculate_queue_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))

        # Fill to near capacity
        sim._queue_bytes[RATType.PC5] = expected_capacity - 100

        # Should still accept small packet
        assert sim.can_enqueue(100, RATType.PC5)

        # Should reject when at capacity
        sim._queue_bytes[RATType.PC5] = expected_capacity
        assert not sim.can_enqueue(1, RATType.PC5)

    def test_simulator_respects_rb_tx_limit(self):
        """Simulator should cap packet size at RB-based TX limit."""
        sim = QueueSimulator()

        # Create decision with PC5
        decision = RATDecision(
            selected_rat=RATType.PC5,
            confidence=0.9,
            predicted_latency_ms=8.0,
            predicted_pdr=0.999,  # High PDR to trigger DTMC increase
            all_predictions={},
        )

        # Force DTMC to maximum state
        dtmc = sim.dtmc_sizers[RATType.PC5]
        dtmc.current_state = dtmc.num_levels - 1
        assert dtmc.current_size == 4096

        # Get packet decision
        packet_decision = sim.decide_packet_size(decision, current_time=1.0)

        # Should not exceed PHY max
        max_phy = calculate_tx_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))
        assert packet_decision.packet_size_bytes <= max_phy

    def test_tx_time_calculation_consistent(self):
        """TX time should be consistent with capacity formula."""
        packet_size = 1000  # bytes

        for rat in [RATType.PC5, RATType.FiveG, RATType.DSRC]:
            bits_per_ms = calculate_subframe_capacity_bits(rat)
            tx_time = calculate_tx_time_ms(packet_size, rat)

            # TX time should be approximately packet_bits / bits_per_ms
            expected_base = (packet_size * 8) / bits_per_ms
            # Allow for contention delay in DSRC
            assert tx_time >= expected_base * 0.9


# =============================================================================
# Per-RAT Queue Tests
# =============================================================================

class TestPerRATQueues:
    """Tests for per-RAT independent queue behavior."""

    def test_per_rat_queue_independence(self):
        """Enqueue to 5G and DSRC separately, verify independent counters."""
        sim = QueueSimulator()

        sim.enqueue(1000, RATType.FiveG)
        sim.enqueue(2000, RATType.DSRC)

        assert sim._queue_depth[RATType.FiveG] == 1
        assert sim._queue_depth[RATType.DSRC] == 1
        assert sim._queue_depth[RATType.PC5] == 0
        assert sim._queue_bytes[RATType.FiveG] == 1000
        assert sim._queue_bytes[RATType.DSRC] == 2000
        assert sim.queue_depth == 2  # Aggregate
        assert sim.queue_bytes == 3000

    def test_per_rat_queue_context(self):
        """get_queue_context() should return per-RAT depths."""
        sim = QueueSimulator()
        sim.enqueue(1000, RATType.FiveG)
        sim.enqueue(1000, RATType.FiveG)
        sim.enqueue(1000, RATType.PC5)

        ctx = sim.get_queue_context()
        assert ctx.queue_depth == 3
        assert ctx.per_rat_queue_depth[RATType.FiveG] == 2
        assert ctx.per_rat_queue_depth[RATType.PC5] == 1

    def test_aggregate_queue_depth_property(self):
        """Sum of per-RAT depths should equal queue_depth property."""
        sim = QueueSimulator()

        sim.enqueue(500, RATType.FiveG)
        sim.enqueue(500, RATType.PC5)
        sim.enqueue(500, RATType.DSRC)

        assert sim.queue_depth == sum(sim._queue_depth.values())

    def test_per_rat_dropped_packets(self):
        """Overflow on one RAT should not affect another."""
        sim = QueueSimulator()

        # Fill PC5 queue to capacity
        pc5_capacity = calculate_queue_capacity_bytes(
            RATType.PC5, get_tx_interval_ms(RATType.PC5),
        )
        sim._queue_bytes[RATType.PC5] = pc5_capacity

        # PC5 should reject
        assert sim.enqueue(1000, RATType.PC5) is False
        assert sim.metrics.per_rat_dropped_packets.get(RATType.PC5, 0) == 1

        # 5G should still accept
        assert sim.enqueue(1000, RATType.FiveG) is True
        assert sim.metrics.per_rat_dropped_packets.get(RATType.FiveG, 0) == 0

    def test_dropped_packet_counts_as_pdr_loss(self):
        """A dropped packet should feed update_pdr_estimate(delivered=False)."""
        from api_types import TransmissionOutcome, NetworkState

        sim = QueueSimulator()
        dtmc = sim.dtmc_sizers[RATType.PC5]

        # Record several successes to establish baseline
        for i in range(10):
            dtmc.record_outcome(i * 0.1, True)

        pdr_before = dtmc.get_window_pdr(1.0)

        # Feed a drop as a failure
        outcome = TransmissionOutcome(
            timestamp_ms=1100,
            rat_used=RATType.PC5,
            packet_size_bytes=1000,
            actual_latency_ms=0.0,
            delivered=False,
            network_state=NetworkState(timestamp_ms=1100, latitude=43.56, longitude=1.47),
        )
        sim.update_pdr_estimate(outcome)

        pdr_after = dtmc.get_window_pdr(1.1)
        assert pdr_after is not None
        assert pdr_before is not None
        assert pdr_after < pdr_before

    def test_integrated_parallel_processing(self):
        """IntegratedQueueSimulator should produce results from multiple RATs."""
        # Create data with mixed RATs
        data = pd.DataFrame({
            "latitude": [43.56] * 20,
            "longitude": [1.47] * 20,
            "selected_rat": ["5g"] * 7 + ["dsrc"] * 7 + ["pc5"] * 6,
            "pred_pdr": [0.95] * 20,
            "pred_latency": [15.0] * 20,
            "confidence": [0.9] * 20,
        })

        sim = IntegratedQueueSimulator(network_data=data, seed=42)
        metrics = sim.run(max_packets=20)

        assert isinstance(metrics, SimulationMetrics)
        assert metrics.total_packets > 0

        # Should have results from multiple RATs
        df = sim.get_results_dataframe()
        assert len(df) > 0
        # Results should be sorted by timestamp
        assert df["timestamp"].is_monotonic_increasing

    def test_per_rat_metrics_populated(self):
        """Per-RAT metrics should be populated after simulation."""
        data = pd.DataFrame({
            "latitude": [43.56] * 10,
            "longitude": [1.47] * 10,
            "selected_rat": ["5g"] * 5 + ["dsrc"] * 5,
            "pred_pdr": [0.95] * 10,
            "pred_latency": [15.0] * 10,
            "confidence": [0.9] * 10,
        })

        sim = IntegratedQueueSimulator(network_data=data, seed=42)
        metrics = sim.run(max_packets=10)

        # per_rat_max_queue_depth should have entries for RATs that were used
        # (may be empty if all packets dequeued instantly)
        assert isinstance(metrics.per_rat_max_queue_depth, dict)
        assert isinstance(metrics.per_rat_mean_queue_depth, dict)
        assert isinstance(metrics.per_rat_dropped_packets, dict)
