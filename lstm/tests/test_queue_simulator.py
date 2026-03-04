"""
Tests for queuesim/queue_simulator.py - QueueSimulator and IntegratedQueueSimulator.

PHY-layer tests are in test_phy_layer.py.
DTMC sizer tests are in test_dtmc_sizer.py.
SimulationMetrics / TransmissionRecord tests are in test_sim_types.py.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import simpy
    SIMPY_AVAILABLE = True
except ImportError:
    SIMPY_AVAILABLE = False

from api_types import RATType, NetworkState, RATDecision

pytestmark = pytest.mark.skipif(
    not SIMPY_AVAILABLE,
    reason="SimPy not installed. Install with: sudo apt install python3-simpy3"
)

if SIMPY_AVAILABLE:
    from queuesim.queue_simulator import (
        QueueSimulator,
        IntegratedQueueSimulator,
    )
    from queuesim.sim_types import SimulationMetrics
    from queuesim.phy_layer import (
        calculate_tx_capacity_bytes,
        calculate_queue_capacity_bytes,
    )
    from config import get_tx_interval_ms


class TestQueueSimulator:
    """Tests for QueueSimulator class."""

    def test_initialization(self):
        """QueueSimulator should initialize correctly."""
        sim = QueueSimulator()
        assert sim.base_packet_size == 1024
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

    def test_enable_dtmc_false_returns_base_size(self):
        """When enable_dtmc=False, decide_packet_size always returns base_packet_size."""
        sim = QueueSimulator(base_packet_size=1234, enable_dtmc=False)
        for rat in [RATType.FiveG, RATType.PC5, RATType.DSRC]:
            decision = RATDecision(
                selected_rat=rat,
                confidence=0.9,
                predicted_latency_ms=15.0,
                predicted_pdr=0.99,
                all_predictions={},
            )
            pkt = sim.decide_packet_size(decision, current_time=1.0)
            assert pkt.packet_size_bytes == 1234, (
                f"Expected base_packet_size=1234 for {rat}, got {pkt.packet_size_bytes}"
            )
            assert pkt.fragment_count == 1

    def test_pdr_trend_calculation(self):
        """PDR trend should update based on outcomes."""
        sim = QueueSimulator()
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


class TestQueueSimulatorPHY:
    """Tests for QueueSimulator PHY-layer queue management features."""

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

        sim.enqueue(1000, RATType.FiveG)
        assert sim.queue_bytes == 1000
        assert sim.queue_depth == 1

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

        sim._queue_bytes[RATType.FiveG] = capacity - 1000
        assert sim.can_enqueue(1000, RATType.FiveG)

        sim._queue_bytes[RATType.FiveG] = capacity
        assert not sim.can_enqueue(1, RATType.FiveG)

    def test_enqueue_drops_on_overflow(self):
        """Enqueue should return False and track dropped packets on overflow."""
        sim = QueueSimulator()
        capacity = calculate_queue_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))

        sim._queue_bytes[RATType.FiveG] = capacity
        result = sim.enqueue(1000, RATType.FiveG)
        assert result is False
        assert sim.metrics.dropped_packets == 1
        assert sim.metrics.per_rat_dropped_packets[RATType.FiveG] == 1

    def test_enqueue_without_phy_limits(self):
        """Enqueue should always succeed without PHY limits."""
        sim = QueueSimulator(enforce_phy_limits=False)
        sim._queue_bytes[RATType.FiveG] = 1_000_000_000
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

        dtmc = sim.dtmc_sizers[RATType.FiveG]
        assert dtmc.current_size == 1024

        capacity = sim.get_queue_capacity()
        queue_bytes = calculate_queue_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        expected = queue_bytes // 1024
        assert capacity == expected

    def test_packet_size_capped_at_max(self):
        """Packet size should be capped at RAT's PHY max."""
        sim = QueueSimulator()

        decision = RATDecision(
            selected_rat=RATType.PC5,
            confidence=0.9,
            predicted_latency_ms=10.0,
            predicted_pdr=0.99,
            all_predictions={},
        )

        dtmc = sim.dtmc_sizers[RATType.PC5]
        dtmc.current_state = dtmc.num_levels - 1
        assert dtmc.current_size == 4096

        packet_decision = sim.decide_packet_size(decision, current_time=1.0)
        max_phy = calculate_tx_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))
        assert packet_decision.packet_size_bytes <= max_phy
        assert packet_decision.packet_size_bytes == 4096

    def test_reset_clears_queue_bytes(self):
        """Reset should clear queue bytes."""
        sim = QueueSimulator()
        sim._queue_bytes[RATType.FiveG] = 5000
        sim._queue_depth[RATType.FiveG] = 5

        sim.reset()
        assert sim.queue_bytes == 0
        assert sim.queue_depth == 0


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
        from config import DEFAULT_TX_INTERVAL_MS
        assert sim.arrival_rate_hz == 1000 / DEFAULT_TX_INTERVAL_MS
        assert sim.base_packet_size == 1024

    def test_run_returns_metrics(self, sample_data):
        """Run should return SimulationMetrics."""
        sim = IntegratedQueueSimulator(network_data=sample_data, seed=42)
        metrics = sim.run(max_packets=10)

        assert isinstance(metrics, SimulationMetrics)
        assert metrics.total_packets > 0

    def test_results_dataframe(self, sample_data):
        """Should produce results DataFrame."""
        sim = IntegratedQueueSimulator(network_data=sample_data, seed=42)
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
        sim = IntegratedQueueSimulator(network_data=sample_data, enforce_phy_limits=True)
        assert sim.enforce_phy_limits is True
        assert sim.queue_sim.enforce_phy_limits is True

        sim = IntegratedQueueSimulator(network_data=sample_data, enforce_phy_limits=False)
        assert sim.enforce_phy_limits is False
        assert sim.queue_sim.enforce_phy_limits is False

    def test_simulation_tracks_tx_time(self, sample_data):
        """Simulation should track mean TX time."""
        sim = IntegratedQueueSimulator(network_data=sample_data, seed=42)
        metrics = sim.run(max_packets=10)
        assert metrics.mean_tx_time_ms >= 0

    def test_results_include_tx_time(self, sample_data):
        """Results DataFrame should include TX time."""
        sim = IntegratedQueueSimulator(network_data=sample_data, seed=42)
        sim.run(max_packets=10)
        df = sim.get_results_dataframe()

        assert "tx_time_ms" in df.columns
        assert df["tx_time_ms"].notna().all()

    def test_dropped_packets_tracked(self, sample_data):
        """Dropped packets should be tracked in metrics."""
        sim = IntegratedQueueSimulator(network_data=sample_data, seed=42)
        metrics = sim.run(max_packets=10)

        assert isinstance(metrics.dropped_packets, int)
        assert metrics.dropped_packets >= 0

    def test_max_queue_depth_tracked(self, sample_data):
        """Max queue depth should be tracked in metrics."""
        sim = IntegratedQueueSimulator(network_data=sample_data, seed=42)
        metrics = sim.run(max_packets=10)

        assert isinstance(metrics.max_queue_depth, int)
        assert metrics.max_queue_depth >= 0


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
        assert sim.queue_depth == 2
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
        pc5_capacity = calculate_queue_capacity_bytes(
            RATType.PC5, get_tx_interval_ms(RATType.PC5),
        )
        sim._queue_bytes[RATType.PC5] = pc5_capacity

        assert sim.enqueue(1000, RATType.PC5) is False
        assert sim.metrics.per_rat_dropped_packets.get(RATType.PC5, 0) == 1

        assert sim.enqueue(1000, RATType.FiveG) is True
        assert sim.metrics.per_rat_dropped_packets.get(RATType.FiveG, 0) == 0

    def test_dropped_packet_counts_as_pdr_loss(self):
        """A dropped packet should feed update_pdr_estimate(delivered=False)."""
        from api_types import TransmissionOutcome

        sim = QueueSimulator()
        dtmc = sim.dtmc_sizers[RATType.PC5]

        for i in range(10):
            dtmc.record_outcome(i * 0.1, True)

        pdr_before = dtmc.get_window_pdr(1.0)

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

        df = sim.get_results_dataframe()
        assert len(df) > 0
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

        assert isinstance(metrics.per_rat_max_queue_depth, dict)
        assert isinstance(metrics.per_rat_mean_queue_depth, dict)
        assert isinstance(metrics.per_rat_dropped_packets, dict)


class TestRBCapacityIntegration:
    """Integration tests for RB-based capacity with simulator."""

    def test_queue_simulator_uses_rb_capacity(self):
        """QueueSimulator should use RB-based capacity for queue limits."""
        sim = QueueSimulator()
        expected_capacity = calculate_queue_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))

        sim._queue_bytes[RATType.PC5] = expected_capacity - 100
        assert sim.can_enqueue(100, RATType.PC5)

        sim._queue_bytes[RATType.PC5] = expected_capacity
        assert not sim.can_enqueue(1, RATType.PC5)

    def test_simulator_respects_rb_tx_limit(self):
        """Simulator should cap packet size at RB-based TX limit."""
        sim = QueueSimulator()

        decision = RATDecision(
            selected_rat=RATType.PC5,
            confidence=0.9,
            predicted_latency_ms=8.0,
            predicted_pdr=0.999,
            all_predictions={},
        )

        dtmc = sim.dtmc_sizers[RATType.PC5]
        dtmc.current_state = dtmc.num_levels - 1
        assert dtmc.current_size == 4096

        packet_decision = sim.decide_packet_size(decision, current_time=1.0)
        max_phy = calculate_tx_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))
        assert packet_decision.packet_size_bytes <= max_phy

    def test_tx_time_calculation_consistent(self):
        """TX time should be consistent with capacity formula."""
        from queuesim.phy_layer import calculate_subframe_capacity_bits, calculate_tx_time_ms

        packet_size = 1000
        for rat in [RATType.PC5, RATType.FiveG, RATType.DSRC]:
            bits_per_ms = calculate_subframe_capacity_bits(rat)
            tx_time = calculate_tx_time_ms(packet_size, rat)
            expected_base = (packet_size * 8) / bits_per_ms
            assert tx_time >= expected_base * 0.9
