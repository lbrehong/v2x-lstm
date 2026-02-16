"""
Tests for queuesim/sim_types.py - TransmissionRecord and SimulationMetrics.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import simpy
    SIMPY_AVAILABLE = True
except ImportError:
    SIMPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SIMPY_AVAILABLE,
    reason="SimPy not installed. Install with: sudo apt install python3-simpy3"
)

from api_types import RATType

if SIMPY_AVAILABLE:
    from queuesim.sim_types import SimulationMetrics, TransmissionRecord


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

    def test_new_phy_fields_exist(self):
        """SimulationMetrics should have PHY fields."""
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
        assert d["dropped_packets"] == 5
        assert d["max_queue_depth"] == 10
        assert d["mean_tx_time_ms"] == 2.5
        assert d["capacity_limited_count"] == 3
