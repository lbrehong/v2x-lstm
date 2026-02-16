"""
Tests for queuesim/dtmc_sizer.py - DTMC adaptive packet sizer.

Tests cover:
- DTMCPacketSizer initialization and packet size levels
- Transition logic: increase, decrease, stay
- State boundary enforcement (min/max)
- History tracking and statistics
- Moving window PDR calculation
- PDR correction for packet size
"""
import pytest
import numpy as np
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
from config import DTMC_PACKET_SIZES

if SIMPY_AVAILABLE:
    from queuesim.dtmc_sizer import DTMCPacketSizer, correct_pdr_for_packet_size


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
        assert dtmc.current_state == 0
        assert len(dtmc.size_levels) == 4
        assert dtmc.min_size == 1024
        assert dtmc.max_size == 4096

    def test_packet_sizes_are_1024_increments(self):
        """Packet sizes should be 1024, 2048, 3072, 4096."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        assert list(dtmc.size_levels) == DTMC_PACKET_SIZES
        assert list(dtmc.size_levels) == [1024, 2048, 3072, 4096]

    def test_all_rats_same_sizes(self):
        """All RATs should use the same packet size levels."""
        for rat in [RATType.FiveG, RATType.DSRC, RATType.PC5]:
            dtmc = DTMCPacketSizer(rat)
            assert list(dtmc.size_levels) == [1024, 2048, 3072, 4096]

    def test_high_pdr_increases(self):
        """PDR > 0.99 should increase packet size."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        assert dtmc.current_state == 0
        assert dtmc.current_size == 1024

        dtmc.transition(0.995, step=1)
        assert dtmc.current_state == 1
        assert dtmc.current_size == 2048

    def test_low_pdr_decreases(self):
        """PDR < 0.95 should decrease packet size."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.current_state = 2

        dtmc.transition(0.94, step=1)
        assert dtmc.current_state == 1
        assert dtmc.current_size == 2048

    def test_mid_pdr_stays(self):
        """PDR between 0.95 and 0.99 should maintain size."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.current_state = 1

        dtmc.transition(0.97, step=1)
        assert dtmc.current_state == 1
        assert dtmc.current_size == 2048

    def test_cannot_go_below_minimum(self):
        """State should not go below 0."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.current_state = 0

        dtmc.transition(0.50, step=1)
        assert dtmc.current_state == 0
        assert dtmc.current_size == 1024

    def test_cannot_go_above_maximum(self):
        """State should not exceed max."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.current_state = 3

        dtmc.transition(0.999, step=1)
        assert dtmc.current_state == 3
        assert dtmc.current_size == 4096

    def test_history_tracking(self):
        """Transitions should be recorded in history."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.transition(0.999, step=1)
        dtmc.transition(0.94, step=2)
        dtmc.transition(0.97, step=3)

        assert len(dtmc.history) == 3
        assert dtmc.history[0][0] == 1
        assert dtmc.history[2][0] == 3

    def test_reset_clears_state(self):
        """Reset should return to initial state (minimum)."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.current_state = 3
        dtmc.transition(0.97, step=1)

        dtmc.reset()
        assert dtmc.current_state == 0
        assert dtmc.current_size == 1024
        assert len(dtmc.history) == 0

    def test_statistics(self):
        """Statistics should compute correctly."""
        dtmc = DTMCPacketSizer(RATType.FiveG)
        dtmc.transition(0.999, step=1)
        dtmc.transition(0.94, step=2)
        dtmc.transition(0.97, step=3)

        stats = dtmc.get_statistics()
        assert "mean_state" in stats
        assert stats["increase_count"] == 1
        assert stats["decrease_count"] == 1
        assert stats["stay_count"] == 1

    def test_moving_window_pdr(self):
        """Should calculate PDR from moving window."""
        dtmc = DTMCPacketSizer(RATType.FiveG, window_seconds=1.0, tx_rate_hz=10)

        for i in range(10):
            dtmc.record_outcome(i * 0.1, True)

        pdr = dtmc.get_window_pdr(1.0)
        assert pdr == 1.0

        for i in range(5):
            dtmc.record_outcome(1.0 + i * 0.1, False)

        pdr = dtmc.get_window_pdr(1.5)
        assert pdr is not None
        assert pdr < 1.0

    def test_window_pdr_insufficient_data(self):
        """Should return None with insufficient data."""
        dtmc = DTMCPacketSizer(RATType.FiveG)

        assert dtmc.get_window_pdr(0.0) is None

        dtmc.record_outcome(0.1, True)
        dtmc.record_outcome(0.2, True)
        assert dtmc.get_window_pdr(0.3) is None
