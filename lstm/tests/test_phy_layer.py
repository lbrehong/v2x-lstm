"""
Tests for queuesim/phy_layer.py - PHY-layer capacity calculations per RAT.

Tests cover:
- PHY configuration for all RATs
- RB-based subframe capacity formula
- TX and queue capacity calculations
- PHY helper functions (can_transmit, tx_time, base_latency)
- Channel contention: utilization, allocation, PDR degradation, latency
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if simpy is available (phy_layer imports may depend on it indirectly)
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
from config import RAT_PHY_CONFIG

if SIMPY_AVAILABLE:
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
        compute_channel_utilization,
        allocate_channel,
        compute_contention_pdr,
        compute_contention_latency,
    )
    from config import get_tx_interval_ms


# =============================================================================
# PHY Configuration Tests
# =============================================================================

class TestPHYConfig:
    """Tests for PHY-layer configuration."""

    def test_config_exists_for_all_rats(self):
        """PHY config should exist for all RAT types."""
        for rat_key in ["5g", "pc5", "dsrc"]:
            assert rat_key in RAT_PHY_CONFIG
            config = RAT_PHY_CONFIG[rat_key]
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


# =============================================================================
# PHY Helper Functions Tests
# =============================================================================

class TestPHYHelperFunctions:
    """Tests for PHY-layer helper functions."""

    def test_get_phy_config_valid_rat(self):
        """Should return config for valid RAT."""
        assert get_phy_config(RATType.FiveG) == RAT_PHY_CONFIG["5g"]
        assert get_phy_config(RATType.DSRC) == RAT_PHY_CONFIG["dsrc"]
        assert get_phy_config(RATType.PC5) == RAT_PHY_CONFIG["pc5"]

    def test_get_phy_config_unavailable_rat(self):
        """Should return default config for UNAVAILABLE RAT."""
        config = get_phy_config(RATType.UNAVAILABLE)
        assert config == RAT_PHY_CONFIG["5g"]

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
        assert capacity == 100

    def test_get_max_packet_size(self):
        """Should return max packet size (TX capacity) for each RAT."""
        for rat in [RATType.FiveG, RATType.PC5, RATType.DSRC]:
            max_size = get_max_packet_size(rat)
            expected = calculate_tx_capacity_bytes(rat, get_tx_interval_ms(rat))
            assert max_size == expected

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


# =============================================================================
# RB-Based Capacity Model Tests
# =============================================================================

class TestRBCapacityModel:
    """Tests for RB-based PHY capacity calculation."""

    def test_pc5_subframe_capacity_formula(self):
        """PC5 capacity should follow dSF = NSC x Nsym x NRB x Rmod x CR."""
        bits = calculate_subframe_capacity_bits(RATType.PC5)
        expected = 12 * 14 * 10 * 2 * 0.5
        assert bits == int(expected)
        assert bits == 1680

    def test_5g_subframe_capacity_formula(self):
        """5G capacity should follow dSF = NSC x Nsym x NRB x Rmod x CR."""
        bits = calculate_subframe_capacity_bits(RATType.FiveG)
        expected = 12 * 14 * 106 * 6 * 0.66
        assert bits == int(expected)

    def test_dsrc_uses_ofdm_capacity_model(self):
        """DSRC should use OFDM formula: 52 subcarriers x 125 sym/ms x QPSK x 1/2 = 6500 bits/ms."""
        bits = calculate_subframe_capacity_bits(RATType.DSRC)
        assert bits == 6500

    def test_tx_capacity_bytes_pc5(self):
        """PC5 TX capacity for 100ms interval."""
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

    def test_capacity_ordering(self):
        """5G should have highest capacity, PC5 lowest."""
        tx_5g = calculate_tx_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        tx_dsrc = calculate_tx_capacity_bytes(RATType.DSRC, get_tx_interval_ms(RATType.DSRC))
        tx_pc5 = calculate_tx_capacity_bytes(RATType.PC5, get_tx_interval_ms(RATType.PC5))
        assert tx_5g > tx_dsrc
        assert tx_dsrc > tx_pc5

    def test_pc5_subchannel_scaling(self):
        """PC5 capacity should scale with number of subchannels."""
        bits_1ch = calculate_subframe_capacity_bits(RATType.PC5)
        assert bits_1ch == 1680

    def test_unavailable_rat_uses_5g_defaults(self):
        """UNAVAILABLE RAT should use 5G defaults for capacity."""
        unavailable_bits = calculate_subframe_capacity_bits(RATType.UNAVAILABLE)
        fiveg_bits = calculate_subframe_capacity_bits(RATType.FiveG)
        assert unavailable_bits == fiveg_bits


# =============================================================================
# Channel Contention Tests
# =============================================================================

class TestComputeChannelUtilization:
    """Tests for compute_channel_utilization."""

    def test_zero_vehicles(self):
        """Zero vehicles should give zero utilization."""
        assert compute_channel_utilization(0, [], RATType.DSRC) == 0.0

    def test_5g_trivial_utilization(self):
        """5G with 20 vehicles @ 2KB should have very low utilization."""
        sizes = [2048] * 20
        util = compute_channel_utilization(20, sizes, RATType.FiveG)
        assert util < 0.01

    def test_pc5_overloaded(self):
        """PC5 with 20 vehicles @ 2KB should be overloaded (utilization > 1.0)."""
        sizes = [2048] * 20
        util = compute_channel_utilization(20, sizes, RATType.PC5)
        assert util > 1.0

    def test_dsrc_moderate(self):
        """DSRC with 20 vehicles @ 2KB should have moderate utilization."""
        sizes = [2048] * 20
        util = compute_channel_utilization(20, sizes, RATType.DSRC)
        assert 0.1 < util < 2.0

    def test_single_vehicle_low(self):
        """Single vehicle should have low utilization on any RAT."""
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            util = compute_channel_utilization(1, [1024], rat)
            assert util < 1.0, f"Single vehicle on {rat.value} should not overload"


class TestAllocateChannel:
    """Tests for allocate_channel."""

    def test_5g_defers_excess(self):
        """5G should defer vehicles when demand exceeds capacity."""
        rng = np.random.RandomState(42)
        cap = calculate_tx_capacity_bytes(RATType.FiveG, get_tx_interval_ms(RATType.FiveG))
        demands = [cap] * 3
        results = allocate_channel(demands, RATType.FiveG, rng)
        transmitted = sum(1 for r in results if r.transmitted)
        deferred = sum(1 for r in results if r.deferred)
        assert transmitted == 1
        assert deferred == 2
        assert all(not r.collision for r in results)

    def test_5g_all_fit_small_packets(self):
        """5G with small packets should transmit all."""
        rng = np.random.RandomState(42)
        demands = [1024] * 20
        results = allocate_channel(demands, RATType.FiveG, rng)
        assert all(r.transmitted for r in results)

    def test_pc5_birthday_collisions(self):
        """PC5 with many vehicles should produce some subchannel collisions."""
        rng = np.random.RandomState(42)
        demands = [1024] * 20
        results = allocate_channel(demands, RATType.PC5, rng)
        collisions = sum(1 for r in results if r.collision)
        assert collisions > 0, "20 vehicles on 5 subchannels should produce collisions"

    def test_dsrc_probabilistic(self):
        """DSRC should produce some collisions with many vehicles."""
        rng = np.random.RandomState(42)
        demands = [2048] * 20
        results = allocate_channel(demands, RATType.DSRC, rng)
        collisions = sum(1 for r in results if r.collision)
        assert collisions >= 0
        assert len(results) == 20

    def test_empty_demands(self):
        """Empty demand list should return empty results."""
        rng = np.random.RandomState(42)
        results = allocate_channel([], RATType.DSRC, rng)
        assert results == []

    def test_single_vehicle_no_collision(self):
        """Single vehicle should always transmit without collision."""
        rng = np.random.RandomState(42)
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            results = allocate_channel([1024], rat, rng)
            assert len(results) == 1
            assert results[0].transmitted
            assert not results[0].collision


class TestComputeContentionPdr:
    """Tests for compute_contention_pdr."""

    def test_single_vehicle_unchanged(self):
        """Single vehicle should return base PDR."""
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            assert compute_contention_pdr(0.99, 0.5, rat, 1) == 0.99

    def test_5g_no_loss(self):
        """5G should have no PDR loss from contention (scheduled access)."""
        result = compute_contention_pdr(0.99, 0.5, RATType.FiveG, 20)
        assert result == 0.99

    def test_pc5_birthday_degradation(self):
        """PC5 should degrade PDR via birthday-problem subchannel collisions."""
        result = compute_contention_pdr(0.99, 1.0, RATType.PC5, 20)
        assert result < 0.99 * 0.1

    def test_dsrc_cbr_degradation(self):
        """DSRC should degrade PDR with moderate utilization."""
        result = compute_contention_pdr(0.99, 0.5, RATType.DSRC, 10)
        assert result < 0.99

    def test_zero_pdr_stays_zero(self):
        """Zero PDR should remain zero."""
        assert compute_contention_pdr(0.0, 0.5, RATType.PC5, 10) == 0.0

    def test_ordering_5g_best_pc5_worst(self):
        """With N=20, 5G should maintain PDR, PC5 should be worst (birthday)."""
        pdr_5g = compute_contention_pdr(0.99, 0.5, RATType.FiveG, 20)
        pdr_dsrc = compute_contention_pdr(0.99, 0.5, RATType.DSRC, 20)
        pdr_pc5 = compute_contention_pdr(0.99, 1.9, RATType.PC5, 20)
        assert pdr_5g > pdr_dsrc
        assert pdr_5g > pdr_pc5


class TestComputeContentionLatency:
    """Tests for compute_contention_latency."""

    def test_single_vehicle_unchanged(self):
        """Single vehicle should return base latency."""
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            assert compute_contention_latency(10.0, 0.5, 1, rat) == 10.0

    def test_5g_scheduling_delay(self):
        """5G should add scheduling delay proportional to N."""
        result = compute_contention_latency(10.0, 0.001, 20, RATType.FiveG)
        assert result == pytest.approx(10.0 + 19 * 0.5 / 2, abs=0.01)

    def test_dsrc_backoff_scaling(self):
        """DSRC latency should increase with utilization and vehicles."""
        lat_low = compute_contention_latency(10.0, 0.1, 5, RATType.DSRC)
        lat_high = compute_contention_latency(10.0, 0.9, 20, RATType.DSRC)
        assert lat_high > lat_low

    def test_pc5_collision_delay(self):
        """PC5 should add re-sensing delay based on collision probability."""
        result = compute_contention_latency(10.0, 1.0, 10, RATType.PC5)
        assert result > 10.0
