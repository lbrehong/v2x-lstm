"""
PHY-Layer helper functions for RAT capacity and transmission modeling.

Provides OFDM-based capacity calculations for 5G NR, C-V2X PC5, and DSRC (802.11p),
including subframe capacity, TX budget, queue sizing, and transmission time estimation.

All calculations are based on 3GPP/IEEE PHY-layer parameters defined in config.RAT_PHY_CONFIG.
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from api_types import RATType
from config import RAT_PHY_CONFIG, get_tx_interval_ms


def get_phy_config(rat: RATType) -> Dict:
    """
    Get PHY-layer configuration for a RAT.

    Args:
        rat: The RAT type

    Returns:
        PHY configuration dictionary, or defaults if RAT not found
    """
    if rat == RATType.UNAVAILABLE:
        return RAT_PHY_CONFIG.get("5g", {})  # Default to 5G
    return RAT_PHY_CONFIG.get(rat.value, RAT_PHY_CONFIG.get("5g", {}))


def calculate_subframe_capacity_bits(rat: RATType) -> int:
    """
    Calculate data bits per subframe/slot using unified OFDM formula.

    All RATs use: capacity = total_data_subcarriers x Nsym x Rmod x CR

    For 5G/PC5 (RB-based):  total_data_subcarriers = NSC x NRB
        NSC   = Subcarriers per RB (12)
        NRB   = Resource blocks (direct count or subchannel x n_subchannels)
    For DSRC (802.11p):      total_data_subcarriers = n_data_subcarriers (52)

    Common parameters:
        Nsym  = Symbols per subframe/slot/ms
        Rmod  = Bits per symbol (QPSK=2, 16QAM=4, 64QAM=6)
        CR    = Coding rate

    Args:
        rat: The RAT type

    Returns:
        Data bits per subframe/slot/ms
    """
    config = get_phy_config(rat)

    r_mod = config.get("modulation_order", 2)
    cr = config.get("coding_rate", 0.5)
    n_sym = config.get("n_symbols", 14)

    if rat.value == "dsrc":
        # DSRC 802.11p: OFDM-level formula (no RB grouping)
        # capacity = n_data_subcarriers x n_symbols x Rmod x CR
        n_data_sc = config.get("n_data_subcarriers", 52)
        return int(n_data_sc * n_sym * r_mod * cr)

    # 5G / PC5: RB-based formula
    # capacity = (NSC x NRB) x Nsym x Rmod x CR
    n_sc = config.get("n_subcarriers", 12)

    if rat.value == "pc5":
        # PC5: subchannel-based allocation
        n_rbs_per_subchannel = config.get("n_rbs_per_subchannel", 10)
        n_subchannels = config.get("n_subchannels", 1)
        n_rb = n_rbs_per_subchannel * n_subchannels
    else:
        # 5G NR: direct RB count
        n_rb = config.get("n_rbs", 106)

    return int(n_sc * n_sym * n_rb * r_mod * cr)


def calculate_tx_capacity_bytes(rat: RATType, tx_interval_ms: int) -> int:
    """
    Calculate maximum bytes transmittable in one TX interval.

    Args:
        rat: The RAT type
        tx_interval_ms: TX interval in milliseconds

    Returns:
        Maximum bytes per TX interval
    """
    bits_per_ms = calculate_subframe_capacity_bits(rat)
    total_bits = bits_per_ms * tx_interval_ms
    return total_bits // 8


def calculate_queue_capacity_bytes(rat: RATType, tx_interval_ms: int) -> int:
    """
    Calculate queue capacity in bytes based on TX capacity and multiplier.

    Args:
        rat: The RAT type
        tx_interval_ms: TX interval in milliseconds

    Returns:
        Queue capacity in bytes
    """
    config = get_phy_config(rat)
    queue_multiplier = config.get("queue_multiplier", 2)
    tx_capacity = calculate_tx_capacity_bytes(rat, tx_interval_ms)
    return tx_capacity * queue_multiplier


def can_transmit(packet_size: int, rat: RATType) -> bool:
    """
    Check if packet fits within one TX interval capacity.

    Args:
        packet_size: Packet size in bytes
        rat: Selected RAT type

    Returns:
        True if packet can be transmitted in one TX interval
    """
    max_bytes = calculate_tx_capacity_bytes(rat, get_tx_interval_ms(rat))
    return packet_size <= max_bytes


def get_queue_capacity_packets(rat: RATType, packet_size: int) -> int:
    """
    Calculate queue capacity in packets based on byte budget.

    Args:
        rat: Selected RAT type
        packet_size: Current packet size in bytes

    Returns:
        Maximum number of packets the queue can hold
    """
    if packet_size <= 0:
        return 100  # Default capacity
    capacity_bytes = calculate_queue_capacity_bytes(rat, get_tx_interval_ms(rat))
    return max(1, capacity_bytes // packet_size)


def get_max_packet_size(rat: RATType) -> int:
    """
    Get maximum allowed packet size for a RAT based on TX capacity.

    Args:
        rat: Selected RAT type

    Returns:
        Maximum packet size in bytes
    """
    return calculate_tx_capacity_bytes(rat, get_tx_interval_ms(rat))


def calculate_tx_time_ms(packet_size: int, rat: RATType) -> float:
    """
    Calculate transmission time based on PHY characteristics.

    For scheduled access (5G/PC5): TX time based on bits/ms capacity
    For contention-based (DSRC): TX time includes contention delay

    Args:
        packet_size: Packet size in bytes
        rat: Selected RAT type

    Returns:
        Transmission time in milliseconds
    """
    config = get_phy_config(rat)
    bits_per_ms = calculate_subframe_capacity_bits(rat)

    if bits_per_ms <= 0:
        return 1.0  # Default fallback

    # Calculate base TX time from capacity
    packet_bits = packet_size * 8
    tx_time_ms = packet_bits / bits_per_ms

    if rat.value == "dsrc":
        # DSRC: add average contention delay (simplified model)
        avg_backoff_ms = config.get("contention_window_min", 15) * 0.009 / 2
        return tx_time_ms + avg_backoff_ms

    return tx_time_ms


def get_base_latency_ms(rat: RATType) -> float:
    """
    Get base end-to-end latency for a RAT.

    Args:
        rat: Selected RAT type

    Returns:
        Base latency in milliseconds
    """
    config = get_phy_config(rat)
    return config.get("base_latency_ms", 10.0)


# =========================================================================
# Multi-vehicle contention model (RB-consumption based)
# =========================================================================

@dataclass
class ChannelAllocation:
    """Result of channel allocation for a single vehicle."""
    vehicle_idx: int
    transmitted: bool
    collision: bool
    deferred: bool


def compute_channel_utilization(
    n_vehicles: int, packet_sizes: List[int], rat: RATType,
) -> float:
    """
    Compute channel utilization as total demand / capacity per TX interval.

    Args:
        n_vehicles: Number of vehicles transmitting on this RAT
        packet_sizes: Packet size in bytes for each vehicle
        rat: The RAT type

    Returns:
        Utilization ratio (0.0+, can exceed 1.0 when overloaded)
    """
    if n_vehicles <= 0 or not packet_sizes:
        return 0.0
    tx_interval = get_tx_interval_ms(rat)
    capacity_bytes = calculate_tx_capacity_bytes(rat, tx_interval)
    # PC5 capacity is per-subchannel; scale to full channel
    if rat.value == "pc5":
        config = get_phy_config(rat)
        capacity_bytes *= config.get("n_subchannels_total", 5)
    if capacity_bytes <= 0:
        return float("inf")
    total_demand = sum(packet_sizes[:n_vehicles])
    return total_demand / capacity_bytes


def allocate_channel(
    vehicle_demands: List[int], rat: RATType,
    rng: np.random.RandomState,
) -> List[ChannelAllocation]:
    """
    Allocate channel resources to vehicles using RAT-specific access mechanisms.

    - 5G (scheduled): gNB fills RBs in random order; excess vehicles deferred.
    - PC5 (SPS Mode 2): Each vehicle randomly selects a subchannel; collisions
      occur when multiple vehicles pick the same subchannel.
    - DSRC (CSMA/CA): All attempt; collision probability from Bianchi-inspired
      model scaled by utilization.

    Args:
        vehicle_demands: Packet size in bytes per vehicle
        rat: The RAT type
        rng: NumPy RandomState for reproducibility

    Returns:
        List of ChannelAllocation, one per vehicle
    """
    n = len(vehicle_demands)
    if n == 0:
        return []

    config = get_phy_config(rat)
    tx_interval = get_tx_interval_ms(rat)
    capacity_bytes = calculate_tx_capacity_bytes(rat, tx_interval)

    if rat.value == "5g":
        # Scheduled access: gNB serves vehicles in random order until capacity exhausted
        order = rng.permutation(n)
        used = 0
        results = [None] * n
        for idx in order:
            if used + vehicle_demands[idx] <= capacity_bytes:
                used += vehicle_demands[idx]
                results[idx] = ChannelAllocation(idx, transmitted=True, collision=False, deferred=False)
            else:
                results[idx] = ChannelAllocation(idx, transmitted=False, collision=False, deferred=True)
        return results

    if rat.value == "pc5":
        # SPS: each vehicle selects a (subchannel, subframe) resource slot
        n_subch = config.get("n_subchannels_total", 5)
        subframe_ms = config.get("subframe_duration_ms", 1.0)
        n_subframes = int(tx_interval / subframe_ms)
        n_resources = n_subch * n_subframes  # 5 × 20 = 100
        selections = rng.randint(0, n_resources, size=n)
        occupancy = {}
        for i, s in enumerate(selections):
            occupancy.setdefault(int(s), []).append(i)
        results = [None] * n
        for slot, vehicles in occupancy.items():
            if len(vehicles) == 1:
                i = vehicles[0]
                results[i] = ChannelAllocation(i, transmitted=True, collision=False, deferred=False)
            else:
                for i in vehicles:
                    results[i] = ChannelAllocation(i, transmitted=False, collision=True, deferred=False)
        return results

    # DSRC (CSMA/CA): probabilistic collision based on utilization + CW
    cw_min = config.get("contention_window_min", 15)
    utilization = compute_channel_utilization(n, vehicle_demands, rat)

    results = []
    for i in range(n):
        if n <= 1:
            p_coll = 0.0
        else:
            # Bianchi-inspired: prob at least one other picks same slot
            p_coll = (1.0 - ((cw_min - 1) / cw_min) ** (n - 1)) * min(utilization, 1.0)
            # When overloaded, collisions rise sharply
            if utilization > 1.0:
                p_coll = min(1.0, p_coll + (utilization - 1.0) * 0.3)
        collided = rng.random() < p_coll
        results.append(ChannelAllocation(i, transmitted=not collided, collision=collided, deferred=False))
    return results


def compute_contention_pdr(
    base_pdr: float, utilization: float, rat: RATType, n_vehicles: int,
) -> float:
    """
    Adjust PDR for multi-vehicle contention based on RAT access mechanism.

    - 5G (scheduled): No PDR loss from contention (gNB manages resources).
    - PC5 (SPS): Birthday-problem subchannel collision model.
    - DSRC (CSMA/CA): Collision probability from CW and utilization.

    Args:
        base_pdr: PDR after packet-size correction (before contention)
        utilization: Channel utilization ratio from compute_channel_utilization
        rat: The RAT type
        n_vehicles: Number of vehicles on this RAT

    Returns:
        Contention-adjusted PDR
    """
    if n_vehicles <= 1:
        return base_pdr

    config = get_phy_config(rat)

    if rat.value == "5g":
        # Scheduled access: no collision-based PDR loss
        return base_pdr

    if rat.value == "pc5":
        # SPS selects (subchannel, subframe) pairs — resource pool is S × T
        s = config.get("n_subchannels_total", 5)
        subframe_ms = config.get("subframe_duration_ms", 1.0)
        tx_interval = get_tx_interval_ms(rat)
        resources = int(s * (tx_interval / subframe_ms))
        p_no_collision = ((resources - 1) / resources) ** (n_vehicles - 1)
        return base_pdr * p_no_collision

    # DSRC (CSMA/CA)
    cw_min = config.get("contention_window_min", 15)
    p_success_slot = ((cw_min - 1) / cw_min) ** (n_vehicles - 1)
    # Scale by utilization: at low utilization, contention is mild
    p_success = p_success_slot ** min(utilization, 1.0)
    # When overloaded, additional sharp degradation
    if utilization > 1.0:
        p_success *= max(0.1, 1.0 / utilization)
    return base_pdr * p_success


def compute_contention_latency(
    base_latency: float, utilization: float, n_vehicles: int, rat: RATType,
) -> float:
    """
    Add contention-induced latency based on RAT access mechanism.

    - 5G (scheduled): Average scheduling queue delay.
    - PC5 (SPS): Expected re-sensing delay on collision.
    - DSRC (CSMA/CA): Backoff delay scaled by CW and utilization.

    Args:
        base_latency: Latency before contention effects (ms)
        utilization: Channel utilization ratio
        n_vehicles: Number of vehicles on this RAT
        rat: The RAT type

    Returns:
        Latency with contention delay added (ms)
    """
    if n_vehicles <= 1:
        return base_latency

    config = get_phy_config(rat)

    if rat.value == "5g":
        # Average scheduling queue position: (N-1)/2 * per-UE delay
        sched_delay = config.get("scheduling_delay_per_ue_ms", 0.5)
        return base_latency + (n_vehicles - 1) * sched_delay / 2.0

    if rat.value == "pc5":
        # On collision, vehicle must re-sense for ~2 subframes
        s = config.get("n_subchannels_total", 5)
        subframe_ms = config.get("subframe_duration_ms", 1.0)
        tx_interval = get_tx_interval_ms(rat)
        resources = int(s * (tx_interval / subframe_ms))
        p_collision = 1.0 - ((resources - 1) / resources) ** (n_vehicles - 1)
        return base_latency + p_collision * subframe_ms * 2.0

    # DSRC (CSMA/CA): backoff grows with CW and utilization
    cw_min = config.get("contention_window_min", 15)
    slot_time_us = config.get("slot_time_us", 13)
    slot_time_ms = slot_time_us / 1000.0
    # Effective CW grows with number of vehicles and utilization
    effective_cw = cw_min * max(1.0, utilization) * (1.0 + 0.1 * (n_vehicles - 1))
    # Average backoff = CW/2 * slot_time
    return base_latency + effective_cw * slot_time_ms / 2.0
