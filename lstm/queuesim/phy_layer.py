"""
PHY-Layer helper functions for RAT capacity and transmission modeling.

Provides OFDM-based capacity calculations for 5G NR, C-V2X PC5, and DSRC (802.11p),
including subframe capacity, TX budget, queue sizing, and transmission time estimation.

All calculations are based on 3GPP/IEEE PHY-layer parameters defined in config.RAT_PHY_CONFIG.
"""
from typing import Dict

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
