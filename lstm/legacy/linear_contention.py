"""
Archived linear contention model (pre-RB-consumption).

This module contains the original linear PDR penalty and latency delay
factors that were used before the RB-consumption model was implemented
in queuesim/phy_layer.py.

The linear model applied:
    PDR_factor = max(0.5, 1.0 - (N-1) * penalty_per_vehicle)
    Latency = base_latency * (1 + (N-1) * delay_factor)

These were replaced by PHY-aware contention functions that use actual
channel capacity, subchannel allocation (PC5), CSMA/CA collision
probability (DSRC), and scheduling delay (5G).
"""
from api_types import RATType


# PDR penalty per additional vehicle sharing the same RAT
PDR_PENALTY = {
    RATType.FiveG: 0.01,   # scheduled access, minimal collision
    RATType.PC5: 0.03,     # semi-persistent sensing, moderate collision
    RATType.DSRC: 0.04,    # CSMA/CA contention, highest collision risk
}

# Latency increase factor per additional vehicle
LATENCY_DELAY_FACTOR = {
    RATType.FiveG: 0.05,
    RATType.PC5: 0.08,
    RATType.DSRC: 0.12,
}


def apply_pdr_contention(base_pdr: float, n_vehicles: int, rat: RATType) -> float:
    """Degrade PDR based on number of vehicles sharing the same RAT."""
    if n_vehicles <= 1:
        return base_pdr
    penalty = PDR_PENALTY.get(rat, 0.03)
    contention_factor = max(0.5, 1.0 - (n_vehicles - 1) * penalty)
    return base_pdr * contention_factor


def apply_latency_contention(
    base_latency: float, n_vehicles: int, rat: RATType,
) -> float:
    """Increase latency based on number of vehicles sharing the same RAT."""
    if n_vehicles <= 1:
        return base_latency
    delay_factor = LATENCY_DELAY_FACTOR.get(rat, 0.08)
    contention_delay = base_latency * (n_vehicles - 1) * delay_factor
    return base_latency + contention_delay
