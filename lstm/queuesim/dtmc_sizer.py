"""
Discrete Time Markov Chain (DTMC) adaptive packet sizer.

Implements a state machine that adjusts packet size based on observed PDR.
Includes a PDR correction model for translating predictions made at one
packet size to estimates at a different packet size.
"""
from typing import Optional, List, Dict, Tuple

import numpy as np

from api_types import RATType
from config import (
    DTMC_PACKET_SIZES,
    DTMC_THRESHOLD_HIGH, DTMC_THRESHOLD_LOW,
    DTMC_WINDOW_SECONDS,
    DEFAULT_TX_INTERVAL_MS,
)


class DTMCPacketSizer:
    """
    Discrete Time Markov Chain for adaptive packet sizing.

    Packet sizes: 1024 -> 2048 -> 3072 -> 4096 bytes (increments of 1024)

    State transitions based on moving window average PDR:
    - PDR > 0.99: Increase packet size (more data per transmission)
    - PDR < 0.95: Decrease packet size (improve reliability)
    - Otherwise: Maintain current size

    State Diagram:
        [1024] <--PDR<0.95-- [2048] <--PDR<0.95-- [3072] <--PDR<0.95-- [4096]
           |                    |                    |                    |
           +----PDR>0.99------>-+----PDR>0.99------>-+----PDR>0.99------>-+
    """

    def __init__(
        self,
        rat: RATType,
        threshold_high: float = DTMC_THRESHOLD_HIGH,
        threshold_low: float = DTMC_THRESHOLD_LOW,
        window_seconds: float = DTMC_WINDOW_SECONDS,
        tx_rate_hz: float = 1000 / DEFAULT_TX_INTERVAL_MS,
    ):
        """
        Initialize DTMC packet sizer.

        Args:
            rat: RAT type
            threshold_high: PDR above this -> increase size (default 0.99)
            threshold_low: PDR below this -> decrease size (default 0.95)
            window_seconds: Moving window for PDR averaging (default 1.0s)
            tx_rate_hz: Transmission rate for window size calculation
        """
        self.rat = rat
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low

        # Fixed packet size levels: 1024, 2048, 3072, 4096
        self.size_levels = np.array(DTMC_PACKET_SIZES, dtype=int)
        self.num_levels = len(self.size_levels)

        # Current state (start at minimum size for safety)
        self.current_state = 0

        # Moving window for PDR: store (timestamp, success) tuples
        self.window_seconds = window_seconds
        self.window_size = int(window_seconds * tx_rate_hz)  # ~50 samples for 1s at 50Hz
        self.pdr_history: List[Tuple[float, bool]] = []  # (timestamp, delivered)

        # History for analysis
        self.history: List[Tuple[int, int, float, str]] = []  # (step, state, pdr, action)

    @property
    def current_size(self) -> int:
        """Get current packet size in bytes."""
        return int(self.size_levels[self.current_state])

    @property
    def min_size(self) -> int:
        """Minimum packet size."""
        return int(self.size_levels[0])

    @property
    def max_size(self) -> int:
        """Maximum packet size."""
        return int(self.size_levels[-1])

    def record_outcome(self, timestamp: float, delivered: bool) -> None:
        """
        Record a transmission outcome for PDR calculation.

        Args:
            timestamp: Simulation timestamp
            delivered: Whether packet was successfully delivered
        """
        self.pdr_history.append((timestamp, delivered))

        # Trim old entries outside window
        if len(self.pdr_history) > self.window_size * 2:
            cutoff_time = timestamp - self.window_seconds
            self.pdr_history = [
                (t, d) for t, d in self.pdr_history if t >= cutoff_time
            ]

    def get_window_pdr(self, current_time: float) -> Optional[float]:
        """
        Calculate PDR over the moving window.

        Args:
            current_time: Current simulation timestamp

        Returns:
            Average PDR over window, or None if insufficient data
        """
        if not self.pdr_history:
            return None

        cutoff_time = current_time - self.window_seconds
        window_outcomes = [d for t, d in self.pdr_history if t >= cutoff_time]

        if len(window_outcomes) < 5:  # Minimum samples for reliable estimate
            return None

        return sum(window_outcomes) / len(window_outcomes)

    def transition(self, pdr: float, step: int = 0) -> int:
        """
        Perform DTMC transition based on PDR.

        Transitions are deterministic:
        - PDR > 0.99 and not at max -> increase
        - PDR < 0.95 and not at min -> decrease
        - Otherwise -> stay

        Args:
            pdr: Moving window average PDR (or predicted if no history)
            step: Simulation step for logging

        Returns:
            New packet size in bytes
        """
        action = "stay"

        if pdr > self.threshold_high and self.current_state < self.num_levels - 1:
            # PDR excellent -> increase packet size
            self.current_state += 1
            action = "increase"

        elif pdr < self.threshold_low and self.current_state > 0:
            # PDR degraded -> decrease packet size
            self.current_state -= 1
            action = "decrease"

        self.history.append((step, self.current_state, pdr, action))
        return self.current_size

    def reset(self):
        """Reset to initial state."""
        self.current_state = 0  # Start at minimum size
        self.pdr_history = []
        self.history = []

    def get_statistics(self) -> Dict:
        """Get statistics about DTMC behavior."""
        if not self.history:
            return {}

        states = [h[1] for h in self.history]
        actions = [h[3] for h in self.history]

        return {
            "mean_state": np.mean(states),
            "std_state": np.std(states),
            "min_state": min(states),
            "max_state": max(states),
            "increase_count": actions.count("increase"),
            "decrease_count": actions.count("decrease"),
            "stay_count": actions.count("stay"),
            "mean_packet_size": np.mean([self.size_levels[s] for s in states]),
        }


def correct_pdr_for_packet_size(
    base_pdr: float,
    base_size: int,
    target_size: int,
    correction_exponent: float = 0.8
) -> float:
    """
    Apply packet size correction to PDR estimate.

    Uses a power-law model based on bit error probability:
    - Larger packets have more bits that can fail, lowering PDR
    - Smaller packets have fewer bits that can fail, improving PDR

    The model: PDR(size) ~ PDR_base ^ (target_size / base_size)
    This reflects that PDR is roughly (1 - BER)^num_bits

    Args:
        base_pdr: PDR measured/predicted at base packet size
        base_size: Packet size at which PDR was measured (typically 1000 bytes)
        target_size: Target packet size for correction
        correction_exponent: Scaling factor for the size ratio (0.5-1.5 typical)

    Returns:
        Corrected PDR estimate for target packet size
    """
    if target_size <= 0 or base_size <= 0:
        return base_pdr

    if base_pdr <= 0:
        return 0.0

    if base_pdr >= 1.0:
        return 1.0

    # Size ratio determines the exponent
    # Larger target -> higher exponent -> lower PDR (more bits to fail)
    # Smaller target -> lower exponent -> higher PDR (fewer bits to fail)
    size_ratio = target_size / base_size

    # Apply power-law correction with scaling
    # PDR_new = PDR_base ^ (size_ratio ^ exponent)
    effective_exponent = size_ratio ** correction_exponent
    corrected_pdr = base_pdr ** effective_exponent

    return max(0.0, min(1.0, corrected_pdr))
