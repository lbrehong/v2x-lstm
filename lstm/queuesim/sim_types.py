"""
Data types for the queue simulation.

Provides TransmissionRecord (per-packet outcome) and SimulationMetrics
(aggregated statistics) used throughout the queue simulator.
"""
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from api_types import RATType


@dataclass
class TransmissionRecord:
    """Record of a single transmission attempt."""
    timestamp: float
    rat: RATType
    packet_size: int
    predicted_pdr: float
    corrected_pdr: float
    success: bool
    latency_ms: float
    queue_depth: int
    dtmc_state: int


@dataclass
class SimulationMetrics:
    """Aggregated simulation metrics."""
    total_packets: int = 0
    successful_packets: int = 0
    failed_packets: int = 0
    dropped_packets: int = 0  # Dropped due to queue overflow or size limit
    total_bytes_sent: int = 0
    successful_bytes: int = 0

    rat_usage: Dict[RATType, int] = field(default_factory=dict)
    rat_switches: int = 0

    mean_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    mean_queue_depth: float = 0.0
    max_queue_depth: int = 0

    mean_packet_size: float = 0.0
    dtmc_increases: int = 0
    dtmc_decreases: int = 0

    # PHY-layer metrics
    mean_tx_time_ms: float = 0.0
    capacity_limited_count: int = 0  # Packets capped at max TX capacity

    # Per-RAT metrics
    per_rat_max_queue_depth: Dict[RATType, int] = field(default_factory=dict)
    per_rat_mean_queue_depth: Dict[RATType, float] = field(default_factory=dict)
    per_rat_dropped_packets: Dict[RATType, int] = field(default_factory=dict)
    per_rat_pdr: Dict[RATType, float] = field(default_factory=dict)
    per_rat_mean_latency_ms: Dict[RATType, float] = field(default_factory=dict)
    per_rat_max_latency_ms: Dict[RATType, float] = field(default_factory=dict)
    per_rat_throughput_bps: Dict[RATType, float] = field(default_factory=dict)

    records: List[TransmissionRecord] = field(default_factory=list)

    @property
    def pdr(self) -> float:
        """Achieved PDR."""
        if self.total_packets == 0:
            return 0.0
        return self.successful_packets / self.total_packets

    @property
    def throughput_bps(self) -> float:
        """Effective throughput in bytes per second."""
        if not self.records:
            return 0.0
        duration = self.records[-1].timestamp - self.records[0].timestamp
        if duration <= 0:
            return 0.0
        return self.successful_bytes / duration

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV output."""
        return {
            "total_packets": self.total_packets,
            "successful_packets": self.successful_packets,
            "failed_packets": self.failed_packets,
            "dropped_packets": self.dropped_packets,
            "pdr": self.pdr,
            "total_bytes_sent": self.total_bytes_sent,
            "successful_bytes": self.successful_bytes,
            "throughput_bps": self.throughput_bps,
            "rat_switches": self.rat_switches,
            "mean_latency_ms": self.mean_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "mean_queue_depth": self.mean_queue_depth,
            "max_queue_depth": self.max_queue_depth,
            "mean_packet_size": self.mean_packet_size,
            "dtmc_increases": self.dtmc_increases,
            "dtmc_decreases": self.dtmc_decreases,
            "mean_tx_time_ms": self.mean_tx_time_ms,
            "capacity_limited_count": self.capacity_limited_count,
            **{f"rat_{rat.value}_count": count for rat, count in self.rat_usage.items()},
            **{f"max_queue_depth_{rat.value}": depth for rat, depth in self.per_rat_max_queue_depth.items()},
            **{f"mean_queue_depth_{rat.value}": depth for rat, depth in self.per_rat_mean_queue_depth.items()},
            **{f"dropped_packets_{rat.value}": count for rat, count in self.per_rat_dropped_packets.items()},
            **{f"pdr_{rat.value}": val for rat, val in self.per_rat_pdr.items()},
            **{f"mean_latency_ms_{rat.value}": val for rat, val in self.per_rat_mean_latency_ms.items()},
            **{f"max_latency_ms_{rat.value}": val for rat, val in self.per_rat_max_latency_ms.items()},
            **{f"throughput_bps_{rat.value}": val for rat, val in self.per_rat_throughput_bps.items()},
        }
