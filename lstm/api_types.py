"""
Data structures for RAT Selection + Packet Queue Simulator Integration.

This module defines the API interface that allows an external packet queue
simulator to integrate with the RAT prediction project using a hierarchical
cooperative control architecture:

- RAT Selection (This Project): Decides WHICH network to use (strategic layer)
- Queue Simulator (External): Decides HOW to size packets (tactical layer)
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any
from enum import Enum


class RATType(Enum):
    """Radio Access Technology types supported by the system."""
    DSRC = "dsrc"
    PC5 = "pc5"
    FiveG = "5g"
    UNAVAILABLE = "NaN"

    @classmethod
    def from_string(cls, value: str) -> "RATType":
        """Convert string representation to RATType enum."""
        value_lower = value.lower()
        if value_lower == "dsrc":
            return cls.DSRC
        elif value_lower == "pc5":
            return cls.PC5
        elif value_lower in ("5g", "fiveg"):
            return cls.FiveG
        elif value_lower == "nan":
            return cls.UNAVAILABLE
        else:
            raise ValueError(f"Unknown RAT type: {value}")


@dataclass
class NetworkState:
    """
    Input: Current network measurements for all RATs.

    Represents a snapshot of network conditions at a specific location
    and time, including measurements from all available RATs.

    Attributes:
        timestamp_ms: Unix timestamp in milliseconds (must be >= 0)
        latitude: GPS latitude coordinate (-90 to 90)
        longitude: GPS longitude coordinate (-180 to 180)
        dsrc_*: DSRC (802.11p) measurements
        pc5_*: C-V2X PC5 sidelink measurements
        fiveg_*: 5G SA measurements
    """
    timestamp_ms: int
    latitude: float
    longitude: float

    def __post_init__(self):
        """Validate coordinate ranges."""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Latitude must be -90 to 90, got {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Longitude must be -180 to 180, got {self.longitude}")

    # DSRC measurements (None if unavailable)
    dsrc_latency_ms: Optional[float] = None
    dsrc_pdr: Optional[float] = None
    dsrc_rsrp_1: Optional[float] = None
    dsrc_rsrp_2: Optional[float] = None

    # PC5/C-V2X measurements (None if unavailable)
    pc5_latency_ms: Optional[float] = None
    pc5_pdr: Optional[float] = None

    # 5G SA measurements (None if unavailable)
    fiveg_latency_ms: Optional[float] = None
    fiveg_pdr: Optional[float] = None
    fiveg_sinr: Optional[float] = None
    fiveg_rsrp: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/file output."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "dsrc_latency_ms": self.dsrc_latency_ms,
            "dsrc_pdr": self.dsrc_pdr,
            "dsrc_rsrp_1": self.dsrc_rsrp_1,
            "dsrc_rsrp_2": self.dsrc_rsrp_2,
            "pc5_latency_ms": self.pc5_latency_ms,
            "pc5_pdr": self.pc5_pdr,
            "fiveg_latency_ms": self.fiveg_latency_ms,
            "fiveg_pdr": self.fiveg_pdr,
            "fiveg_sinr": self.fiveg_sinr,
            "fiveg_rsrp": self.fiveg_rsrp,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "NetworkState":
        """Create NetworkState from dictionary."""
        return cls(
            timestamp_ms=int(d.get("timestamp_ms", 0)),
            latitude=float(d["latitude"]),
            longitude=float(d["longitude"]),
            dsrc_latency_ms=d.get("dsrc_latency_ms"),
            dsrc_pdr=d.get("dsrc_pdr"),
            dsrc_rsrp_1=d.get("dsrc_rsrp_1"),
            dsrc_rsrp_2=d.get("dsrc_rsrp_2"),
            pc5_latency_ms=d.get("pc5_latency_ms"),
            pc5_pdr=d.get("pc5_pdr"),
            fiveg_latency_ms=d.get("fiveg_latency_ms"),
            fiveg_pdr=d.get("fiveg_pdr"),
            fiveg_sinr=d.get("fiveg_sinr"),
            fiveg_rsrp=d.get("fiveg_rsrp"),
        )


@dataclass
class QueueContext:
    """
    Input from Queue Simulator: Current queue state.

    Provides context about the packet queue to enable queue-aware
    RAT selection decisions.

    Attributes:
        queue_depth: Number of packets waiting in queue (must be >= 0)
        avg_packet_size_bytes: Average packet size in the queue (must be > 0)
        urgency_level: 0.0 (low) to 1.0 (critical)
        recent_pdr_trend: -1.0 (declining) to +1.0 (improving)
        target_latency_ms: Application latency requirement (must be > 0)
        target_pdr: Application PDR requirement (0.0 to 1.0)
    """
    queue_depth: int
    avg_packet_size_bytes: int
    urgency_level: float  # 0.0 (low) to 1.0 (critical)
    recent_pdr_trend: float  # -1.0 (declining) to +1.0 (improving)
    target_latency_ms: float
    target_pdr: float

    def __post_init__(self):
        """Validate ranges."""
        if self.queue_depth < 0:
            raise ValueError(f"queue_depth must be >= 0, got {self.queue_depth}")
        if not (0.0 <= self.urgency_level <= 1.0):
            raise ValueError(f"urgency_level must be 0.0-1.0, got {self.urgency_level}")
        if not (-1.0 <= self.recent_pdr_trend <= 1.0):
            raise ValueError(f"recent_pdr_trend must be -1.0 to 1.0, got {self.recent_pdr_trend}")
        if not (0.0 <= self.target_pdr <= 1.0):
            raise ValueError(f"target_pdr must be 0.0-1.0, got {self.target_pdr}")

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/file output."""
        return {
            "queue_depth": self.queue_depth,
            "avg_packet_size_bytes": self.avg_packet_size_bytes,
            "urgency_level": self.urgency_level,
            "recent_pdr_trend": self.recent_pdr_trend,
            "target_latency_ms": self.target_latency_ms,
            "target_pdr": self.target_pdr,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "QueueContext":
        """Create QueueContext from dictionary."""
        return cls(
            queue_depth=int(d["queue_depth"]),
            avg_packet_size_bytes=int(d["avg_packet_size_bytes"]),
            urgency_level=float(d["urgency_level"]),
            recent_pdr_trend=float(d["recent_pdr_trend"]),
            target_latency_ms=float(d["target_latency_ms"]),
            target_pdr=float(d["target_pdr"]),
        )


@dataclass
class RATDecision:
    """
    Output: RAT selection result with predictions.

    Contains the selected RAT along with predictions for all RATs
    and optional queue-aware recommendations.

    Attributes:
        selected_rat: The chosen RAT for transmission
        confidence: Selection confidence 0.0 to 1.0
        predicted_latency_ms: Predicted latency for selected RAT
        predicted_pdr: Predicted PDR for selected RAT
        all_predictions: Predictions for all RATs {RAT: (latency, pdr)}
        recommended_max_packet_size: Optional packet size recommendation
        model_type: Model architecture used (lstm, gru, rnn)
    """
    selected_rat: RATType
    confidence: float  # 0.0 to 1.0
    predicted_latency_ms: float
    predicted_pdr: float
    all_predictions: Dict[RATType, Tuple[float, float]]  # {RAT: (latency, pdr)}
    recommended_max_packet_size: Optional[int] = None
    model_type: str = "lstm"

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/file output."""
        result = {
            "selected_rat": self.selected_rat.value,
            "confidence": self.confidence,
            "pred_latency": self.predicted_latency_ms,
            "pred_pdr": self.predicted_pdr,
            "model_type": self.model_type,
        }
        # Add per-RAT predictions
        for rat, (lat, pdr) in self.all_predictions.items():
            result[f"pred_latency_{rat.value}"] = lat
            result[f"pred_pdr_{rat.value}"] = pdr
        if self.recommended_max_packet_size is not None:
            result["recommended_max_packet_size"] = self.recommended_max_packet_size
        return result

    @classmethod
    def from_dict(cls, d: Dict) -> "RATDecision":
        """Create RATDecision from dictionary."""
        all_predictions = {}
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            lat_key = f"pred_latency_{rat.value}"
            pdr_key = f"pred_pdr_{rat.value}"
            if lat_key in d and pdr_key in d:
                all_predictions[rat] = (float(d[lat_key]), float(d[pdr_key]))

        return cls(
            selected_rat=RATType.from_string(d["selected_rat"]),
            confidence=float(d.get("confidence", 1.0)),
            predicted_latency_ms=float(d["pred_latency"]),
            predicted_pdr=float(d["pred_pdr"]),
            all_predictions=all_predictions,
            recommended_max_packet_size=d.get("recommended_max_packet_size"),
            model_type=d.get("model_type", "lstm"),
        )


@dataclass
class PacketSizeDecision:
    """
    Output from Queue Simulator: Packet sizing parameters.

    Represents the DTMC-based packet sizing decision made by the
    external queue simulator.

    Attributes:
        packet_size_bytes: Chosen packet size
        fragment_count: Number of fragments (1 = no fragmentation)
        priority_level: 802.1Q priority (0-7)
        send_rate_hz: Transmission rate in packets per second
    """
    packet_size_bytes: int
    fragment_count: int  # 1 = no fragmentation
    priority_level: int  # 0-7
    send_rate_hz: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/file output."""
        return {
            "packet_size_bytes": self.packet_size_bytes,
            "fragment_count": self.fragment_count,
            "priority_level": self.priority_level,
            "send_rate_hz": self.send_rate_hz,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PacketSizeDecision":
        """Create PacketSizeDecision from dictionary."""
        return cls(
            packet_size_bytes=int(d["packet_size_bytes"]),
            fragment_count=int(d.get("fragment_count", 1)),
            priority_level=int(d.get("priority_level", 0)),
            send_rate_hz=float(d.get("send_rate_hz", 50.0)),
        )


@dataclass
class TransmissionOutcome:
    """
    Feedback: Actual result after transmission.

    Captures the outcome of a transmission attempt for use in
    model retraining and PDR estimation updates.

    Attributes:
        timestamp_ms: When the transmission completed
        rat_used: Which RAT was used
        packet_size_bytes: Size of transmitted packet
        actual_latency_ms: Measured end-to-end latency
        delivered: Whether packet was successfully delivered
        network_state: Network conditions at transmission time
    """
    timestamp_ms: int
    rat_used: RATType
    packet_size_bytes: int
    actual_latency_ms: float
    delivered: bool
    network_state: NetworkState

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV/file output."""
        result = {
            "timestamp_ms": self.timestamp_ms,
            "rat_used": self.rat_used.value,
            "packet_size_bytes": self.packet_size_bytes,
            "actual_latency_ms": self.actual_latency_ms,
            "delivered": self.delivered,
        }
        # Flatten network state into the dictionary
        for key, value in self.network_state.to_dict().items():
            result[f"ns_{key}"] = value
        return result

    @classmethod
    def from_dict(cls, d: Dict) -> "TransmissionOutcome":
        """Create TransmissionOutcome from dictionary."""
        # Extract network state from flattened keys
        ns_dict = {}
        for key, value in d.items():
            if key.startswith("ns_"):
                ns_dict[key[3:]] = value
        network_state = NetworkState.from_dict(ns_dict)

        return cls(
            timestamp_ms=int(d["timestamp_ms"]),
            rat_used=RATType.from_string(d["rat_used"]),
            packet_size_bytes=int(d["packet_size_bytes"]),
            actual_latency_ms=float(d["actual_latency_ms"]),
            delivered=bool(d["delivered"]),
            network_state=network_state,
        )
