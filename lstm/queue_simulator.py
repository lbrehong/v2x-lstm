"""
SimPy-based Packet Queue Simulator with DTMC Packet Sizing.

This module implements a discrete event simulation for packet transmission
over multiple RATs, with adaptive packet sizing based on a Discrete Time
Markov Chain (DTMC) model.

Architecture:
    [App Layer] --> [Packet Queue] --> [DTMC Sizer] --> [RAT Selector] --> [TX]
                          ^                                                   |
                          +------------------ PDR Feedback -------------------+

Dependencies:
    pip install simpy

Usage:
    # Standalone simulation with pre-computed RAT decisions
    python queue_simulator.py --input rat_decisions.csv --output sim_results.csv

    # Integrated with live RAT selection API
    from queue_simulator import IntegratedQueueSimulator
    sim = IntegratedQueueSimulator(rat_api, network_data)
    results = sim.run(duration=1000)
"""
import os
import random
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum
import numpy as np
import pandas as pd

try:
    import simpy
except ImportError:
    raise ImportError("SimPy is required. Install with: pip install simpy")

from api_types import (
    RATType, NetworkState, QueueContext, RATDecision,
    PacketSizeDecision, TransmissionOutcome,
)
from config import (
    OUTPUT_DIR,
    PACKET_SIZE_BOUNDS,
    DTMC_THRESHOLD_HIGH, DTMC_THRESHOLD_LOW,
    DTMC_P_UP, DTMC_P_DOWN,
)


# =============================================================================
# DTMC Packet Sizer
# =============================================================================

class DTMCPacketSizer:
    """
    Discrete Time Markov Chain for adaptive packet sizing.

    State transitions based on observed/predicted PDR:
    - PDR >= threshold_high: May increase packet size (p_up probability)
    - PDR <= threshold_low: May decrease packet size (p_down probability)
    - Otherwise: Stay at current size

    State Diagram:
        [s_min] <--low_pdr-- [s_1] <--low_pdr-- [s_2] ... [s_max]
           |                   |                   |          |
           +----high_pdr----->-+----high_pdr----->-+   ...   -+
    """

    def __init__(
        self,
        rat: RATType,
        num_levels: int = 8,
        threshold_high: float = DTMC_THRESHOLD_HIGH,
        threshold_low: float = DTMC_THRESHOLD_LOW,
        p_up: float = DTMC_P_UP,
        p_down: float = DTMC_P_DOWN,
    ):
        """
        Initialize DTMC packet sizer.

        Args:
            rat: RAT type (determines packet size bounds)
            num_levels: Number of discrete packet size states
            threshold_high: PDR above this may increase size
            threshold_low: PDR below this may decrease size
            p_up: Probability of increasing when PDR high
            p_down: Probability of decreasing when PDR low
        """
        self.rat = rat
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.p_up = p_up
        self.p_down = p_down

        # Get bounds for this RAT
        bounds = PACKET_SIZE_BOUNDS.get(rat.value, {"min": 100, "max": 1400})
        self.min_size = bounds["min"]
        self.max_size = bounds["max"]
        self.num_levels = num_levels

        # Create discrete size levels
        self.size_levels = np.linspace(
            self.min_size, self.max_size, num_levels, dtype=int
        )

        # Current state (start at middle)
        self.current_state = num_levels // 2

        # History for analysis
        self.history: List[Tuple[int, int, float, str]] = []  # (step, state, pdr, action)

    @property
    def current_size(self) -> int:
        """Get current packet size in bytes."""
        return int(self.size_levels[self.current_state])

    def transition(self, pdr: float, step: int = 0) -> int:
        """
        Perform DTMC transition based on PDR.

        Args:
            pdr: Observed or predicted PDR
            step: Simulation step for logging

        Returns:
            New packet size in bytes
        """
        old_state = self.current_state
        action = "stay"

        if pdr >= self.threshold_high and self.current_state < self.num_levels - 1:
            # May increase
            if random.random() < self.p_up:
                self.current_state += 1
                action = "increase"

        elif pdr <= self.threshold_low and self.current_state > 0:
            # May decrease
            if random.random() < self.p_down:
                self.current_state -= 1
                action = "decrease"

        self.history.append((step, self.current_state, pdr, action))
        return self.current_size

    def reset(self):
        """Reset to initial state."""
        self.current_state = self.num_levels // 2
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


# =============================================================================
# PDR Correction for Packet Size
# =============================================================================

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


# =============================================================================
# Simulation Metrics
# =============================================================================

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
    total_bytes_sent: int = 0
    successful_bytes: int = 0

    rat_usage: Dict[RATType, int] = field(default_factory=dict)
    rat_switches: int = 0

    mean_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    mean_queue_depth: float = 0.0

    mean_packet_size: float = 0.0
    dtmc_increases: int = 0
    dtmc_decreases: int = 0

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
            "pdr": self.pdr,
            "total_bytes_sent": self.total_bytes_sent,
            "successful_bytes": self.successful_bytes,
            "throughput_bps": self.throughput_bps,
            "rat_switches": self.rat_switches,
            "mean_latency_ms": self.mean_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "mean_queue_depth": self.mean_queue_depth,
            "mean_packet_size": self.mean_packet_size,
            "dtmc_increases": self.dtmc_increases,
            "dtmc_decreases": self.dtmc_decreases,
            **{f"rat_{rat.value}_count": count for rat, count in self.rat_usage.items()},
        }


# =============================================================================
# Queue Simulator (implements QueueSimulatorAPI interface)
# =============================================================================

class QueueSimulator:
    """
    SimPy-based queue simulator with DTMC packet sizing.

    Implements the QueueSimulatorAPI interface for integration with
    JointController and RATSelectionAPI.
    """

    def __init__(
        self,
        base_packet_size: int = 1000,
        correction_exponent: float = 0.8,
        target_latency_ms: float = 20.0,
        target_pdr: float = 0.99,
    ):
        """
        Initialize queue simulator.

        Args:
            base_packet_size: Packet size at which PDR was measured
            correction_exponent: PDR correction power-law exponent
            target_latency_ms: Application latency requirement
            target_pdr: Application PDR requirement
        """
        self.base_packet_size = base_packet_size
        self.correction_exponent = correction_exponent
        self.target_latency_ms = target_latency_ms
        self.target_pdr = target_pdr

        # DTMC sizers per RAT
        self.dtmc_sizers: Dict[RATType, DTMCPacketSizer] = {}
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            self.dtmc_sizers[rat] = DTMCPacketSizer(rat)

        # Current state
        self.current_rat: Optional[RATType] = None
        self.queue_depth = 0
        self.recent_pdrs: List[float] = []
        self.pdr_window = 20  # Window for trend calculation

        # Metrics
        self.metrics = SimulationMetrics()

    def get_queue_context(self) -> QueueContext:
        """
        Return current queue state for RAT selection.

        Implements QueueSimulatorAPI.get_queue_context()
        """
        # Calculate PDR trend
        if len(self.recent_pdrs) < 2:
            pdr_trend = 0.0
        else:
            # Simple linear trend
            recent = self.recent_pdrs[-self.pdr_window:]
            if len(recent) >= 2:
                trend = (recent[-1] - recent[0]) / len(recent)
                pdr_trend = max(-1.0, min(1.0, trend * 10))  # Scale to [-1, 1]
            else:
                pdr_trend = 0.0

        # Calculate urgency based on queue depth
        urgency = min(1.0, self.queue_depth / 20.0)

        # Get current packet size
        if self.current_rat and self.current_rat in self.dtmc_sizers:
            avg_size = self.dtmc_sizers[self.current_rat].current_size
        else:
            avg_size = self.base_packet_size

        return QueueContext(
            queue_depth=self.queue_depth,
            avg_packet_size_bytes=avg_size,
            urgency_level=urgency,
            recent_pdr_trend=pdr_trend,
            target_latency_ms=self.target_latency_ms,
            target_pdr=self.target_pdr,
        )

    def decide_packet_size(
        self,
        rat_decision: RATDecision,
        current_state: NetworkState
    ) -> PacketSizeDecision:
        """
        Size packets based on RAT selection and predictions.

        Implements QueueSimulatorAPI.decide_packet_size()
        """
        rat = rat_decision.selected_rat

        if rat == RATType.UNAVAILABLE:
            return PacketSizeDecision(
                packet_size_bytes=self.base_packet_size,
                fragment_count=1,
                priority_level=0,
                send_rate_hz=50.0,
            )

        # Get DTMC for this RAT
        dtmc = self.dtmc_sizers.get(rat)
        if not dtmc:
            return PacketSizeDecision(
                packet_size_bytes=self.base_packet_size,
                fragment_count=1,
                priority_level=0,
                send_rate_hz=50.0,
            )

        # Use predicted PDR to inform DTMC (with correction for current size)
        predicted_pdr = rat_decision.predicted_pdr
        current_size = dtmc.current_size
        corrected_pdr = correct_pdr_for_packet_size(
            predicted_pdr, self.base_packet_size, current_size, self.correction_exponent
        )

        # DTMC transition based on corrected PDR
        new_size = dtmc.transition(corrected_pdr, self.metrics.total_packets)

        # Determine priority based on urgency
        priority = min(7, int(self.get_queue_context().urgency_level * 7))

        return PacketSizeDecision(
            packet_size_bytes=new_size,
            fragment_count=1,
            priority_level=priority,
            send_rate_hz=50.0,
        )

    def update_pdr_estimate(self, outcome: TransmissionOutcome) -> None:
        """
        Update internal PDR estimates from outcome.

        Implements QueueSimulatorAPI.update_pdr_estimate()
        """
        pdr_value = 1.0 if outcome.delivered else 0.0
        self.recent_pdrs.append(pdr_value)

        # Keep window limited
        if len(self.recent_pdrs) > self.pdr_window * 2:
            self.recent_pdrs = self.recent_pdrs[-self.pdr_window:]

    def reset(self):
        """Reset simulator state."""
        for dtmc in self.dtmc_sizers.values():
            dtmc.reset()
        self.current_rat = None
        self.queue_depth = 0
        self.recent_pdrs = []
        self.metrics = SimulationMetrics()


# =============================================================================
# Integrated SimPy Simulation
# =============================================================================

class IntegratedQueueSimulator:
    """
    Full SimPy-based simulation integrating queue, DTMC, and RAT selection.

    This class runs a complete discrete event simulation where:
    1. Packets arrive according to a specified process
    2. RAT selection is performed for each packet
    3. DTMC adapts packet size based on PDR
    4. Transmission success is simulated based on corrected PDR
    """

    def __init__(
        self,
        rat_api=None,
        network_data: Optional[pd.DataFrame] = None,
        arrival_rate_hz: float = 50.0,
        base_packet_size: int = 1000,
        correction_exponent: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Initialize integrated simulator.

        Args:
            rat_api: RATSelectionAPI instance (optional, uses pre-computed if None)
            network_data: DataFrame with network states (for replay mode)
            arrival_rate_hz: Packet arrival rate
            base_packet_size: Base packet size for PDR correction
            correction_exponent: PDR correction exponent
            seed: Random seed for reproducibility
        """
        self.rat_api = rat_api
        self.network_data = network_data
        self.arrival_rate_hz = arrival_rate_hz
        self.base_packet_size = base_packet_size
        self.correction_exponent = correction_exponent

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Queue simulator component
        self.queue_sim = QueueSimulator(
            base_packet_size=base_packet_size,
            correction_exponent=correction_exponent,
        )

        # SimPy environment (created on run)
        self.env: Optional[simpy.Environment] = None

        # Current data index for replay mode
        self.data_index = 0

        # Flag to signal simulation should stop
        self._stop_simulation = False

        # Results
        self.results: List[Dict] = []

    def _get_network_state(self, timestamp: float) -> Optional[NetworkState]:
        """Get network state for current simulation time."""
        if self.network_data is None:
            return None

        if self.data_index >= len(self.network_data):
            return None

        row = self.network_data.iloc[self.data_index]
        self.data_index += 1

        # Convert row to NetworkState
        try:
            return NetworkState(
                timestamp_ms=int(timestamp * 1000),
                latitude=float(row.get("latitude", row.get("tx_latitude", 43.56))),
                longitude=float(row.get("longitude", row.get("tx_longitude", 1.47))),
                dsrc_latency_ms=self._safe_float(row.get("dsrc_latency_ms", row.get("latency_ms_dsrc"))),
                dsrc_pdr=self._safe_float(row.get("dsrc_pdr", row.get("pdr_dsrc"))),
                pc5_latency_ms=self._safe_float(row.get("pc5_latency_ms", row.get("latency_ms_pc5"))),
                pc5_pdr=self._safe_float(row.get("pc5_pdr", row.get("pdr_pc5"))),
                fiveg_latency_ms=self._safe_float(row.get("fiveg_latency_ms", row.get("latency_ms_5g", row.get("latency_ms")))),
                fiveg_pdr=self._safe_float(row.get("fiveg_pdr", row.get("pdr_5g", row.get("pdr")))),
                fiveg_sinr=self._safe_float(row.get("fiveg_sinr", row.get("sinr"))),
                fiveg_rsrp=self._safe_float(row.get("fiveg_rsrp", row.get("rsrp"))),
            )
        except (KeyError, ValueError):
            return None

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float."""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _get_rat_decision(self, state: NetworkState) -> RATDecision:
        """Get RAT decision from API or pre-computed data."""
        if self.rat_api is not None:
            queue_context = self.queue_sim.get_queue_context()
            return self.rat_api.select_rat(state, queue_context)

        # Fallback: use data from DataFrame if available
        if self.network_data is not None and self.data_index > 0:
            row = self.network_data.iloc[self.data_index - 1]
            if "selected_rat" in row:
                try:
                    rat = RATType.from_string(str(row["selected_rat"]))
                    return RATDecision(
                        selected_rat=rat,
                        confidence=float(row.get("confidence", 1.0)),
                        predicted_latency_ms=float(row.get("pred_latency", 10.0)),
                        predicted_pdr=float(row.get("pred_pdr", 0.95)),
                        all_predictions={},
                        model_type="precomputed",
                    )
                except (KeyError, ValueError):
                    pass

        # Default fallback
        return RATDecision(
            selected_rat=RATType.FiveG,
            confidence=0.5,
            predicted_latency_ms=15.0,
            predicted_pdr=0.95,
            all_predictions={},
            model_type="fallback",
        )

    def _simulate_transmission(
        self,
        rat: RATType,
        packet_size: int,
        predicted_pdr: float,
    ) -> Tuple[bool, float]:
        """
        Simulate transmission outcome.

        Args:
            rat: Selected RAT
            packet_size: Packet size in bytes
            predicted_pdr: PDR prediction (at base packet size)

        Returns:
            (success, latency_ms)
        """
        # Apply packet size correction
        corrected_pdr = correct_pdr_for_packet_size(
            predicted_pdr,
            self.base_packet_size,
            packet_size,
            self.correction_exponent,
        )

        # Simulate success/failure
        success = random.random() < corrected_pdr

        # Simulate latency (simplified model)
        base_latency = {
            RATType.DSRC: 5.0,
            RATType.PC5: 8.0,
            RATType.FiveG: 15.0,
        }.get(rat, 10.0)

        # Add variability and size-dependent component
        size_factor = packet_size / self.base_packet_size
        latency = base_latency * (0.8 + 0.4 * random.random()) * (0.9 + 0.2 * size_factor)

        return success, latency

    def _packet_generator(self, env: simpy.Environment, queue: simpy.Store):
        """SimPy process: Generate packets at specified rate."""
        interval = 1.0 / self.arrival_rate_hz

        while not self._stop_simulation:
            yield env.timeout(interval)
            if self._stop_simulation:
                break
            packet = {"arrival_time": env.now, "id": self.queue_sim.metrics.total_packets}
            yield queue.put(packet)
            self.queue_sim.queue_depth += 1

    def _packet_processor(self, env: simpy.Environment, queue: simpy.Store):
        """SimPy process: Process packets from queue."""
        previous_rat = None

        while True:
            # Wait for packet
            packet = yield queue.get()
            self.queue_sim.queue_depth -= 1

            # Get network state
            state = self._get_network_state(env.now)
            if state is None:
                # No more data, signal stop and exit
                self._stop_simulation = True
                break

            # Get RAT decision
            rat_decision = self._get_rat_decision(state)
            selected_rat = rat_decision.selected_rat

            # Track RAT switches
            if previous_rat is not None and selected_rat != previous_rat:
                self.queue_sim.metrics.rat_switches += 1
            previous_rat = selected_rat

            # Get packet size decision
            packet_decision = self.queue_sim.decide_packet_size(rat_decision, state)
            packet_size = packet_decision.packet_size_bytes

            # Simulate transmission
            success, latency = self._simulate_transmission(
                selected_rat,
                packet_size,
                rat_decision.predicted_pdr,
            )

            # Create outcome and update PDR estimate
            outcome = TransmissionOutcome(
                timestamp_ms=int(env.now * 1000),
                rat_used=selected_rat,
                packet_size_bytes=packet_size,
                actual_latency_ms=latency,
                delivered=success,
                network_state=state,
            )
            self.queue_sim.update_pdr_estimate(outcome)

            # Update metrics
            metrics = self.queue_sim.metrics
            metrics.total_packets += 1
            metrics.total_bytes_sent += packet_size

            if success:
                metrics.successful_packets += 1
                metrics.successful_bytes += packet_size
            else:
                metrics.failed_packets += 1

            # Track RAT usage
            if selected_rat not in metrics.rat_usage:
                metrics.rat_usage[selected_rat] = 0
            metrics.rat_usage[selected_rat] += 1

            # Get DTMC state for this RAT
            dtmc = self.queue_sim.dtmc_sizers.get(selected_rat)
            dtmc_state = dtmc.current_state if dtmc else 0

            # Record transmission
            corrected_pdr = correct_pdr_for_packet_size(
                rat_decision.predicted_pdr,
                self.base_packet_size,
                packet_size,
                self.correction_exponent,
            )

            record = TransmissionRecord(
                timestamp=env.now,
                rat=selected_rat,
                packet_size=packet_size,
                predicted_pdr=rat_decision.predicted_pdr,
                corrected_pdr=corrected_pdr,
                success=success,
                latency_ms=latency,
                queue_depth=self.queue_sim.queue_depth,
                dtmc_state=dtmc_state,
            )
            metrics.records.append(record)

            # Store result for export
            self.results.append({
                "timestamp": env.now,
                "packet_id": packet["id"],
                "rat": selected_rat.value,
                "packet_size": packet_size,
                "predicted_pdr": rat_decision.predicted_pdr,
                "corrected_pdr": corrected_pdr,
                "success": success,
                "latency_ms": latency,
                "queue_depth": self.queue_sim.queue_depth,
                "dtmc_state": dtmc_state,
                "latitude": state.latitude,
                "longitude": state.longitude,
            })

            # Simulate transmission delay
            yield env.timeout(latency / 1000.0)

    def run(
        self,
        duration: Optional[float] = None,
        max_packets: Optional[int] = None,
    ) -> SimulationMetrics:
        """
        Run the simulation.

        Args:
            duration: Simulation duration in seconds (optional)
            max_packets: Maximum packets to process (optional)

        Returns:
            SimulationMetrics with results
        """
        # Reset state
        self.queue_sim.reset()
        self.data_index = 0
        self.results = []
        self._stop_simulation = False

        # Determine run length
        if duration is None and max_packets is None:
            if self.network_data is not None:
                max_packets = len(self.network_data)
            else:
                duration = 100.0

        if max_packets is not None:
            # Calculate duration based on packet count and arrival rate
            # Add buffer for processing time
            duration = max_packets / self.arrival_rate_hz * 2.0

        # Create SimPy environment
        self.env = simpy.Environment()
        queue = simpy.Store(self.env)

        # Start processes
        self.env.process(self._packet_generator(self.env, queue))
        self.env.process(self._packet_processor(self.env, queue))

        # Run simulation
        if duration is not None:
            self.env.run(until=duration)
        else:
            # Run until data exhausted (processor will break)
            try:
                self.env.run()
            except simpy.core.StopSimulation:
                pass

        # Finalize metrics
        metrics = self.queue_sim.metrics
        if metrics.records:
            latencies = [r.latency_ms for r in metrics.records]
            metrics.mean_latency_ms = np.mean(latencies)
            metrics.max_latency_ms = max(latencies)
            metrics.mean_queue_depth = np.mean([r.queue_depth for r in metrics.records])
            metrics.mean_packet_size = np.mean([r.packet_size for r in metrics.records])

        # DTMC statistics
        for dtmc in self.queue_sim.dtmc_sizers.values():
            stats = dtmc.get_statistics()
            metrics.dtmc_increases += stats.get("increase_count", 0)
            metrics.dtmc_decreases += stats.get("decrease_count", 0)

        return metrics

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get simulation results as DataFrame."""
        return pd.DataFrame(self.results)

    def save_results(self, output_path: str):
        """Save results to CSV."""
        df = self.get_results_dataframe()
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


# =============================================================================
# Standalone Mode: Process Pre-computed RAT Decisions
# =============================================================================

def run_standalone(
    input_csv: str,
    output_csv: str,
    arrival_rate_hz: float = 50.0,
    base_packet_size: int = 1000,
    correction_exponent: float = 0.8,
    seed: Optional[int] = None,
):
    """
    Run simulation using pre-computed RAT decisions from CSV.

    This mode is for Phase 1 integration where RAT selection was run
    separately and results are in rat_decisions.csv.

    Args:
        input_csv: Path to rat_decisions.csv from RAT selector
        output_csv: Path for simulation results
        arrival_rate_hz: Packet arrival rate
        base_packet_size: Base packet size for PDR correction
        correction_exponent: PDR correction exponent
        seed: Random seed
    """
    print(f"Loading RAT decisions from {input_csv}")
    data = pd.read_csv(input_csv)

    print(f"Running simulation with {len(data)} packets")
    simulator = IntegratedQueueSimulator(
        rat_api=None,  # Use pre-computed decisions
        network_data=data,
        arrival_rate_hz=arrival_rate_hz,
        base_packet_size=base_packet_size,
        correction_exponent=correction_exponent,
        seed=seed,
    )

    metrics = simulator.run(max_packets=len(data))

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Total packets:     {metrics.total_packets}")
    print(f"Successful:        {metrics.successful_packets}")
    print(f"Failed:            {metrics.failed_packets}")
    print(f"PDR:               {metrics.pdr:.4f}")
    print(f"Throughput:        {metrics.throughput_bps:.2f} Bps")
    print(f"Mean latency:      {metrics.mean_latency_ms:.2f} ms")
    print(f"Max latency:       {metrics.max_latency_ms:.2f} ms")
    print(f"RAT switches:      {metrics.rat_switches}")
    print(f"Mean packet size:  {metrics.mean_packet_size:.0f} bytes")
    print(f"DTMC increases:    {metrics.dtmc_increases}")
    print(f"DTMC decreases:    {metrics.dtmc_decreases}")
    print("\nRAT usage:")
    for rat, count in metrics.rat_usage.items():
        pct = count / metrics.total_packets * 100
        print(f"  {rat.value}: {count} ({pct:.1f}%)")
    print("=" * 60)

    # Save results
    simulator.save_results(output_csv)

    # Also save metrics summary
    metrics_path = output_csv.replace(".csv", "_metrics.csv")
    pd.DataFrame([metrics.to_dict()]).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    return metrics


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SimPy-based Queue Simulator with DTMC Packet Sizing"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input CSV with RAT decisions or network data"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output CSV for simulation results"
    )
    parser.add_argument(
        "--arrival-rate", type=float, default=50.0,
        help="Packet arrival rate in Hz (default: 50)"
    )
    parser.add_argument(
        "--base-size", type=int, default=1000,
        help="Base packet size for PDR correction (default: 1000)"
    )
    parser.add_argument(
        "--correction-exp", type=float, default=0.8,
        help="PDR correction exponent (default: 0.8)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    output = args.output or os.path.join(OUTPUT_DIR, "sim_results.csv")

    run_standalone(
        input_csv=args.input,
        output_csv=output,
        arrival_rate_hz=args.arrival_rate,
        base_packet_size=args.base_size,
        correction_exponent=args.correction_exp,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
