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
    python -m queuesim.queue_simulator --input rat_decisions.csv --output sim_results.csv

    # Integrated with live RAT selection API
    from queuesim.queue_simulator import IntegratedQueueSimulator
    sim = IntegratedQueueSimulator(rat_api, network_data)
    results = sim.run(duration=1000)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import argparse
from typing import Optional, List, Dict, Tuple

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
from config import OUTPUT_DIR, TX_INTERVAL_MS

# Import from sub-modules
from queuesim.phy_layer import (
    get_phy_config,
    calculate_subframe_capacity_bits,
    calculate_tx_capacity_bytes,
    calculate_queue_capacity_bytes,
    can_transmit,
    get_queue_capacity_packets,
    get_max_packet_size,
    calculate_tx_time_ms,
    get_base_latency_ms,
)
from queuesim.dtmc_sizer import DTMCPacketSizer, correct_pdr_for_packet_size
from queuesim.sim_types import TransmissionRecord, SimulationMetrics


# =============================================================================
# Queue Simulator (implements QueueSimulatorAPI interface)
# =============================================================================

class QueueSimulator:
    """
    SimPy-based queue simulator with DTMC packet sizing.

    Implements the QueueSimulatorAPI interface for integration with
    JointController and RATSelectionAPI.

    Includes PHY-layer constraints:
    - Bounded queue capacity based on RAT characteristics
    - Maximum packet size enforcement per TX interval
    - Realistic transmission time calculation
    """

    def __init__(
        self,
        base_packet_size: int = 1000,
        correction_exponent: float = 0.8,
        target_latency_ms: float = 20.0,
        target_pdr: float = 0.99,
        enforce_phy_limits: bool = True,
    ):
        """
        Initialize queue simulator.

        Args:
            base_packet_size: Packet size at which PDR was measured
            correction_exponent: PDR correction power-law exponent
            target_latency_ms: Application latency requirement
            target_pdr: Application PDR requirement
            enforce_phy_limits: Whether to enforce PHY-layer constraints
        """
        self.base_packet_size = base_packet_size
        self.correction_exponent = correction_exponent
        self.target_latency_ms = target_latency_ms
        self.target_pdr = target_pdr
        self.enforce_phy_limits = enforce_phy_limits

        # DTMC sizers per RAT
        self.dtmc_sizers: Dict[RATType, DTMCPacketSizer] = {}
        for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
            self.dtmc_sizers[rat] = DTMCPacketSizer(rat)

        # Current state
        self.current_rat: Optional[RATType] = None
        self.queue_depth = 0
        self.queue_bytes = 0  # Track bytes in queue
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

    def get_queue_capacity(self, rat: Optional[RATType] = None) -> int:
        """
        Get queue capacity in packets for current or specified RAT.

        Args:
            rat: RAT type (uses current_rat if None)

        Returns:
            Queue capacity in packets
        """
        target_rat = rat or self.current_rat or RATType.FiveG
        dtmc = self.dtmc_sizers.get(target_rat)
        packet_size = dtmc.current_size if dtmc else self.base_packet_size
        return get_queue_capacity_packets(target_rat, packet_size)

    def can_enqueue(self, packet_size: int, rat: Optional[RATType] = None) -> bool:
        """
        Check if a packet can be added to the queue without overflow.

        Args:
            packet_size: Size of packet to enqueue
            rat: RAT type (uses current_rat if None)

        Returns:
            True if packet can be enqueued
        """
        if not self.enforce_phy_limits:
            return True

        target_rat = rat or self.current_rat or RATType.FiveG
        capacity_bytes = calculate_queue_capacity_bytes(target_rat, TX_INTERVAL_MS)
        return (self.queue_bytes + packet_size) <= capacity_bytes

    def enqueue(self, packet_size: int) -> bool:
        """
        Add a packet to the queue.

        Args:
            packet_size: Size of packet in bytes

        Returns:
            True if packet was enqueued, False if dropped
        """
        if not self.can_enqueue(packet_size):
            self.metrics.dropped_packets += 1
            return False

        self.queue_depth += 1
        self.queue_bytes += packet_size
        self.metrics.max_queue_depth = max(
            self.metrics.max_queue_depth, self.queue_depth
        )
        return True

    def dequeue(self, packet_size: int) -> None:
        """
        Remove a packet from the queue.

        Args:
            packet_size: Size of packet being dequeued
        """
        self.queue_depth = max(0, self.queue_depth - 1)
        self.queue_bytes = max(0, self.queue_bytes - packet_size)

    def decide_packet_size(
        self,
        rat_decision: RATDecision,
        current_time: float = 0.0
    ) -> PacketSizeDecision:
        """
        Size packets based on RAT selection and moving window PDR.

        DTMC uses actual PDR from past transmissions (moving window),
        falling back to predicted PDR if insufficient history.

        Implements QueueSimulatorAPI.decide_packet_size()
        """
        rat = rat_decision.selected_rat

        if rat == RATType.UNAVAILABLE:
            return PacketSizeDecision(
                packet_size_bytes=self.base_packet_size,
                fragment_count=1,
                priority_level=0,
                send_rate_hz=1000 / TX_INTERVAL_MS,
            )

        # Get DTMC for this RAT
        dtmc = self.dtmc_sizers.get(rat)
        if not dtmc:
            return PacketSizeDecision(
                packet_size_bytes=self.base_packet_size,
                fragment_count=1,
                priority_level=0,
                send_rate_hz=1000 / TX_INTERVAL_MS,
            )

        # Get PDR from moving window (actual outcomes), fallback to prediction
        window_pdr = dtmc.get_window_pdr(current_time)
        if window_pdr is not None:
            # Use actual PDR from recent transmissions
            pdr_for_decision = window_pdr
        else:
            # Not enough history - use predicted PDR with size correction
            predicted_pdr = rat_decision.predicted_pdr
            current_size = dtmc.current_size
            pdr_for_decision = correct_pdr_for_packet_size(
                predicted_pdr, self.base_packet_size, current_size, self.correction_exponent
            )

        # DTMC transition based on PDR
        new_size = dtmc.transition(pdr_for_decision, self.metrics.total_packets)

        # Enforce PHY-layer max packet size
        if self.enforce_phy_limits:
            max_size = get_max_packet_size(rat)
            if new_size > max_size:
                new_size = max_size
                self.metrics.capacity_limited_count += 1

        # Determine priority based on urgency
        priority = min(7, int(self.get_queue_context().urgency_level * 7))

        return PacketSizeDecision(
            packet_size_bytes=new_size,
            fragment_count=1,
            priority_level=priority,
            send_rate_hz=1000 / TX_INTERVAL_MS,
        )

    def update_pdr_estimate(self, outcome: TransmissionOutcome) -> None:
        """
        Update internal PDR estimates from outcome.

        Records the outcome to the DTMC's moving window for the RAT used.

        Implements QueueSimulatorAPI.update_pdr_estimate()
        """
        pdr_value = 1.0 if outcome.delivered else 0.0
        self.recent_pdrs.append(pdr_value)

        # Keep window limited
        if len(self.recent_pdrs) > self.pdr_window * 2:
            self.recent_pdrs = self.recent_pdrs[-self.pdr_window:]

        # Record to DTMC for the RAT used
        dtmc = self.dtmc_sizers.get(outcome.rat_used)
        if dtmc:
            timestamp_sec = outcome.timestamp_ms / 1000.0
            dtmc.record_outcome(timestamp_sec, outcome.delivered)

    def reset(self):
        """Reset simulator state."""
        for dtmc in self.dtmc_sizers.values():
            dtmc.reset()
        self.current_rat = None
        self.queue_depth = 0
        self.queue_bytes = 0
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
        arrival_rate_hz: float = 1000 / TX_INTERVAL_MS,
        base_packet_size: int = 1000,
        correction_exponent: float = 0.8,
        seed: Optional[int] = None,
        enforce_phy_limits: bool = True,
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
            enforce_phy_limits: Whether to enforce PHY-layer constraints
        """
        self.rat_api = rat_api
        self.network_data = network_data
        self.arrival_rate_hz = arrival_rate_hz
        self.base_packet_size = base_packet_size
        self.correction_exponent = correction_exponent
        self.enforce_phy_limits = enforce_phy_limits

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Queue simulator component
        self.queue_sim = QueueSimulator(
            base_packet_size=base_packet_size,
            correction_exponent=correction_exponent,
            enforce_phy_limits=enforce_phy_limits,
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
    ) -> Tuple[bool, float, float]:
        """
        Simulate transmission outcome with PHY-layer modeling.

        Args:
            rat: Selected RAT
            packet_size: Packet size in bytes
            predicted_pdr: PDR prediction (at base packet size)

        Returns:
            (success, latency_ms, tx_time_ms)
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

        # Calculate TX time using PHY model
        tx_time_ms = calculate_tx_time_ms(packet_size, rat)

        # Get base latency from PHY config
        base_latency = get_base_latency_ms(rat)

        # Add variability (jitter) - more for contention-based access (DSRC)
        if rat == RATType.DSRC:
            jitter_factor = 0.5 + random.random()  # Higher variance
        else:
            jitter_factor = 0.8 + 0.4 * random.random()  # Lower variance

        # Size-dependent component
        size_factor = packet_size / self.base_packet_size

        # Total latency = base + tx_time + size adjustment + jitter
        latency = (base_latency + tx_time_ms) * jitter_factor * (0.95 + 0.1 * size_factor)

        return success, latency, tx_time_ms

    def _packet_generator(self, env: simpy.Environment, queue: simpy.Store):
        """SimPy process: Generate packets at specified rate with bounded queue."""
        interval = 1.0 / self.arrival_rate_hz
        packet_id = 0

        while not self._stop_simulation:
            yield env.timeout(interval)
            if self._stop_simulation:
                break

            # Estimate packet size for queue capacity check
            estimated_size = self.base_packet_size

            # Check if packet can be enqueued (bounded queue)
            if self.enforce_phy_limits and not self.queue_sim.can_enqueue(estimated_size):
                self.queue_sim.metrics.dropped_packets += 1
                packet_id += 1
                continue

            packet = {
                "arrival_time": env.now,
                "id": packet_id,
                "estimated_size": estimated_size,
            }
            yield queue.put(packet)
            self.queue_sim.queue_depth += 1
            self.queue_sim.queue_bytes += estimated_size
            self.queue_sim.metrics.max_queue_depth = max(
                self.queue_sim.metrics.max_queue_depth,
                self.queue_sim.queue_depth,
            )
            packet_id += 1

    def _packet_processor(self, env: simpy.Environment, queue: simpy.Store):
        """SimPy process: Process packets from queue with PHY-layer modeling."""
        previous_rat = None
        tx_times: List[float] = []

        while True:
            # Wait for packet
            packet = yield queue.get()
            estimated_size = packet.get("estimated_size", self.base_packet_size)
            self.queue_sim.queue_depth = max(0, self.queue_sim.queue_depth - 1)
            self.queue_sim.queue_bytes = max(0, self.queue_sim.queue_bytes - estimated_size)

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

            # Get packet size decision (uses moving window PDR from actual outcomes)
            packet_decision = self.queue_sim.decide_packet_size(rat_decision, env.now)
            packet_size = packet_decision.packet_size_bytes

            # Simulate transmission with PHY-layer model
            success, latency, tx_time = self._simulate_transmission(
                selected_rat,
                packet_size,
                rat_decision.predicted_pdr,
            )
            tx_times.append(tx_time)

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
                "tx_time_ms": tx_time,
                "queue_depth": self.queue_sim.queue_depth,
                "dtmc_state": dtmc_state,
                "latitude": state.latitude,
                "longitude": state.longitude,
            })

            # Simulate transmission delay
            yield env.timeout(latency / 1000.0)

        # Update mean TX time
        if tx_times:
            self.queue_sim.metrics.mean_tx_time_ms = float(np.mean(tx_times))

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
    arrival_rate_hz: float = 1000 / TX_INTERVAL_MS,
    base_packet_size: int = 1000,
    correction_exponent: float = 0.8,
    seed: Optional[int] = None,
    enforce_phy_limits: bool = True,
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
        enforce_phy_limits: Whether to enforce PHY-layer constraints
    """
    print(f"Loading RAT decisions from {input_csv}")
    data = pd.read_csv(input_csv)

    print(f"Running simulation with {len(data)} packets (PHY limits: {enforce_phy_limits})")
    simulator = IntegratedQueueSimulator(
        rat_api=None,  # Use pre-computed decisions
        network_data=data,
        arrival_rate_hz=arrival_rate_hz,
        base_packet_size=base_packet_size,
        correction_exponent=correction_exponent,
        seed=seed,
        enforce_phy_limits=enforce_phy_limits,
    )

    metrics = simulator.run(max_packets=len(data))

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Total packets:     {metrics.total_packets}")
    print(f"Successful:        {metrics.successful_packets}")
    print(f"Failed:            {metrics.failed_packets}")
    print(f"Dropped:           {metrics.dropped_packets}")
    print(f"PDR:               {metrics.pdr:.4f}")
    print(f"Throughput:        {metrics.throughput_bps:.2f} Bps")
    print(f"Mean latency:      {metrics.mean_latency_ms:.2f} ms")
    print(f"Max latency:       {metrics.max_latency_ms:.2f} ms")
    print(f"Mean TX time:      {metrics.mean_tx_time_ms:.2f} ms")
    print(f"RAT switches:      {metrics.rat_switches}")
    print(f"Mean packet size:  {metrics.mean_packet_size:.0f} bytes")
    print(f"Max queue depth:   {metrics.max_queue_depth}")
    print(f"Capacity limited:  {metrics.capacity_limited_count}")
    print(f"DTMC increases:    {metrics.dtmc_increases}")
    print(f"DTMC decreases:    {metrics.dtmc_decreases}")
    print("\nRAT usage:")
    for rat, count in metrics.rat_usage.items():
        pct = count / metrics.total_packets * 100 if metrics.total_packets > 0 else 0
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
        "--arrival-rate", type=float, default=10.0,
        help="Packet arrival rate in Hz (default: 10)"
    )
    parser.add_argument(
        "--base-size", type=int, default=1024,
        help="Base packet size for PDR correction (default: 1024)"
    )
    parser.add_argument(
        "--correction-exp", type=float, default=0.8,
        help="PDR correction exponent (default: 0.8)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-phy-limits", action="store_true",
        help="Disable PHY-layer constraints (unbounded queue, no TX limits)"
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
        enforce_phy_limits=not args.no_phy_limits,
    )


if __name__ == "__main__":
    main()
