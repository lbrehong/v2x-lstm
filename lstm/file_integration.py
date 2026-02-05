"""
File-based integration for RAT Selection + Queue Simulator.

This module provides file-based integration methods since the queue simulator
cannot make API calls but can read/write files.

Two modes are supported:
- Phase 1 (Batch Sequential): RAT selection runs first on entire dataset,
  then queue simulator processes the output.
- Phase 2 (Interleaved): Row-by-row file exchange for coupling study.
"""
import os
import time
from typing import Optional, List, Dict, Callable
import pandas as pd

from api_types import (
    RATType, NetworkState, QueueContext, RATDecision,
    PacketSizeDecision, TransmissionOutcome,
)
from api import RATSelectionAPI
from config import OUTPUT_DIR


# =============================================================================
# Phase 1: Batch Sequential Processing
# =============================================================================

def process_batch(
    input_csv: str,
    output_csv: str,
    model_type: str = "lstm",
    include_all_predictions: bool = True
) -> pd.DataFrame:
    """
    Process entire dataset and output RAT decisions to CSV.

    This is the recommended starting point for Phase 1 batch sequential
    processing. RAT selection runs once on all data, producing a CSV
    that the queue simulator can then process.

    Args:
        input_csv: Path to input CSV with matched data
        output_csv: Path for output rat_decisions.csv
        model_type: RNN model type ('lstm', 'gru', 'rnn')
        include_all_predictions: Whether to include predictions for all RATs

    Returns:
        DataFrame with RAT decisions

    Output CSV Schema:
        timestamp_ms, latitude, longitude, selected_rat,
        pred_latency_5g, pred_pdr_5g,
        pred_latency_pc5, pred_pdr_pc5,
        pred_latency_dsrc, pred_pdr_dsrc,
        confidence, recommended_max_packet_size
    """
    # Load input data
    df = pd.read_csv(input_csv)

    # Initialize API
    api = RATSelectionAPI(model_type=model_type)

    # Process each row
    results = []
    for idx, row in df.iterrows():
        # Build NetworkState from row
        state = _row_to_network_state(row)

        # Get RAT decision
        decision = api.select_rat(state)

        # Build result row
        result = {
            "timestamp_ms": state.timestamp_ms,
            "latitude": state.latitude,
            "longitude": state.longitude,
            "selected_rat": decision.selected_rat.value,
            "pred_latency": decision.predicted_latency_ms,
            "pred_pdr": decision.predicted_pdr,
            "confidence": decision.confidence,
        }

        if include_all_predictions:
            for rat, (lat, pdr) in decision.all_predictions.items():
                result[f"pred_latency_{rat.value}"] = lat
                result[f"pred_pdr_{rat.value}"] = pdr

        if decision.recommended_max_packet_size is not None:
            result["recommended_max_packet_size"] = decision.recommended_max_packet_size

        results.append(result)

        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows")

    # Create output DataFrame
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"RAT decisions saved to {output_csv}")

    return result_df


def _row_to_network_state(row: pd.Series) -> NetworkState:
    """
    Convert DataFrame row to NetworkState.

    Handles various column naming conventions from different data sources.
    """
    # Try different timestamp column names
    timestamp = 0
    for col in ["timestamp_ms", "timestamp", "time_ms"]:
        if col in row and pd.notna(row[col]):
            timestamp = int(row[col])
            break

    # GPS coordinates
    lat = row.get("tx_latitude", row.get("latitude", 0.0))
    lon = row.get("tx_longitude", row.get("longitude", 0.0))

    return NetworkState(
        timestamp_ms=timestamp,
        latitude=float(lat),
        longitude=float(lon),
        # DSRC measurements
        dsrc_latency_ms=_safe_float(row.get("latency_ms_dsrc", row.get("dsrc_latency_ms"))),
        dsrc_pdr=_safe_float(row.get("pdr_dsrc", row.get("dsrc_pdr"))),
        dsrc_rsrp_1=_safe_float(row.get("rsrp_1", row.get("dsrc_rsrp_1"))),
        dsrc_rsrp_2=_safe_float(row.get("rsrp_2", row.get("dsrc_rsrp_2"))),
        # PC5 measurements
        pc5_latency_ms=_safe_float(row.get("latency_ms_pc5", row.get("pc5_latency_ms"))),
        pc5_pdr=_safe_float(row.get("pdr_pc5", row.get("pc5_pdr"))),
        # 5G measurements
        fiveg_latency_ms=_safe_float(row.get("latency_ms_5g", row.get("fiveg_latency_ms", row.get("latency_ms")))),
        fiveg_pdr=_safe_float(row.get("pdr_5g", row.get("fiveg_pdr", row.get("pdr")))),
        fiveg_sinr=_safe_float(row.get("sinr", row.get("fiveg_sinr"))),
        fiveg_rsrp=_safe_float(row.get("rsrp", row.get("fiveg_rsrp"))),
    )


def _safe_float(value: any) -> Optional[float]:
    """
    Safely convert value to float, returning None for invalid values.

    Args:
        value: Value to convert (can be int, float, str, None, NaN)

    Returns:
        Float value, or None if conversion fails or value is None/NaN
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


# =============================================================================
# Phase 2: Interleaved File Exchange
# =============================================================================

class FileExchangeCoordinator:
    """
    Coordinates interleaved file exchange between RAT selector and queue simulator.

    This class manages the Phase 2 integration where both systems interact
    per-timestep through shared files.

    Exchange Directory Structure:
        exchange/
        ├── network_state.csv      # Current network measurements
        ├── queue_context.csv      # Written by queue simulator
        ├── rat_decision.csv       # Written by RAT selector
        └── sync.txt               # Coordination file

    Protocol:
        For each row i in dataset:
        1. Coordinator writes network_state.csv (row i data)
        2. Queue simulator reads network_state.csv, writes queue_context.csv
        3. RAT selector reads both, writes rat_decision.csv
        4. Queue simulator reads rat_decision.csv, transitions DTMC
        5. Coordinator increments step counter
    """

    def __init__(
        self,
        exchange_dir: str,
        model_type: str = "lstm",
        poll_interval_ms: int = 100,
        timeout_ms: int = 5000
    ):
        """
        Initialize the file exchange coordinator.

        Args:
            exchange_dir: Directory for file exchange
            model_type: RNN model type
            poll_interval_ms: Polling interval when waiting for files
            timeout_ms: Maximum wait time for file updates
        """
        self.exchange_dir = exchange_dir
        self.model_type = model_type
        self.poll_interval = poll_interval_ms / 1000.0
        self.timeout = timeout_ms / 1000.0

        # File paths
        self.network_state_path = os.path.join(exchange_dir, "network_state.csv")
        self.queue_context_path = os.path.join(exchange_dir, "queue_context.csv")
        self.rat_decision_path = os.path.join(exchange_dir, "rat_decision.csv")
        self.sync_path = os.path.join(exchange_dir, "sync.txt")

        # Initialize API
        self.api = RATSelectionAPI(model_type=model_type)

        # Ensure exchange directory exists
        os.makedirs(exchange_dir, exist_ok=True)

    def run_rat_selector_loop(self, on_decision: Optional[Callable[[int, RATDecision], None]] = None):
        """
        Run the RAT selector in a loop, responding to network state updates.

        This is the main loop for the RAT selector side of the file exchange.
        It watches for network_state.csv updates and writes rat_decision.csv.

        Args:
            on_decision: Optional callback called after each decision (step, decision)
        """
        last_step = -1
        print(f"RAT selector starting, watching {self.exchange_dir}")

        while True:
            # Read sync file to get current step
            step, phase = self._read_sync()

            if step is None:
                time.sleep(self.poll_interval)
                continue

            # Check if we should make a decision
            if phase == "rat_decide" and step > last_step:
                # Read network state
                state = self._read_network_state()
                if state is None:
                    continue

                # Read queue context (optional)
                queue_context = self._read_queue_context()

                # Make decision
                decision = self.api.select_rat(state, queue_context)

                # Write decision
                self._write_rat_decision(decision)

                if on_decision:
                    on_decision(step, decision)

                last_step = step
                print(f"Step {step}: Selected {decision.selected_rat.value}")

            elif phase == "done":
                print("Exchange complete")
                break

            time.sleep(self.poll_interval)

    def run_coordinator(
        self,
        input_csv: str,
        output_log: str,
        queue_write_callback: Optional[Callable[[int, NetworkState], None]] = None,
        dtmc_step_callback: Optional[Callable[[int, RATDecision], None]] = None
    ):
        """
        Run the coordinator that drives the interleaved exchange.

        This processes each row in the input data, coordinating between
        the RAT selector and queue simulator.

        Args:
            input_csv: Path to input data CSV
            output_log: Path for combined output log
            queue_write_callback: Called when queue sim should write context
            dtmc_step_callback: Called when queue sim should process decision
        """
        df = pd.read_csv(input_csv)
        results = []

        for i, row in df.iterrows():
            state = _row_to_network_state(row)

            # Step 1: Write network state
            self._write_network_state(state)

            # Step 2: Signal queue simulator to write context
            self._write_sync(i, "queue_write")
            if queue_write_callback:
                queue_write_callback(i, state)

            # Wait for queue_context.csv (with timeout)
            self._wait_for_file_update(self.queue_context_path)

            # Step 3: Signal RAT selector to make decision
            self._write_sync(i, "rat_decide")

            # Wait for rat_decision.csv
            if not self._wait_for_file_update(self.rat_decision_path):
                print(f"Timeout waiting for RAT decision at step {i}")
                continue

            # Read the decision
            decision = self._read_rat_decision()

            # Step 4: Signal queue simulator to read decision and transition
            self._write_sync(i, "dtmc_step")
            if dtmc_step_callback:
                dtmc_step_callback(i, decision)

            # Step 5: Log result
            result = {
                "step": i,
                "timestamp_ms": state.timestamp_ms,
                "latitude": state.latitude,
                "longitude": state.longitude,
                "selected_rat": decision.selected_rat.value if decision else "NaN",
            }
            if decision:
                result.update({
                    "pred_latency": decision.predicted_latency_ms,
                    "pred_pdr": decision.predicted_pdr,
                    "confidence": decision.confidence,
                })
            results.append(result)

            # Progress
            if (i + 1) % 100 == 0:
                print(f"Processed step {i + 1}/{len(df)}")

        # Signal completion
        self._write_sync(len(df), "done")

        # Save output log
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_log, index=False)
        print(f"Exchange log saved to {output_log}")

    def _write_sync(self, step: int, phase: str):
        """Write sync file with current step and phase."""
        with open(self.sync_path, "w") as f:
            f.write(f"step:{step}:{phase}")

    def _read_sync(self) -> tuple:
        """Read sync file and return (step, phase)."""
        try:
            with open(self.sync_path, "r") as f:
                content = f.read().strip()
            parts = content.split(":")
            if len(parts) >= 3:
                return int(parts[1]), parts[2]
        except (FileNotFoundError, ValueError):
            pass
        return None, None

    def _write_network_state(self, state: NetworkState):
        """Write network state to CSV."""
        df = pd.DataFrame([state.to_dict()])
        df.to_csv(self.network_state_path, index=False)

    def _read_network_state(self) -> Optional[NetworkState]:
        """Read network state from CSV."""
        try:
            df = pd.read_csv(self.network_state_path)
            if len(df) > 0:
                return NetworkState.from_dict(df.iloc[0].to_dict())
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        return None

    def _write_queue_context(self, context: QueueContext):
        """Write queue context to CSV."""
        df = pd.DataFrame([context.to_dict()])
        df.to_csv(self.queue_context_path, index=False)

    def _read_queue_context(self) -> Optional[QueueContext]:
        """Read queue context from CSV."""
        try:
            df = pd.read_csv(self.queue_context_path)
            if len(df) > 0:
                return QueueContext.from_dict(df.iloc[0].to_dict())
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        return None

    def _write_rat_decision(self, decision: RATDecision):
        """Write RAT decision to CSV."""
        df = pd.DataFrame([decision.to_dict()])
        df.to_csv(self.rat_decision_path, index=False)

    def _read_rat_decision(self) -> Optional[RATDecision]:
        """Read RAT decision from CSV."""
        try:
            df = pd.read_csv(self.rat_decision_path)
            if len(df) > 0:
                return RATDecision.from_dict(df.iloc[0].to_dict())
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        return None

    def _wait_for_file_update(self, filepath: str) -> bool:
        """
        Wait for a file to be updated.

        Returns True if file was updated within timeout, False otherwise.
        """
        start_time = time.time()
        try:
            initial_mtime = os.path.getmtime(filepath)
        except FileNotFoundError:
            initial_mtime = 0

        while time.time() - start_time < self.timeout:
            try:
                current_mtime = os.path.getmtime(filepath)
                if current_mtime > initial_mtime:
                    return True
            except FileNotFoundError:
                pass
            time.sleep(self.poll_interval)

        return False


# =============================================================================
# Phase 2 Alternative: Append-Only Logs
# =============================================================================

class AppendOnlyLogger:
    """
    Simple append-only logging for Phase 2 without explicit synchronization.

    Both RAT selector and queue simulator append to shared logs.
    Each system processes rows in order, polling for new entries.

    Log Files:
        rat_selector_log.csv:  RAT decisions
        queue_simulator_log.csv:  DTMC states (written by external simulator)
    """

    def __init__(self, log_dir: str, model_type: str = "lstm"):
        """
        Initialize append-only logger.

        Args:
            log_dir: Directory for log files
            model_type: RNN model type
        """
        self.log_dir = log_dir
        self.model_type = model_type

        self.rat_log_path = os.path.join(log_dir, "rat_selector_log.csv")
        self.queue_log_path = os.path.join(log_dir, "queue_simulator_log.csv")

        os.makedirs(log_dir, exist_ok=True)

        # Initialize API
        self.api = RATSelectionAPI(model_type=model_type)

        # Initialize log files with headers
        self._init_logs()

    def _init_logs(self):
        """Initialize log files with headers."""
        if not os.path.exists(self.rat_log_path):
            with open(self.rat_log_path, "w") as f:
                f.write("step,timestamp_ms,latitude,longitude,selected_rat,"
                        "pred_latency,pred_pdr,confidence,"
                        "pred_latency_dsrc,pred_pdr_dsrc,"
                        "pred_latency_pc5,pred_pdr_pc5,"
                        "pred_latency_5g,pred_pdr_5g\n")

    def process_input_data(self, input_csv: str):
        """
        Process input data and append RAT decisions to log.

        Args:
            input_csv: Path to input CSV with matched data
        """
        df = pd.read_csv(input_csv)

        for step, row in df.iterrows():
            state = _row_to_network_state(row)
            decision = self.api.select_rat(state)

            # Build log entry
            entry = {
                "step": step,
                "timestamp_ms": state.timestamp_ms,
                "latitude": state.latitude,
                "longitude": state.longitude,
                "selected_rat": decision.selected_rat.value,
                "pred_latency": decision.predicted_latency_ms,
                "pred_pdr": decision.predicted_pdr,
                "confidence": decision.confidence,
            }

            # Add per-RAT predictions
            for rat in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
                if rat in decision.all_predictions:
                    lat, pdr = decision.all_predictions[rat]
                    entry[f"pred_latency_{rat.value}"] = lat
                    entry[f"pred_pdr_{rat.value}"] = pdr
                else:
                    entry[f"pred_latency_{rat.value}"] = ""
                    entry[f"pred_pdr_{rat.value}"] = ""

            # Append to log
            self._append_to_log(entry)

            if (step + 1) % 500 == 0:
                print(f"Processed {step + 1}/{len(df)} rows")

        print(f"RAT decisions appended to {self.rat_log_path}")

    def _append_to_log(self, entry: Dict):
        """Append entry to RAT selector log."""
        with open(self.rat_log_path, "a") as f:
            values = [str(entry.get(col, "")) for col in [
                "step", "timestamp_ms", "latitude", "longitude", "selected_rat",
                "pred_latency", "pred_pdr", "confidence",
                "pred_latency_dsrc", "pred_pdr_dsrc",
                "pred_latency_pc5", "pred_pdr_pc5",
                "pred_latency_5g", "pred_pdr_5g"
            ]]
            f.write(",".join(values) + "\n")

    def get_last_processed_step(self) -> int:
        """Get the last step number in the RAT selector log."""
        try:
            df = pd.read_csv(self.rat_log_path)
            if len(df) > 0:
                return int(df["step"].max())
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        return -1

    def read_queue_log(self, from_step: int = 0) -> pd.DataFrame:
        """
        Read queue simulator log entries from a specific step.

        Args:
            from_step: Starting step number

        Returns:
            DataFrame with queue simulator entries
        """
        try:
            df = pd.read_csv(self.queue_log_path)
            return df[df["step"] >= from_step]
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()


# =============================================================================
# CLI Entry Points
# =============================================================================

def main():
    """CLI entry point for file integration."""
    import argparse

    parser = argparse.ArgumentParser(description="File-based RAT selection integration")
    parser.add_argument("--mode", choices=["batch", "exchange", "append"],
                        required=True, help="Integration mode")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm", "gru", "rnn"], help="Model type")
    parser.add_argument("--exchange_dir", type=str, help="Exchange directory for Phase 2")

    args = parser.parse_args()

    if args.mode == "batch":
        output = args.output or os.path.join(OUTPUT_DIR, "rat_decisions.csv")
        process_batch(args.input, output, args.model_type)

    elif args.mode == "exchange":
        if not args.exchange_dir:
            print("Error: --exchange_dir required for exchange mode")
            return
        coordinator = FileExchangeCoordinator(args.exchange_dir, args.model_type)
        output = args.output or os.path.join(args.exchange_dir, "exchange_log.csv")
        coordinator.run_coordinator(args.input, output)

    elif args.mode == "append":
        log_dir = args.output or OUTPUT_DIR
        logger = AppendOnlyLogger(log_dir, args.model_type)
        logger.process_input_data(args.input)


if __name__ == "__main__":
    main()
