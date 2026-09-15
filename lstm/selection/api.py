"""
RAT Selection API for Queue Simulator Integration.

This module provides the RATSelectionAPI class that wraps existing RAT selection
functionality and exposes it through a clean API interface for integration with
an external packet queue simulator.

Architecture:
    - RAT Selection (This Project): Decides WHICH network to use (strategic layer)
    - Queue Simulator (External): Decides HOW to size packets (tactical layer)
"""
import os
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd

from api_types import (
    RATType, NetworkState, QueueContext, RATDecision,
    PacketSizeDecision, TransmissionOutcome,
)
from config import (
    MODEL_DIR, OUTPUT_DIR, TIMESTEPS, TARGET_COLS,
    PDR_RELIABILITY_THRESHOLD, PDR_AVAILABILITY_THRESHOLD, LATENCY_TIE_MARGIN_MS,
    PACKET_SIZE_BOUNDS, DEFAULT_RATS, LSTM_FALLBACK_RAT, validate_rats,
    create_gps_scaler, create_latency_scaler,
    create_sinr_5g_scaler, create_rsrp_5g_scaler, create_rsrp_dsrc_scaler,
)
from queuesim.phy_layer import compute_contention_pdr
from utils import get_latest_model
from learning.model import load_torch_model, predict_torch
from learning.data_preprocessing import preprocess_lstm_input


class RATSelectionAPI:
    """
    API for RAT selection with queue-aware decision support.

    This class wraps the existing RAT selection logic and provides a clean
    interface for integration with an external packet queue simulator.

    Attributes:
        model_type: RNN architecture type ('lstm', 'gru', 'rnn')
        models: Dictionary of loaded Keras models {rat: model}
        gps_scaler: Scaler for GPS coordinate normalization
        latency_scaler: Scaler for latency value normalization
        outcome_buffer: Buffer for transmission outcomes awaiting retraining
        pdr_threshold: PDR reliability threshold for RAT selection
        pdr_availability: Minimum PDR to consider RAT available
        latency_margin: Latency tie-breaking margin in ms
    """

    def __init__(self, model_type: str = "lstm", model_dir: str = MODEL_DIR,
                 rats=DEFAULT_RATS):
        """
        Initialize the RAT Selection API.

        Args:
            model_type: RNN model architecture ('lstm', 'gru', 'rnn')
            model_dir: Directory containing saved model files
            rats: RATs eligible for selection (default: DEFAULT_RATS). Models
                  of other RATs are never loaded, even if present in model_dir.
        """
        self.model_type = model_type
        self.model_dir = model_dir
        self.rats = validate_rats(rats)
        self._rat_enums = [(rat, RATType.from_string(rat)) for rat in self.rats]
        self.models: Dict[str, object] = {}
        self.gps_scaler = create_gps_scaler()
        self.latency_scaler = create_latency_scaler()  # global fallback
        self.latency_scalers = {
            rat: create_latency_scaler(rat)
            for rat in self.rats
        }
        self.sinr_5g_scaler = create_sinr_5g_scaler()
        self.rsrp_5g_scaler = create_rsrp_5g_scaler()
        self.rsrp_dsrc_scaler = create_rsrp_dsrc_scaler()

        # Configurable thresholds
        self.pdr_threshold = PDR_RELIABILITY_THRESHOLD
        self.pdr_availability = PDR_AVAILABILITY_THRESHOLD
        self.latency_margin = LATENCY_TIE_MARGIN_MS

        # Outcome buffer for incremental learning
        self.outcome_buffer: List[TransmissionOutcome] = []
        self.retrain_interval = 500

        # Opportunistic: previous RAT for sticky policy
        self._previous_rat: RATType = RATType.FiveG

        # Observed feature history per RAT (filled by observe(), after each decision)
        self._history: Dict[str, List[np.ndarray]] = {rat: [] for rat in self.rats}
        # Most recent observed state; decisions never see the state they decide for
        self._last_state: Optional[NetworkState] = None

        # Load models
        self._load_models()

    def _load_models(self) -> None:
        """Load the models of the selected RATs for the specified model type."""
        for rat in self.rats:
            model_path = get_latest_model(self.model_type, rat, self.model_dir)
            if model_path:
                self.models[rat] = load_torch_model(model_path)
                print(f"Loaded {self.model_type} model for {rat}")
            else:
                print(f"Warning: No model found for {self.model_type}_{rat}")

    def _state_to_features(self, state: NetworkState, rat: str) -> np.ndarray:
        """
        Convert NetworkState to feature array for a specific RAT.

        Args:
            state: Current network state
            rat: RAT type ('dsrc', 'pc5', '5g')

        Returns:
            Numpy array of normalized features, in FEATURE_COLS[rat] order.
            Missing measurements are NaN (never zero-filled), matching
            learning.data_preprocessing.normalize_features.
        """
        def value(v):
            return np.nan if v is None else float(v)

        lat_lon = self.gps_scaler.transform([[state.latitude, state.longitude]])[0]
        lat_scaler = self.latency_scalers.get(rat, self.latency_scaler)

        if rat == "5g":
            # Features: lat, lon, latency, sinr, rsrp, pdr
            latency = value(state.fiveg_latency_ms)
            sinr = value(state.fiveg_sinr)
            rsrp = value(state.fiveg_rsrp)
            pdr = value(state.fiveg_pdr)
            latency_norm = lat_scaler.transform([[latency]])[0][0]
            sinr_norm = self.sinr_5g_scaler.transform([[sinr]])[0][0]
            rsrp_norm = self.rsrp_5g_scaler.transform([[rsrp]])[0][0]
            return np.array([lat_lon[0], lat_lon[1], latency_norm, sinr_norm, rsrp_norm, pdr])

        elif rat == "pc5":
            # Features: lat, lon, latency, pdr
            latency = value(state.pc5_latency_ms)
            pdr = value(state.pc5_pdr)
            latency_norm = lat_scaler.transform([[latency]])[0][0]
            return np.array([lat_lon[0], lat_lon[1], latency_norm, pdr])

        elif rat == "dsrc":
            # Features: lat, lon, rsrp_1, rsrp_2, latency, pdr
            # RSRP sentinel values (> 0 dBm, modem error) are missing, as in training
            rsrp_1 = value(state.dsrc_rsrp_1)
            rsrp_2 = value(state.dsrc_rsrp_2)
            rsrp_1 = np.nan if rsrp_1 > 0 else rsrp_1
            rsrp_2 = np.nan if rsrp_2 > 0 else rsrp_2
            latency = value(state.dsrc_latency_ms)
            pdr = value(state.dsrc_pdr)
            latency_norm = lat_scaler.transform([[latency]])[0][0]
            rsrp_1_norm = self.rsrp_dsrc_scaler.transform([[rsrp_1]])[0][0]
            rsrp_2_norm = self.rsrp_dsrc_scaler.transform([[rsrp_2]])[0][0]
            return np.array([lat_lon[0], lat_lon[1], rsrp_1_norm, rsrp_2_norm, latency_norm, pdr])

        else:
            raise ValueError(f"Unknown RAT: {rat}")

    def observe(self, state: NetworkState) -> None:
        """
        Record a step's measured state, after the decision for that step.

        Appends the state's features to every active RAT's history (keeping the
        last TIMESTEPS) and remembers it as the last observed state. Call this
        once per step, including steps whose decision was UNAVAILABLE, so the
        next decision sees data up to this step and never this step's own data
        before deciding.

        Args:
            state: Measured network state of the step just decided
        """
        for rat in self.rats:
            history = self._history[rat]
            history.append(self._state_to_features(state, rat))
            if len(history) > TIMESTEPS:
                del history[:-TIMESTEPS]
        self._last_state = state

    def input_window(self, rat: str) -> Optional[np.ndarray]:
        """
        Model input built from the observed history of a RAT.

        Matches training: the last TIMESTEPS observed steps, oldest first.

        Args:
            rat: RAT type

        Returns:
            Array of shape (1, TIMESTEPS, num_features), or None when fewer than
            TIMESTEPS steps were observed or the window has a missing measurement
        """
        history = self._history.get(rat, [])
        if len(history) < TIMESTEPS:
            return None
        window = np.array(history[-TIMESTEPS:])
        if np.isnan(window).any():
            return None
        return window.reshape(1, TIMESTEPS, -1)

    def get_context(self) -> Dict:
        """Copy of the per-stream decision state (history, last observation, sticky RAT)."""
        return {
            "history": {rat: list(h) for rat, h in self._history.items()},
            "last_state": self._last_state,
            "previous_rat": self._previous_rat,
        }

    def set_context(self, context: Dict) -> None:
        """Restore decision state saved by get_context (copied, not aliased)."""
        self._history = {rat: list(context["history"].get(rat, [])) for rat in self.rats}
        self._last_state = context["last_state"]
        self._previous_rat = context["previous_rat"]

    def get_predictions(
        self,
        state: Optional[NetworkState] = None,
    ) -> Dict[RATType, Tuple[float, float]]:
        """
        Get (latency, pdr) predictions for all RATs with a full observed window.

        Inputs come only from observed history (see observe/input_window), so
        `state` — the step being decided — is never a model input. It is kept
        as a parameter for API compatibility.

        Returns:
            Dictionary mapping RATType to (predicted_latency, predicted_pdr);
            RATs without a full window of measurements are omitted
        """
        predictions = {}

        for rat_str, rat_enum in self._rat_enums:
            if rat_str not in self.models:
                continue

            model = self.models[rat_str]
            sequence = self.input_window(rat_str)
            if sequence is None:
                continue

            pred_lat_arr, pred_pdr_arr = predict_torch(model, sequence)
            pred_latency = pred_lat_arr[0]
            pred_pdr = pred_pdr_arr[0]

            # Clamp normalized outputs to [0, 1] and denormalize
            pred_latency = float(np.clip(pred_latency, 0.0, 1.0))
            lat_scaler = self.latency_scalers.get(rat_str, self.latency_scaler)
            pred_latency = lat_scaler.inverse_transform([[pred_latency]])[0][0]
            pred_pdr = float(np.clip(pred_pdr, 0.0, 1.0))

            predictions[rat_enum] = (pred_latency, pred_pdr)

        return predictions

    def _compute_confidence(
        self,
        selected_rat: RATType,
        all_predictions: Dict[RATType, Tuple[float, float]],
        state: NetworkState
    ) -> float:
        """
        Compute confidence score for RAT selection.

        Confidence is based on:
        - PDR margin above threshold
        - Latency margin vs alternatives
        - Data availability for the selected RAT

        Args:
            selected_rat: The selected RAT
            all_predictions: Predictions for all RATs
            state: Current network state

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if selected_rat == RATType.UNAVAILABLE:
            return 0.0

        pred_lat, pred_pdr = all_predictions.get(selected_rat, (float("inf"), 0.0))

        # PDR-based confidence (0.5 weight)
        pdr_margin = pred_pdr - self.pdr_threshold
        pdr_confidence = min(1.0, max(0.0, 0.5 + pdr_margin * 5))

        # Latency-based confidence (0.3 weight)
        other_latencies = [
            lat for rat, (lat, pdr) in all_predictions.items()
            if rat != selected_rat and pdr >= self.pdr_threshold
        ]
        if other_latencies:
            best_alternative = min(other_latencies)
            latency_margin = best_alternative - pred_lat
            latency_confidence = min(1.0, max(0.0, 0.5 + latency_margin / 10))
        else:
            latency_confidence = 1.0

        # Data availability confidence (0.2 weight)
        actual_pdr = self._get_actual_pdr(state, selected_rat)
        availability_confidence = 1.0 if actual_pdr is not None and actual_pdr > self.pdr_availability else 0.5

        return 0.5 * pdr_confidence + 0.3 * latency_confidence + 0.2 * availability_confidence

    def _get_actual_pdr(self, state: Optional[NetworkState], rat: RATType) -> Optional[float]:
        """Get measured PDR from a network state for a specific RAT (None if no state)."""
        if state is None:
            return None
        if rat == RATType.DSRC:
            return state.dsrc_pdr
        elif rat == RATType.PC5:
            return state.pc5_pdr
        elif rat == RATType.FiveG:
            return state.fiveg_pdr
        return None

    def _recommend_packet_size(
        self,
        selected_rat: RATType,
        predicted_pdr: float
    ) -> Optional[int]:
        """
        Recommend maximum packet size based on RAT and predicted PDR.

        Uses a conservative approach: lower PDR predictions result in
        smaller recommended packet sizes.

        Args:
            selected_rat: The selected RAT
            predicted_pdr: Predicted PDR for the selected RAT

        Returns:
            Recommended max packet size in bytes, or None if no recommendation
        """
        if selected_rat == RATType.UNAVAILABLE:
            return None

        bounds = PACKET_SIZE_BOUNDS.get(selected_rat.value, {"min": 100, "max": 1400})
        min_size = bounds["min"]
        max_size = bounds["max"]

        # Scale packet size recommendation based on PDR
        # High PDR (>0.99) -> max size
        # Low PDR (<0.85) -> min size
        if predicted_pdr >= 0.99:
            return max_size
        elif predicted_pdr <= 0.85:
            return min_size
        else:
            # Linear interpolation
            ratio = (predicted_pdr - 0.85) / (0.99 - 0.85)
            return int(min_size + ratio * (max_size - min_size))

    def _no_decision(self, all_predictions: Dict[RATType, Tuple[float, float]]) -> RATDecision:
        """Decision when the model cannot choose: the fallback RAT if active, else UNAVAILABLE."""
        if LSTM_FALLBACK_RAT is not None and LSTM_FALLBACK_RAT in self.rats:
            return RATDecision(
                selected_rat=RATType.from_string(LSTM_FALLBACK_RAT),
                confidence=0.0,
                predicted_latency_ms=float("nan"),
                predicted_pdr=float("nan"),
                all_predictions=all_predictions,
                model_type=self.model_type,
                fallback=True,
            )
        return RATDecision(
            selected_rat=RATType.UNAVAILABLE,
            confidence=0.0,
            predicted_latency_ms=float("inf"),
            predicted_pdr=0.0,
            all_predictions=all_predictions,
            model_type=self.model_type,
        )

    def select_rat(
        self,
        state: NetworkState,
        queue_context: Optional[QueueContext] = None,
        contention_context: Optional[Dict[RATType, Tuple[int, float]]] = None,
    ) -> RATDecision:
        """
        Select optimal RAT based on state and optional queue context.

        Implements a reliability-first, latency-optimized selection algorithm:
        1. Get predictions for all RATs
        2. Apply contention correction if context provided
        3. Filter by PDR reliability threshold
        4. Select lowest latency among qualified RATs
        5. Apply queue-aware adjustments if context provided

        Args:
            state: Current network state with measurements
            queue_context: Optional queue state for queue-aware selection
            contention_context: Optional dict mapping RATType to
                (n_vehicles, utilization) for contention-adjusted PDR filtering

        Returns:
            RATDecision with selected RAT and predictions
        """
        # Get predictions for all RATs
        all_predictions = self.get_predictions(state)

        # Build options list: (rat_enum, pred_latency, effective_pdr, actual_pdr)
        # effective_pdr = contention-corrected if context provided, else raw pred_pdr
        options = []
        for rat_str, rat_enum in self._rat_enums:
            if rat_enum in all_predictions:
                pred_lat, pred_pdr = all_predictions[rat_enum]
                # Last observed PDR (previous step), never the step being decided
                actual_pdr = self._get_actual_pdr(self._last_state, rat_enum)

                effective_pdr = pred_pdr
                if contention_context and rat_enum in contention_context:
                    n_veh, util = contention_context[rat_enum]
                    effective_pdr = compute_contention_pdr(
                        pred_pdr, util, rat_enum, n_veh,
                    )

                options.append((rat_enum, pred_lat, effective_pdr, actual_pdr))

        if not options:
            # No prediction for any RAT (no full measurement window yet, or missing measurements)
            return self._no_decision(all_predictions)

        # Adjust thresholds based on queue context
        pdr_threshold = self.pdr_threshold
        if queue_context:
            # High urgency -> accept slightly lower PDR
            if queue_context.urgency_level > 0.7:
                pdr_threshold = max(0.95, pdr_threshold - 0.02)
            # Declining PDR trend -> be more conservative
            if queue_context.recent_pdr_trend < -0.3:
                pdr_threshold = min(0.999, pdr_threshold + 0.005)

        # Filter by PDR threshold (uses effective_pdr which may be contention-corrected)
        valid_options = [opt for opt in options if opt[2] >= pdr_threshold]

        if not valid_options:
            # When contention context is active, prefer 5G (scheduled access,
            # contention-free) as safe fallback — low predicted PDR early in
            # the simulation is a model warm-up artifact, not a real signal.
            if contention_context:
                fiveg_opt = next(
                    (opt for opt in options if opt[0] == RATType.FiveG), None,
                )
                if fiveg_opt:
                    selected_rat = RATType.FiveG
                else:
                    # No 5G model loaded — fall through to generic fallback
                    keep = [opt for opt in options
                            if opt[3] is not None and opt[3] >= self.pdr_availability]
                    selected_rat = max(keep, key=lambda x: x[2])[0] if keep else RATType.UNAVAILABLE
            else:
                # Original fallback (no contention context — single vehicle)
                keep = [opt for opt in options
                        if opt[3] is not None and opt[3] >= self.pdr_availability]
                if keep:
                    best = max(keep, key=lambda x: x[2])
                    selected_rat = best[0]
                elif options:
                    fiveg_opt = next(
                        (opt for opt in options if opt[0] == RATType.FiveG), None,
                    )
                    if fiveg_opt and (fiveg_opt[3] is None or fiveg_opt[3] >= self.pdr_availability):
                        selected_rat = RATType.FiveG
                    else:
                        selected_rat = RATType.UNAVAILABLE
                else:
                    selected_rat = RATType.UNAVAILABLE
        else:
            # Select lowest latency
            valid_options.sort(key=lambda x: x[1])
            best_latency = valid_options[0][1]
            selected_rat = valid_options[0][0]

            # Gather all options within margin of the best, then apply priority
            tied = [opt for opt in valid_options
                    if abs(opt[1] - best_latency) < self.latency_margin]
            priority = {RATType.FiveG: 0, RATType.PC5: 1, RATType.DSRC: 2}
            selected_rat = min(tied, key=lambda opt: priority.get(opt[0], 99))[0]

        fallback_rat = RATType.from_string(LSTM_FALLBACK_RAT) if LSTM_FALLBACK_RAT is not None else None
        if selected_rat == RATType.UNAVAILABLE and fallback_rat not in all_predictions:
            # Every predicted RAT is unavailable, and the fallback RAT has no prediction to judge it by
            return self._no_decision(all_predictions)

        # Get predictions for selected RAT
        if selected_rat in all_predictions:
            pred_lat, pred_pdr = all_predictions[selected_rat]
        else:
            pred_lat, pred_pdr = float("inf"), 0.0

        # Compute confidence
        confidence = self._compute_confidence(selected_rat, all_predictions, self._last_state)

        # Recommend packet size
        recommended_size = self._recommend_packet_size(selected_rat, pred_pdr)

        return RATDecision(
            selected_rat=selected_rat,
            confidence=confidence,
            predicted_latency_ms=pred_lat,
            predicted_pdr=pred_pdr,
            all_predictions=all_predictions,
            recommended_max_packet_size=recommended_size,
            model_type=self.model_type,
        )

    def select_rat_opportunistic(
        self,
        state: NetworkState,
    ) -> RATDecision:
        """
        Select RAT using reactive/opportunistic logic — no model inference.

        Uses the metrics of the last observed step (see observe), never those
        of the step being decided, with a sticky policy to reduce handovers.
        Mirrors the opportunistic_best_rat algorithm from rat_selection.py.
        Returns UNAVAILABLE until a step has been observed.

        Algorithm:
            1. Filter RATs with actual PDR > 5%
            2. If on 5G and V2X available, switch to lowest latency
            3. Sticky: stay on current RAT if still viable
            4. Fall back to 5G

        Args:
            state: Current network state with measurements

        Returns:
            RATDecision with actual (not predicted) values
        """
        # Build options from actual observed metrics: (rat_str, rat_enum, latency, pdr)
        observed = self._last_state
        if observed is None:
            return RATDecision(
                selected_rat=RATType.UNAVAILABLE,
                confidence=0.0,
                predicted_latency_ms=float("inf"),
                predicted_pdr=0.0,
                all_predictions={},
                model_type="opportunistic",
            )

        options = []
        for rat_str, rat_enum in self._rat_enums:
            field_prefix = "fiveg" if rat_str == "5g" else rat_str
            lat_val = getattr(observed, f"{field_prefix}_latency_ms")
            pdr_val = getattr(observed, f"{field_prefix}_pdr")
            latency = lat_val if lat_val is not None else 0.0
            pdr = pdr_val if pdr_val is not None else 0.0
            options.append((rat_str, rat_enum, latency, pdr))

        # Build all_predictions dict with actual values (not model predictions)
        all_predictions: Dict[RATType, Tuple[float, float]] = {
            opt[1]: (opt[2], opt[3]) for opt in options
        }

        # Filter by PDR > 50%
        valid_options = [opt for opt in options if opt[3] > 0.5]

        if not valid_options:
            # Try any with PDR > 0
            keep = [opt for opt in options if opt[3] > 0.0]
            if keep:
                best = min(keep, key=lambda x: x[2])
                selected_rat = best[1]
            else:
                fiveg = next((opt for opt in options if opt[0] == "5g"), None)
                if fiveg and (fiveg[3] or 0.0) > self.pdr_availability:
                    selected_rat = RATType.FiveG
                else:
                    selected_rat = RATType.UNAVAILABLE
        elif len(valid_options) == 1 and valid_options[0][0] == "5g":
            selected_rat = RATType.FiveG
        elif self._previous_rat == RATType.FiveG and any(
            opt[0] != "5g" for opt in valid_options
        ):
            # On 5G and V2X available → switch to lowest latency
            best = min(valid_options, key=lambda x: x[2])
            selected_rat = best[1]
        elif any(opt[1] == self._previous_rat for opt in valid_options):
            # Sticky: stay on current RAT
            selected_rat = self._previous_rat
        else:
            fiveg = next((opt for opt in options if opt[0] == "5g"), None)
            if fiveg and (fiveg[3] or 0.0) > self.pdr_availability:
                selected_rat = RATType.FiveG
            else:
                selected_rat = RATType.UNAVAILABLE

        self._previous_rat = selected_rat

        # Get actual values for selected RAT
        if selected_rat in all_predictions:
            sel_lat, sel_pdr = all_predictions[selected_rat]
        else:
            sel_lat, sel_pdr = float("inf"), 0.0

        return RATDecision(
            selected_rat=selected_rat,
            confidence=1.0,
            predicted_latency_ms=sel_lat,
            predicted_pdr=sel_pdr,
            all_predictions=all_predictions,
            model_type="opportunistic",
        )

    def report_outcome(self, outcome: TransmissionOutcome) -> None:
        """
        Report transmission outcome for model retraining.

        Outcomes are buffered and models are retrained every N samples.

        Args:
            outcome: The transmission outcome to report
        """
        self.outcome_buffer.append(outcome)

        if len(self.outcome_buffer) >= self.retrain_interval:
            self._retrain_models()
            self.outcome_buffer = []

    def _retrain_models(self) -> None:
        """Trigger model retraining with buffered outcomes."""
        # Group outcomes by RAT
        outcomes_by_rat: Dict[str, List[TransmissionOutcome]] = {rat: [] for rat in self.rats}
        for outcome in self.outcome_buffer:
            rat_str = outcome.rat_used.value
            if rat_str in outcomes_by_rat:
                outcomes_by_rat[rat_str].append(outcome)

        # Retrain each RAT model if enough data
        for rat, outcomes in outcomes_by_rat.items():
            if len(outcomes) < 50 or rat not in self.models:
                continue

            print(f"Retraining {self.model_type}_{rat} with {len(outcomes)} samples")
            # Note: Full retraining implementation would convert outcomes to
            # training data and call incremental_train_torch. This is a placeholder.

    def set_thresholds(
        self,
        pdr_reliability: Optional[float] = None,
        pdr_availability: Optional[float] = None,
        latency_tie_margin_ms: Optional[float] = None
    ) -> None:
        """
        Update selection thresholds dynamically.

        Args:
            pdr_reliability: Minimum PDR for reliable transmission
            pdr_availability: Minimum PDR to consider RAT available
            latency_tie_margin_ms: Latency difference to trigger tie-breaking
        """
        if pdr_reliability is not None:
            self.pdr_threshold = pdr_reliability
        if pdr_availability is not None:
            self.pdr_availability = pdr_availability
        if latency_tie_margin_ms is not None:
            self.latency_margin = latency_tie_margin_ms

    def reset_history(self) -> None:
        """Reset sequence history for all RATs."""
        for rat in self._history:
            self._history[rat] = []


class JointController:
    """
    Orchestrator for RAT selection and queue simulator integration.

    This class coordinates between the RAT selector and an external
    queue simulator, managing the bidirectional state exchange.

    Attributes:
        rat_selector: RATSelectionAPI instance
        queue_sim: External queue simulator API (duck-typed)
    """

    def __init__(self, rat_selector: RATSelectionAPI, queue_sim=None):
        """
        Initialize the joint controller.

        Args:
            rat_selector: RATSelectionAPI instance for RAT decisions
            queue_sim: Optional queue simulator implementing QueueSimulatorAPI
        """
        self.rat_selector = rat_selector
        self.queue_sim = queue_sim

    def process_transmission(
        self,
        state: NetworkState
    ) -> Tuple[RATDecision, Optional[PacketSizeDecision]]:
        """
        Main entry point for joint RAT/packet size decisions.

        Workflow:
        1. Get queue context from simulator (if available)
        2. Select RAT (considering queue context)
        3. Decide packet size (considering RAT decision)

        Args:
            state: Current network state

        Returns:
            Tuple of (RATDecision, PacketSizeDecision or None)
        """
        # Get queue context if simulator available
        queue_context = None
        if self.queue_sim is not None:
            queue_context = self.queue_sim.get_queue_context()

        # Make RAT decision
        rat_decision = self.rat_selector.select_rat(state, queue_context)

        # Get packet size decision if simulator available
        packet_decision = None
        if self.queue_sim is not None:
            packet_decision = self.queue_sim.decide_packet_size(rat_decision, state)

        return rat_decision, packet_decision

    def report_outcome(self, outcome: TransmissionOutcome) -> None:
        """
        Report outcome to both RAT selector and queue simulator.

        Args:
            outcome: The transmission outcome to report
        """
        # Report to RAT selector for model retraining
        self.rat_selector.report_outcome(outcome)

        # Report to queue simulator for PDR estimate update
        if self.queue_sim is not None:
            self.queue_sim.update_pdr_estimate(outcome)
