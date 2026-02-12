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
from keras.models import load_model

from api_types import (
    RATType, NetworkState, QueueContext, RATDecision,
    PacketSizeDecision, TransmissionOutcome,
)
from config import (
    MODEL_DIR, OUTPUT_DIR, TIMESTEPS, TARGET_COLS,
    PDR_RELIABILITY_THRESHOLD, PDR_AVAILABILITY_THRESHOLD, LATENCY_TIE_MARGIN_MS,
    PACKET_SIZE_BOUNDS,
    create_gps_scaler, create_latency_scaler,
    create_sinr_5g_scaler, create_rsrp_5g_scaler, create_rsrp_dsrc_scaler,
)
from utils import get_latest_model
from learning.model import rmse
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

    def __init__(self, model_type: str = "lstm", model_dir: str = MODEL_DIR):
        """
        Initialize the RAT Selection API.

        Args:
            model_type: RNN model architecture ('lstm', 'gru', 'rnn')
            model_dir: Directory containing saved model files
        """
        self.model_type = model_type
        self.model_dir = model_dir
        self.models: Dict[str, object] = {}
        self.gps_scaler = create_gps_scaler()
        self.latency_scaler = create_latency_scaler()
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

        # Sequence history for predictions
        self._history: Dict[str, List[np.ndarray]] = {
            "dsrc": [],
            "pc5": [],
            "5g": [],
        }

        # Load models
        self._load_models()

    def _load_models(self) -> None:
        """Load all RAT models for the specified model type."""
        for rat in ["dsrc", "pc5", "5g"]:
            model_path = get_latest_model(self.model_type, rat, self.model_dir)
            if model_path:
                self.models[rat] = load_model(model_path, custom_objects={"rmse": rmse})
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
            Numpy array of normalized features
        """
        lat_lon = self.gps_scaler.transform([[state.latitude, state.longitude]])[0]

        if rat == "5g":
            # Features: lat, lon, latency, sinr, rsrp, pdr
            latency = state.fiveg_latency_ms or 0.0
            sinr = state.fiveg_sinr or 0.0
            rsrp = state.fiveg_rsrp or 0.0
            pdr = state.fiveg_pdr or 0.0
            latency_norm = self.latency_scaler.transform([[latency]])[0][0]
            sinr_norm = self.sinr_5g_scaler.transform([[sinr]])[0][0]
            rsrp_norm = self.rsrp_5g_scaler.transform([[rsrp]])[0][0]
            return np.array([lat_lon[0], lat_lon[1], latency_norm, sinr_norm, rsrp_norm, pdr])

        elif rat == "pc5":
            # Features: lat, lon, latency, pdr
            latency = state.pc5_latency_ms or 0.0
            pdr = state.pc5_pdr or 0.0
            latency_norm = self.latency_scaler.transform([[latency]])[0][0]
            return np.array([lat_lon[0], lat_lon[1], latency_norm, pdr])

        elif rat == "dsrc":
            # Features: lat, lon, rsrp_1, rsrp_2, latency, pdr
            rsrp_1 = state.dsrc_rsrp_1 or 0.0
            rsrp_2 = state.dsrc_rsrp_2 or 0.0
            latency = state.dsrc_latency_ms or 0.0
            pdr = state.dsrc_pdr or 0.0
            latency_norm = self.latency_scaler.transform([[latency]])[0][0]
            rsrp_1_norm = self.rsrp_dsrc_scaler.transform([[rsrp_1]])[0][0]
            rsrp_2_norm = self.rsrp_dsrc_scaler.transform([[rsrp_2]])[0][0]
            return np.array([lat_lon[0], lat_lon[1], rsrp_1_norm, rsrp_2_norm, latency_norm, pdr])

        else:
            raise ValueError(f"Unknown RAT: {rat}")

    def _build_sequence(self, state: NetworkState, rat: str) -> np.ndarray:
        """
        Build input sequence for model prediction.

        Maintains a sliding window of features for each RAT to enable
        sequence-based predictions.

        Args:
            state: Current network state
            rat: RAT type

        Returns:
            Input sequence array of shape (1, TIMESTEPS, num_features)
        """
        features = self._state_to_features(state, rat)
        self._history[rat].append(features)

        # Keep only last TIMESTEPS entries
        if len(self._history[rat]) > TIMESTEPS:
            self._history[rat] = self._history[rat][-TIMESTEPS:]

        # Build sequence with padding if needed
        history = self._history[rat]
        if len(history) < TIMESTEPS:
            padding_count = TIMESTEPS - len(history)
            padding = [np.zeros_like(features) for _ in range(padding_count)]
            sequence = np.array(padding + history)
        else:
            sequence = np.array(history)

        return sequence.reshape(1, TIMESTEPS, -1)

    def get_predictions(
        self,
        state: NetworkState,
    ) -> Dict[RATType, Tuple[float, float]]:
        """
        Get (latency, pdr) predictions for all RATs.

        Args:
            state: Current network state

        Returns:
            Dictionary mapping RATType to (predicted_latency, predicted_pdr)
        """
        predictions = {}

        for rat_str, rat_enum in [("dsrc", RATType.DSRC), ("pc5", RATType.PC5), ("5g", RATType.FiveG)]:
            if rat_str not in self.models:
                continue

            model = self.models[rat_str]
            sequence = self._build_sequence(state, rat_str)

            pred = model.predict(sequence, verbose=0)
            pred_latency = pred[0].flatten()[0]
            pred_pdr = pred[1].flatten()[0]

            # Denormalize latency
            pred_latency = self.latency_scaler.inverse_transform([[pred_latency]])[0][0]
            # Clamp PDR to [0, 1]
            pred_pdr = max(0.0, min(1.0, pred_pdr))

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

    def _get_actual_pdr(self, state: NetworkState, rat: RATType) -> Optional[float]:
        """Get actual PDR from network state for a specific RAT."""
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

    def select_rat(
        self,
        state: NetworkState,
        queue_context: Optional[QueueContext] = None,
    ) -> RATDecision:
        """
        Select optimal RAT based on state and optional queue context.

        Implements a reliability-first, latency-optimized selection algorithm:
        1. Get predictions for all RATs
        2. Filter by PDR reliability threshold
        3. Select lowest latency among qualified RATs
        4. Apply queue-aware adjustments if context provided

        Args:
            state: Current network state with measurements
            queue_context: Optional queue state for queue-aware selection

        Returns:
            RATDecision with selected RAT and predictions
        """
        # Get predictions for all RATs
        all_predictions = self.get_predictions(state)

        # Build options list: (rat_enum, pred_latency, pred_pdr, actual_pdr)
        options = []
        for rat_str, rat_enum in [("dsrc", RATType.DSRC), ("pc5", RATType.PC5), ("5g", RATType.FiveG)]:
            if rat_enum in all_predictions:
                pred_lat, pred_pdr = all_predictions[rat_enum]
                actual_pdr = self._get_actual_pdr(state, rat_enum)
                options.append((rat_enum, pred_lat, pred_pdr, actual_pdr))

        if not options:
            return RATDecision(
                selected_rat=RATType.UNAVAILABLE,
                confidence=0.0,
                predicted_latency_ms=float("inf"),
                predicted_pdr=0.0,
                all_predictions=all_predictions,
                model_type=self.model_type,
            )

        # Adjust thresholds based on queue context
        pdr_threshold = self.pdr_threshold
        if queue_context:
            # High urgency -> accept slightly lower PDR
            if queue_context.urgency_level > 0.7:
                pdr_threshold = max(0.95, pdr_threshold - 0.02)
            # Declining PDR trend -> be more conservative
            if queue_context.recent_pdr_trend < -0.3:
                pdr_threshold = min(0.999, pdr_threshold + 0.005)

        # Filter by PDR threshold
        valid_options = [opt for opt in options if opt[2] >= pdr_threshold]

        if not valid_options:
            # Fallback: filter out unavailable RATs
            keep = [opt for opt in options if opt[3] is not None and opt[3] >= self.pdr_availability]
            if keep:
                best = max(keep, key=lambda x: x[2])
                selected_rat = best[0]
            elif options:
                # Last resort: select 5G if available
                fiveg_opt = next((opt for opt in options if opt[0] == RATType.FiveG), None)
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

        # Get predictions for selected RAT
        if selected_rat in all_predictions:
            pred_lat, pred_pdr = all_predictions[selected_rat]
        else:
            pred_lat, pred_pdr = float("inf"), 0.0

        # Compute confidence
        confidence = self._compute_confidence(selected_rat, all_predictions, state)

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
        outcomes_by_rat: Dict[str, List[TransmissionOutcome]] = {
            "dsrc": [], "pc5": [], "5g": []
        }
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
            # training data and call incremental_train. This is a placeholder.

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
