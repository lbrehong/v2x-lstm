"""
Closed-loop feedback simulation: RAT selection -> DTMC packet sizing ->
simulated TX -> model retraining -> repeat.

Processes a super_merged CSV (with all RAT measurements at matched GPS
locations) through the full feedback loop, proving the architecture works
end-to-end.

Supports single-vehicle (default) and multi-vehicle platoon simulation
with contention effects when multiple vehicles share the same RAT.

Usage:
    # Single vehicle (original)
    python -m scripts.feedback_loop \
        --input /path/to/super_merged.csv \
        --model_type lstm \
        --seed 42

    # Multi-vehicle platoon (N=20)
    python -m scripts.feedback_loop \
        --input /path/to/super_merged.csv \
        --model_type lstm \
        --seed 42 \
        --num_vehicles 20
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from api_types import (
    RATType, NetworkState, TransmissionOutcome,
    RATDecision, PacketSizeDecision,
)
import config
from config import (
    TIMESTEPS, MODEL_DIR, PDR_CORRECTION_EXPONENT,
    create_latency_scaler, TX_INTERVAL_MS,
)
from utils import row_to_network_state, ensure_dir_exists
from selection.api import RATSelectionAPI
from queuesim.queue_simulator import QueueSimulator
from queuesim.dtmc_sizer import correct_pdr_for_packet_size
from queuesim.phy_layer import (
    calculate_tx_time_ms, get_base_latency_ms,
    compute_channel_utilization, compute_contention_pdr, compute_contention_latency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAT_STR = {RATType.DSRC: "dsrc", RATType.PC5: "pc5", RATType.FiveG: "5g"}


def _actual_latency(state: NetworkState, rat: RATType) -> Optional[float]:
    """Return the ground-truth latency for *rat* from the network state."""
    if rat == RATType.DSRC:
        return state.dsrc_latency_ms
    if rat == RATType.PC5:
        return state.pc5_latency_ms
    if rat == RATType.FiveG:
        return state.fiveg_latency_ms
    return None


def _actual_pdr(state: NetworkState, rat: RATType) -> Optional[float]:
    if rat == RATType.DSRC:
        return state.dsrc_pdr
    if rat == RATType.PC5:
        return state.pc5_pdr
    if rat == RATType.FiveG:
        return state.fiveg_pdr
    return None


def _simulate_tx(
    rat: RATType,
    packet_size: int,
    predicted_pdr: float,
    base_packet_size: int,
    correction_exponent: float,
) -> Tuple[bool, float, float]:
    """Simulate a single transmission.  Returns (delivered, latency_ms, corrected_pdr)."""
    corrected_pdr = correct_pdr_for_packet_size(
        predicted_pdr, base_packet_size, packet_size, correction_exponent,
    )
    delivered = random.random() < corrected_pdr

    tx_time = calculate_tx_time_ms(packet_size, rat)
    base_lat = get_base_latency_ms(rat)

    if rat == RATType.DSRC:
        jitter = 0.5 + random.random()
    else:
        jitter = 0.8 + 0.4 * random.random()

    size_factor = packet_size / base_packet_size
    latency = (base_lat + tx_time) * jitter * (0.95 + 0.1 * size_factor)
    return delivered, latency, corrected_pdr


# ---------------------------------------------------------------------------
# Multi-vehicle contention model (RB-consumption based)
# ---------------------------------------------------------------------------

# Per-vehicle signal noise standard deviations
SIGNAL_NOISE_STD = {
    "latency": 0.5,
    "pdr": 0.02,
    "sinr": 3.0,
    "rsrp": 2.0,
}


def _perturb_state(
    state: NetworkState, rng: np.random.RandomState,
) -> NetworkState:
    """Return a copy of *state* with per-vehicle signal noise added."""
    d = state.to_dict()

    lat_noise = rng.normal(0, SIGNAL_NOISE_STD["latency"])
    pdr_noise = rng.normal(0, SIGNAL_NOISE_STD["pdr"])
    sinr_noise = rng.normal(0, SIGNAL_NOISE_STD["sinr"])
    rsrp_noise = rng.normal(0, SIGNAL_NOISE_STD["rsrp"])

    for key in ("dsrc_latency_ms", "pc5_latency_ms", "fiveg_latency_ms"):
        if d.get(key) is not None:
            d[key] = max(0.1, d[key] + lat_noise)

    for key in ("dsrc_pdr", "pc5_pdr", "fiveg_pdr"):
        if d.get(key) is not None:
            d[key] = float(np.clip(d[key] + pdr_noise, 0.0, 1.0))

    if d.get("fiveg_sinr") is not None:
        d["fiveg_sinr"] = d["fiveg_sinr"] + sinr_noise

    for key in ("fiveg_rsrp", "dsrc_rsrp_1", "dsrc_rsrp_2"):
        if d.get(key) is not None:
            d[key] = d[key] + rsrp_noise

    return NetworkState.from_dict(d)


def _simulate_tx_with_contention(
    rat: RATType,
    packet_size: int,
    predicted_pdr: float,
    base_packet_size: int,
    correction_exponent: float,
    n_vehicles_on_rat: int,
    utilization: float,
) -> Tuple[bool, float, float]:
    """Simulate TX with RB-based contention effects.

    Returns (delivered, latency_ms, contended_pdr).
    """
    # 1. Packet-size correction (same as _simulate_tx)
    corrected_pdr = correct_pdr_for_packet_size(
        predicted_pdr, base_packet_size, packet_size, correction_exponent,
    )

    # 2. RB-based contention PDR
    contended_pdr = compute_contention_pdr(
        corrected_pdr, utilization, rat, n_vehicles_on_rat,
    )
    delivered = random.random() < contended_pdr

    # 3. Base latency with jitter (same as _simulate_tx)
    tx_time = calculate_tx_time_ms(packet_size, rat)
    base_lat = get_base_latency_ms(rat)

    if rat == RATType.DSRC:
        jitter = 0.5 + random.random()
    else:
        jitter = 0.8 + 0.4 * random.random()

    size_factor = packet_size / base_packet_size
    base_latency = (base_lat + tx_time) * jitter * (0.95 + 0.1 * size_factor)

    # 4. RB-based contention latency
    latency = compute_contention_latency(
        base_latency, utilization, n_vehicles_on_rat, rat,
    )
    return delivered, latency, contended_pdr


# ---------------------------------------------------------------------------
# Retraining logic
# ---------------------------------------------------------------------------

def _retrain_model(
    api: RATSelectionAPI,
    rat_str: str,
    x_buffer: np.ndarray,
    y_latency: np.ndarray,
    y_pdr: np.ndarray,
    cycle: int,
    retrain_log: List[Dict],
    latency_scaler=None,
):
    """Run one retraining cycle on the buffered data for *rat_str*.

    Safeguards:
    - Saves model weights before retraining; rolls back if loss spikes >2x.
    - Applies gradient clipping (clipnorm=1.0) to prevent large updates.
    """
    model = api.models[rat_str]

    # Save weights for potential rollback
    weights_before = model.get_weights()

    # Recompile optimizer with gradient clipping if not already set
    if not getattr(model.optimizer, "clipnorm", None):
        from keras.optimizers import Adam
        from learning.model import rmse
        lr = float(model.optimizer.learning_rate)
        model.compile(
            optimizer=Adam(learning_rate=lr, clipnorm=1.0),
            loss={"latency_ms": "mse", "pdr": "mse"},
            loss_weights={"latency_ms": 1.0, "pdr": 1.5},
            metrics={"latency_ms": [rmse], "pdr": [rmse]},
        )

    # Evaluate loss before retraining
    # Returns [total_loss, latency_loss, pdr_loss, latency_rmse, pdr_rmse]
    y_dict = {"latency_ms": y_latency, "pdr": y_pdr}
    eval_before = model.evaluate(x_buffer, y_dict, verbose=0)
    if not isinstance(eval_before, list):
        eval_before = [eval_before, None, None, None, None]

    loss_before = eval_before[0]

    # Retrain for 1 epoch with validation split
    lr_before = float(model.optimizer.learning_rate)
    t0 = time.time()
    history = model.fit(
        x_buffer, y_dict,
        epochs=1, batch_size=32, verbose=1,
        validation_split=0.15,
    )
    train_time_s = time.time() - t0
    h = history.history

    # Evaluate loss after retraining on the full buffer
    eval_after = model.evaluate(x_buffer, y_dict, verbose=0)
    if not isinstance(eval_after, list):
        eval_after = [eval_after]
    loss_after = eval_after[0]

    # Rollback if loss spiked significantly
    rolled_back = False
    if loss_before > 0 and loss_after > loss_before * 2.0:
        model.set_weights(weights_before)
        rolled_back = True
        print(
            f"  WARNING: Rolled back {api.model_type}_{rat_str} cycle {cycle}: "
            f"loss spiked {loss_before:.5f} -> {loss_after:.5f} (>{2.0:.1f}x)"
        )

    # Prediction accuracy snapshot (MAE in real units)
    preds = model.predict(x_buffer, verbose=0)
    pred_latency_norm = preds[0].flatten()
    pred_pdr = preds[1].flatten()
    # Denormalize latency predictions and targets using per-RAT scaler
    lat_scaler = latency_scaler if latency_scaler is not None else api.latency_scaler
    pred_latency_ms = lat_scaler.inverse_transform(
        np.clip(pred_latency_norm, 0.0, 1.0).reshape(-1, 1)).flatten()
    actual_latency_ms = lat_scaler.inverse_transform(
        y_latency.reshape(-1, 1)).flatten()
    latency_mae_ms = float(np.mean(np.abs(pred_latency_ms - actual_latency_ms)))
    pdr_mae = float(np.mean(np.abs(pred_pdr - y_pdr)))

    retrain_log.append({
        "cycle": cycle,
        "rat": rat_str,
        "n_samples": len(x_buffer),
        "train_time_s": round(train_time_s, 4),
        "learning_rate": lr_before,
        "rolled_back": rolled_back,
        # Before-retrain metrics
        "loss_before": eval_before[0],
        "latency_loss_before": eval_before[1],
        "pdr_loss_before": eval_before[2],
        "latency_rmse_before": eval_before[3],
        "pdr_rmse_before": eval_before[4],
        # After-retrain training metrics
        "loss_after": h["loss"][0],
        "latency_loss_after": h["latency_ms_loss"][0],
        "pdr_loss_after": h["pdr_loss"][0],
        "latency_rmse_after": h["latency_ms_rmse"][0],
        "pdr_rmse_after": h["pdr_rmse"][0],
        # After-retrain validation metrics
        "val_loss": h.get("val_loss", [None])[0],
        "val_latency_loss": h.get("val_latency_ms_loss", [None])[0],
        "val_pdr_loss": h.get("val_pdr_loss", [None])[0],
        "val_latency_rmse": h.get("val_latency_ms_rmse", [None])[0],
        "val_pdr_rmse": h.get("val_pdr_rmse", [None])[0],
        # Prediction accuracy (real units)
        "latency_mae_ms": round(latency_mae_ms, 4),
        "pdr_mae": round(pdr_mae, 6),
    })
    status = "ROLLED BACK" if rolled_back else f"loss {loss_before:.5f} -> {loss_after:.5f}"
    print(
        f"  Retrained {api.model_type}_{rat_str}: "
        f"{status}  "
        f"(val_loss={h.get('val_loss', [None])[0]:.5f}, "
        f"lat_mae={latency_mae_ms:.2f}ms, pdr_mae={pdr_mae:.4f})  "
        f"({len(x_buffer)} samples, cycle {cycle}, {train_time_s:.2f}s)"
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_feedback_loop(
    input_csv: str,
    model_type: str = "lstm",
    seed: Optional[int] = None,
    retrain_interval: int = 500,
    base_packet_size: int = 1024,
    correction_exponent: float = PDR_CORRECTION_EXPONENT,
    sim_tx_interval_ms: Optional[int] = None,
    enable_dtmc: bool = True,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    ensure_dir_exists(config.OUTPUT_DIR)
    ensure_dir_exists(MODEL_DIR)

    # Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # Initialise components
    api = RATSelectionAPI(model_type=model_type)
    qsim = QueueSimulator(
        base_packet_size=base_packet_size,
        correction_exponent=correction_exponent,
        enable_dtmc=enable_dtmc,
    )
    latency_scalers = {
        rat: create_latency_scaler(rat) for rat in ("5g", "pc5", "dsrc")
    }

    # Per-RAT retraining buffers:  list of (sequence, y_latency_norm, y_pdr)
    buffers: Dict[str, List[Tuple[np.ndarray, float, float]]] = {
        "dsrc": [], "pc5": [], "5g": [],
    }
    retrain_count: Dict[str, int] = {"dsrc": 0, "pc5": 0, "5g": 0}

    # Logging
    row_log: List[Dict] = []
    retrain_log: List[Dict] = []

    # Simulated time (seconds, incremented by TX_INTERVAL)
    sim_time = 0.0
    sim_dt = (sim_tx_interval_ms / 1000.0) if sim_tx_interval_ms else 0.1
    successful_bytes = 0
    previous_rat: Optional[RATType] = None
    rat_switches = 0

    # Compute repetition factor for TX rate simulation
    if sim_tx_interval_ms is not None:
        data_interval_ms = max(TX_INTERVAL_MS.values())  # 50ms (coarsest CSV rate)
        repeat_factor = data_interval_ms / sim_tx_interval_ms
    else:
        repeat_factor = 1.0

    total_tx_steps = int(len(df) * repeat_factor)
    tx_label = f"{sim_tx_interval_ms}ms (override)" if sim_tx_interval_ms else "per-RAT defaults"
    if repeat_factor != 1.0:
        print(f"TX interval: {tx_label}, repeat factor: {repeat_factor:.3f}, "
              f"effective TX steps: ~{total_tx_steps}")
        if repeat_factor > 20:
            print(f"  WARNING: repeat_factor={repeat_factor:.1f} — simulation will be slow")

    print(f"\nStarting feedback loop ({len(df)} data points, retrain every {retrain_interval})\n")

    accumulator = 0.0
    tx_step = 0

    for idx in range(len(df)):
        row = df.iloc[idx]

        accumulator += repeat_factor
        n_repeats = int(accumulator)
        accumulator -= n_repeats

        for rep in range(n_repeats):
            tx_step += 1

            # 1. Build NetworkState
            state = row_to_network_state(row, timestamp_ms=int(sim_time * 1000))

            # 2. RAT selection (also builds internal history/sequence)
            queue_ctx = qsim.get_queue_context()
            decision = api.select_rat(state, queue_ctx)
            selected_rat = decision.selected_rat

            if selected_rat == RATType.UNAVAILABLE:
                sim_time += sim_dt
                continue

            rat_str = RAT_STR[selected_rat]

            # Track RAT switches
            if previous_rat is not None and selected_rat != previous_rat:
                rat_switches += 1
            previous_rat = selected_rat

            # 3. DTMC packet sizing
            pkt_decision = qsim.decide_packet_size(decision, sim_time)
            packet_size = pkt_decision.packet_size_bytes

            # 4. Simulate transmission (use actual PDR from dataset as ceiling)
            actual_pdr_val = _actual_pdr(state, selected_rat) or decision.predicted_pdr
            delivered, sim_latency, corrected_pdr = _simulate_tx(
                selected_rat, packet_size, actual_pdr_val,
                base_packet_size, correction_exponent,
            )

            # 5. Feed outcome to DTMC (updates moving window immediately)
            outcome = TransmissionOutcome(
                timestamp_ms=int(sim_time * 1000),
                rat_used=selected_rat,
                packet_size_bytes=packet_size,
                actual_latency_ms=sim_latency,
                delivered=delivered,
                network_state=state,
            )
            qsim.update_pdr_estimate(outcome)
            qsim.metrics.total_packets += 1
            if delivered:
                qsim.metrics.successful_packets += 1
                successful_bytes += packet_size

            # 6. Capture sequence + ground-truth target for retraining
            history = api._history[rat_str]
            if len(history) >= TIMESTEPS:
                seq = np.array(history[-TIMESTEPS:])  # (TIMESTEPS, n_features)

                # Ground-truth targets from the CSV row
                actual_lat = _actual_latency(state, selected_rat)
                actual_pdr_val = _actual_pdr(state, selected_rat)

                if actual_lat is not None and actual_pdr_val is not None:
                    # Normalise latency target the same way training data was prepared
                    lat_norm = latency_scalers[rat_str].transform([[actual_lat]])[0][0]

                    # Use ground-truth link-level PDR from the CSV as the
                    # retraining target.  The simulator applies packet-size
                    # correction and contention on top of the model's
                    # prediction, so training on DTMC window_pdr would
                    # double-count those effects and poison the model.
                    buffers[rat_str].append((seq, lat_norm, actual_pdr_val))

            # 7. Check if retraining is due for this RAT
            if len(buffers[rat_str]) >= retrain_interval and rat_str in api.models:
                retrain_count[rat_str] += 1
                buf = buffers[rat_str]

                x_batch = np.array([b[0] for b in buf])
                y_lat = np.array([b[1] for b in buf])
                y_pdr = np.array([b[2] for b in buf])

                _retrain_model(
                    api, rat_str, x_batch, y_lat, y_pdr,
                    cycle=retrain_count[rat_str],
                    retrain_log=retrain_log,
                    latency_scaler=latency_scalers[rat_str],
                )
                buffers[rat_str] = []

            # 8. Retrieve DTMC state for logging
            dtmc = qsim.dtmc_sizers.get(selected_rat)
            dtmc_state = dtmc.current_state if dtmc else 0
            dtmc_size = dtmc.current_size if dtmc else packet_size

            # 9. Row-level log
            row_log.append({
                "idx": idx,
                "sim_time": round(sim_time, 4),
                "latitude": state.latitude,
                "longitude": state.longitude,
                "selected_rat": selected_rat.value,
                "pred_latency": round(decision.predicted_latency_ms, 4),
                "pred_pdr": round(decision.predicted_pdr, 4),
                "actual_latency": round(_actual_latency(state, selected_rat) or 0.0, 4),
                "actual_pdr": round(_actual_pdr(state, selected_rat) or 0.0, 4),
                "sim_latency": round(sim_latency, 4),
                "delivered": delivered,
                "packet_size": packet_size,
                "corrected_pdr": round(corrected_pdr, 4),
                "dtmc_state": dtmc_state,
                "dtmc_size": dtmc_size,
                "confidence": round(decision.confidence, 4),
                "queue_depth": qsim.queue_depth,
            })

            # Advance simulated time
            sim_time += sim_dt

        # Progress
        if (idx + 1) % 500 == 0:
            pdr_so_far = (
                qsim.metrics.successful_packets / qsim.metrics.total_packets
                if qsim.metrics.total_packets else 0
            )
            print(
                f"  [row {idx+1}/{len(df)}, tx {tx_step}/{total_tx_steps}] "
                f"PDR={pdr_so_far:.3f}  "
                f"RAT switches={rat_switches}  "
                f"retrains={sum(retrain_count.values())}"
            )

    # ----- Save outputs -----

    # Row-level log
    log_path = os.path.join(config.OUTPUT_DIR, "feedback_loop_log.csv")
    pd.DataFrame(row_log).to_csv(log_path, index=False)
    print(f"\nRow log saved to {log_path}")

    # Retraining log
    if retrain_log:
        rt_path = os.path.join(config.OUTPUT_DIR, "feedback_loop_retraining_log.csv")
        pd.DataFrame(retrain_log).to_csv(rt_path, index=False)
        print(f"Retraining log saved to {rt_path}")

    # Save final retrained models
    for rat_str, model in api.models.items():
        if retrain_count.get(rat_str, 0) > 0:
            save_path = os.path.join(
                MODEL_DIR,
                f"retrained_{model_type}_{rat_str}_{int(time.time())}.keras",
            )
            model.save(save_path)
            print(f"Final retrained model saved to {save_path}")

    # DTMC stats
    dtmc_stats = {}
    for rat_enum, dtmc in qsim.dtmc_sizers.items():
        stats = dtmc.get_statistics()
        if stats:
            for k, v in stats.items():
                dtmc_stats[f"dtmc_{rat_enum.value}_{k}"] = v

    # DTMC transition log
    dtmc_rows = []
    for rat_enum, dtmc in qsim.dtmc_sizers.items():
        for step, state, pdr, action in dtmc.history:
            dtmc_rows.append({
                "rat": rat_enum.value,
                "step": step,
                "state": state,
                "packet_size": int(dtmc.size_levels[state]),
                "pdr": round(pdr, 6),
                "action": action,
            })
    if dtmc_rows:
        dtmc_path = os.path.join(config.OUTPUT_DIR, "feedback_dtmc_transitions.csv")
        pd.DataFrame(dtmc_rows).to_csv(dtmc_path, index=False)
        print(f"DTMC transitions saved to {dtmc_path}")

    # Throughput
    duration_s = sim_time if sim_time > 0 else 1.0
    throughput_bps = successful_bytes / duration_s

    # Summary
    total = qsim.metrics.total_packets
    success = qsim.metrics.successful_packets
    achieved_pdr = success / total if total else 0
    log_df = pd.DataFrame(row_log)

    rat_dist = log_df["selected_rat"].value_counts().to_dict() if len(log_df) else {}
    mean_pkt = log_df["packet_size"].mean() if len(log_df) else 0

    summary = {
        "input_file": input_csv,
        "model_type": model_type,
        "seed": seed,
        "total_rows": len(df),
        "processed": total,
        "successful": success,
        "achieved_pdr": round(achieved_pdr, 4),
        "mean_sim_latency": round(log_df["sim_latency"].mean(), 4) if len(log_df) else 0,
        "mean_pred_latency": round(log_df["pred_latency"].mean(), 4) if len(log_df) else 0,
        "successful_bytes": successful_bytes,
        "throughput_bytes_per_s": round(throughput_bps, 4),
        "mean_packet_size": round(mean_pkt, 4),
        "rat_switches": rat_switches,
        "retrain_cycles_total": sum(retrain_count.values()),
        **{f"retrain_{k}": v for k, v in retrain_count.items()},
        **{f"rat_{k}": v for k, v in rat_dist.items()},
        **dtmc_stats,
    }
    summary_path = os.path.join(config.OUTPUT_DIR, "feedback_loop_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # Print summary to stdout
    print("\n" + "=" * 64)
    print("FEEDBACK LOOP RESULTS")
    print("=" * 64)
    print(f"  Total rows processed:  {total}")
    print(f"  Achieved PDR:          {achieved_pdr:.4f}")
    print(f"  Mean sim latency:      {summary['mean_sim_latency']:.2f} ms")
    print(f"  Mean pred latency:     {summary['mean_pred_latency']:.2f} ms")
    print(f"  Throughput:            {throughput_bps:.2f} bytes/s  ({successful_bytes} bytes)")
    print(f"  Mean packet size:      {mean_pkt:.0f} bytes")
    print(f"  RAT switches:          {rat_switches}")
    print(f"  Retrain cycles:        {sum(retrain_count.values())} "
          f"(dsrc={retrain_count['dsrc']}, pc5={retrain_count['pc5']}, 5g={retrain_count['5g']})")
    print(f"  RAT distribution:      {rat_dist}")
    for k, v in dtmc_stats.items():
        print(f"  {k}: {v}")
    print("=" * 64)

    return summary


# ---------------------------------------------------------------------------
# Multi-vehicle loop
# ---------------------------------------------------------------------------

def run_multi_vehicle_loop(
    input_csv: str,
    model_type: str = "lstm",
    seed: Optional[int] = None,
    retrain_interval: int = 500,
    base_packet_size: int = 1024,
    correction_exponent: float = PDR_CORRECTION_EXPONENT,
    num_vehicles: int = 20,
    sim_tx_interval_ms: Optional[int] = None,
    enable_contention: bool = True,
    enable_dtmc: bool = True,
):
    """Run multi-vehicle platoon simulation with optional contention effects.

    Each vehicle has its own feature history, DTMC sizer, and queue simulator.
    Models and retraining buffers are shared globally.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    ensure_dir_exists(config.OUTPUT_DIR)
    ensure_dir_exists(MODEL_DIR)

    tag = ""
    if not enable_contention:
        tag += "_baseline"
    if not enable_dtmc:
        tag += "_no_dtmc"

    df = pd.read_csv(input_csv)
    if tag:
        labels = []
        if not enable_contention:
            labels.append("no contention")
        if not enable_dtmc:
            labels.append("no DTMC")
        print("\n" + "=" * 64)
        print(f"VARIANT pass ({', '.join(labels)})")
        print("=" * 64)
    print(f"Loaded {len(df)} rows from {input_csv}")
    print(f"Simulating {num_vehicles} vehicles\n")

    # Shared components
    api = RATSelectionAPI(model_type=model_type)
    latency_scalers = {
        rat: create_latency_scaler(rat) for rat in ("5g", "pc5", "dsrc")
    }

    # Per-vehicle components
    vehicle_rngs = [np.random.RandomState(seed + v if seed is not None else v)
                    for v in range(num_vehicles)]
    vehicle_qsims = [
        QueueSimulator(
            base_packet_size=base_packet_size,
            correction_exponent=correction_exponent,
            enable_dtmc=enable_dtmc,
        )
        for _ in range(num_vehicles)
    ]
    # Per-vehicle history mirrors the api._history structure
    vehicle_histories: List[Dict[str, list]] = [
        {"dsrc": [], "pc5": [], "5g": []}
        for _ in range(num_vehicles)
    ]

    # Global per-RAT retraining buffers
    buffers: Dict[str, List[Tuple[np.ndarray, float, float]]] = {
        "dsrc": [], "pc5": [], "5g": [],
    }
    retrain_count: Dict[str, int] = {"dsrc": 0, "pc5": 0, "5g": 0}

    # Logging
    row_log: List[Dict] = []
    retrain_log: List[Dict] = []
    contention_log: List[Dict] = []

    # Global TX counters
    total_packets = 0
    successful_packets = 0
    successful_bytes = 0
    # Per-vehicle counters
    vehicle_packets = [0] * num_vehicles
    vehicle_success = [0] * num_vehicles
    vehicle_bytes = [0] * num_vehicles
    vehicle_switches = [0] * num_vehicles
    vehicle_prev_rat: List[Optional[RATType]] = [None] * num_vehicles

    sim_time = 0.0
    sim_dt = (sim_tx_interval_ms / 1000.0) if sim_tx_interval_ms else 0.1

    # Compute repetition factor for TX rate simulation
    if sim_tx_interval_ms is not None:
        data_interval_ms = max(TX_INTERVAL_MS.values())  # 50ms (coarsest CSV rate)
        repeat_factor = data_interval_ms / sim_tx_interval_ms
    else:
        repeat_factor = 1.0

    total_tx_steps = int(len(df) * repeat_factor)
    tx_label = f"{sim_tx_interval_ms}ms (override)" if sim_tx_interval_ms else "per-RAT defaults"
    if repeat_factor != 1.0:
        print(f"TX interval: {tx_label}, repeat factor: {repeat_factor:.3f}, "
              f"effective TX steps: ~{total_tx_steps}")
        if repeat_factor > 20:
            print(f"  WARNING: repeat_factor={repeat_factor:.1f} — simulation will be slow")

    print(f"Starting multi-vehicle feedback loop "
          f"({len(df)} time steps x {num_vehicles} vehicles, "
          f"retrain every {retrain_interval}, TX interval: {tx_label})\n")

    accumulator = 0.0
    tx_step = 0

    for idx in range(len(df)):
        row = df.iloc[idx]

        accumulator += repeat_factor
        n_repeats = int(accumulator)
        accumulator -= n_repeats

        for rep in range(n_repeats):
            tx_step += 1
            base_state = row_to_network_state(row, timestamp_ms=int(sim_time * 1000))

            # Phase 1: All vehicles select RATs
            decisions: List[Optional[RATDecision]] = [None] * num_vehicles
            pkt_decisions: List[Optional[PacketSizeDecision]] = [None] * num_vehicles
            vehicle_rats: List[Optional[RATType]] = [None] * num_vehicles

            # Per-vehicle TX time jitter (±1.0 ms interval) to diversify queue contexts and contention effects
            vehicle_times_ms = [
                int(sim_time * 1000 + vehicle_rngs[v].uniform(-1.0, 1.0))
                for v in range(num_vehicles)
            ]
            vehicle_times_s = [t_ms / 1000.0 for t_ms in vehicle_times_ms]

            for v in range(num_vehicles):
                state_v = _perturb_state(base_state, vehicle_rngs[v])

                # Swap in this vehicle's history (copy lists so api doesn't
                # hold a reference into vehicle_histories)
                api._history = {
                    k: list(v_list) for k, v_list in vehicle_histories[v].items()
                }

                queue_ctx = vehicle_qsims[v].get_queue_context()
                decision = api.select_rat(state_v, queue_ctx)

                # Save updated history back (new dict, decoupled from api)
                vehicle_histories[v] = api._history

                if decision.selected_rat == RATType.UNAVAILABLE:
                    continue

                decisions[v] = decision
                vehicle_rats[v] = decision.selected_rat
                pkt_decisions[v] = vehicle_qsims[v].decide_packet_size(
                    decision, vehicle_times_s[v],
                )

            # Phase 2: Count vehicles per RAT and compute channel utilization
            active_rats = [r for r in vehicle_rats if r is not None]
            vehicles_per_rat = Counter(active_rats)

            # Collect packet sizes per RAT for utilization computation
            packets_per_rat: Dict[RATType, List[int]] = {}
            for v in range(num_vehicles):
                if vehicle_rats[v] is not None and pkt_decisions[v] is not None:
                    packets_per_rat.setdefault(vehicle_rats[v], []).append(
                        pkt_decisions[v].packet_size_bytes,
                    )

            utilization_per_rat: Dict[RATType, float] = {}
            for rat_enum in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
                pkts = packets_per_rat.get(rat_enum, [])
                n = vehicles_per_rat.get(rat_enum, 0)
                utilization_per_rat[rat_enum] = compute_channel_utilization(
                    n, pkts, rat_enum, tx_interval_ms=sim_tx_interval_ms,
                )

            # Phase 2.5: Re-select vehicles on overloaded RATs with contention context
            overloaded_rats = {r for r, u in utilization_per_rat.items() if u > 1.0}
            if enable_contention and overloaded_rats:
                contention_ctx = {
                    rat_enum: (vehicles_per_rat.get(rat_enum, 0),
                               utilization_per_rat[rat_enum])
                    for rat_enum in [RATType.DSRC, RATType.PC5, RATType.FiveG]
                }
                for v in range(num_vehicles):
                    if vehicle_rats[v] not in overloaded_rats:
                        continue
                    state_v = _perturb_state(base_state, vehicle_rngs[v])
                    # Copy history into api for re-selection. select_rat will
                    # append to api._history, but we discard those changes to
                    # avoid double-appending (Pass 1 already appended).
                    api._history = {
                        k: list(v_list) for k, v_list in vehicle_histories[v].items()
                    }
                    queue_ctx = vehicle_qsims[v].get_queue_context()
                    new_decision = api.select_rat(
                        state_v, queue_ctx, contention_context=contention_ctx,
                    )
                    # Discard history changes from re-selection (keep Pass 1 history)
                    if new_decision.selected_rat == RATType.UNAVAILABLE:
                        continue
                    decisions[v] = new_decision
                    vehicle_rats[v] = new_decision.selected_rat
                    pkt_decisions[v] = vehicle_qsims[v].decide_packet_size(
                        new_decision, vehicle_times_s[v],
                    )

                # Recompute utilization after re-selection
                active_rats = [r for r in vehicle_rats if r is not None]
                vehicles_per_rat = Counter(active_rats)
                packets_per_rat = {}
                for v in range(num_vehicles):
                    if vehicle_rats[v] is not None and pkt_decisions[v] is not None:
                        packets_per_rat.setdefault(vehicle_rats[v], []).append(
                            pkt_decisions[v].packet_size_bytes,
                        )
                for rat_enum in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
                    pkts = packets_per_rat.get(rat_enum, [])
                    n = vehicles_per_rat.get(rat_enum, 0)
                    utilization_per_rat[rat_enum] = compute_channel_utilization(
                        n, pkts, rat_enum, tx_interval_ms=sim_tx_interval_ms,
                    )

            # Log contention for this time step
            contention_entry = {"idx": idx, "sim_time": round(sim_time, 4)}
            for rat_enum in [RATType.DSRC, RATType.PC5, RATType.FiveG]:
                rstr = RAT_STR[rat_enum]
                contention_entry[f"n_{rstr}"] = vehicles_per_rat.get(rat_enum, 0)
                contention_entry[f"utilization_{rstr}"] = round(
                    utilization_per_rat[rat_enum], 6,
                )
            contention_log.append(contention_entry)

            # Phase 3: Simulate TX with contention
            for v in range(num_vehicles):
                if decisions[v] is None:
                    continue

                decision = decisions[v]
                selected_rat = decision.selected_rat
                rat_str = RAT_STR[selected_rat]
                pkt_decision = pkt_decisions[v]
                packet_size = pkt_decision.packet_size_bytes
                n_on_rat = vehicles_per_rat[selected_rat]
                state_v = _perturb_state(base_state, vehicle_rngs[v])

                # Track switches
                if vehicle_prev_rat[v] is not None and selected_rat != vehicle_prev_rat[v]:
                    vehicle_switches[v] += 1
                vehicle_prev_rat[v] = selected_rat

                # Simulate TX — use actual PDR from dataset (unperturbed) as ceiling
                actual_pdr_val = _actual_pdr(base_state, selected_rat) or decision.predicted_pdr
                utilization = utilization_per_rat[selected_rat]
                delivered, sim_latency, contended_pdr = _simulate_tx_with_contention(
                    selected_rat, packet_size, actual_pdr_val,
                    base_packet_size, correction_exponent, n_on_rat, utilization,
                )

                # Feed outcome to this vehicle's queue simulator
                outcome = TransmissionOutcome(
                    timestamp_ms=vehicle_times_ms[v],
                    rat_used=selected_rat,
                    packet_size_bytes=packet_size,
                    actual_latency_ms=sim_latency,
                    delivered=delivered,
                    network_state=state_v,
                )
                vehicle_qsims[v].update_pdr_estimate(outcome)
                vehicle_qsims[v].metrics.total_packets += 1
                if delivered:
                    vehicle_qsims[v].metrics.successful_packets += 1

                total_packets += 1
                vehicle_packets[v] += 1
                if delivered:
                    successful_packets += 1
                    successful_bytes += packet_size
                    vehicle_success[v] += 1
                    vehicle_bytes[v] += packet_size

                # Build retraining sequence from this vehicle's history
                history = vehicle_histories[v].get(rat_str, [])
                if len(history) >= TIMESTEPS:
                    seq = np.array(history[-TIMESTEPS:])
                    actual_lat = _actual_latency(state_v, selected_rat)
                    actual_pdr_val = _actual_pdr(state_v, selected_rat)

                    if actual_lat is not None and actual_pdr_val is not None:
                        lat_norm = latency_scalers[rat_str].transform([[actual_lat]])[0][0]
                        # Use ground-truth link-level PDR (see single-vehicle comment)
                        buffers[rat_str].append((seq, lat_norm, actual_pdr_val))

                # DTMC state for logging
                dtmc = vehicle_qsims[v].dtmc_sizers.get(selected_rat)
                dtmc_state = dtmc.current_state if dtmc else 0
                dtmc_size = dtmc.current_size if dtmc else packet_size

                row_log.append({
                    "idx": idx,
                    "vehicle_id": v,
                    "sim_time": round(vehicle_times_s[v], 4),
                    "latitude": state_v.latitude,
                    "longitude": state_v.longitude,
                    "selected_rat": selected_rat.value,
                    "pred_latency": round(decision.predicted_latency_ms, 4),
                    "pred_pdr": round(decision.predicted_pdr, 4),
                    "actual_latency": round(_actual_latency(state_v, selected_rat) or 0.0, 4),
                    "actual_pdr": round(_actual_pdr(state_v, selected_rat) or 0.0, 4),
                    "sim_latency": round(sim_latency, 4),
                    "delivered": delivered,
                    "packet_size": packet_size,
                    "corrected_pdr": round(contended_pdr, 4),
                    "contention_n_vehicles": n_on_rat,
                    "contention_utilization": round(utilization, 6),
                    "dtmc_state": dtmc_state,
                    "dtmc_size": dtmc_size,
                    "confidence": round(decision.confidence, 4),
                    "queue_depth": vehicle_qsims[v].queue_depth,
                })

            # Phase 4: Check retraining threshold (global, per RAT)
            for rat_str_r in ("dsrc", "pc5", "5g"):
                if len(buffers[rat_str_r]) >= retrain_interval and rat_str_r in api.models:
                    retrain_count[rat_str_r] += 1
                    buf = buffers[rat_str_r]
                    x_batch = np.array([b[0] for b in buf])
                    y_lat = np.array([b[1] for b in buf])
                    y_pdr = np.array([b[2] for b in buf])
                    _retrain_model(
                        api, rat_str_r, x_batch, y_lat, y_pdr,
                        cycle=retrain_count[rat_str_r],
                        retrain_log=retrain_log,
                        latency_scaler=latency_scalers[rat_str_r],
                    )
                    buffers[rat_str_r] = []

            sim_time += sim_dt

        # Progress
        if (idx + 1) % 500 == 0:
            pdr_so_far = successful_packets / total_packets if total_packets else 0
            print(
                f"  [row {idx+1}/{len(df)}, tx {tx_step}/{total_tx_steps}] "
                f"PDR={pdr_so_far:.3f}  "
                f"total_tx={total_packets}  "
                f"retrains={sum(retrain_count.values())}"
            )

    # ----- Save outputs -----

    # Row-level log
    log_path = os.path.join(config.OUTPUT_DIR, f"feedback_multi{tag}_log.csv")
    pd.DataFrame(row_log).to_csv(log_path, index=False)
    print(f"\nRow log saved to {log_path}")

    # Retraining log
    if retrain_log:
        rt_path = os.path.join(config.OUTPUT_DIR, f"feedback_multi{tag}_retraining_log.csv")
        pd.DataFrame(retrain_log).to_csv(rt_path, index=False)
        print(f"Retraining log saved to {rt_path}")

    # Save final retrained models (only on contention pass)
    if enable_contention:
        for rat_str, model in api.models.items():
            if retrain_count.get(rat_str, 0) > 0:
                save_path = os.path.join(
                    MODEL_DIR,
                    f"retrained_{model_type}_{rat_str}_{int(time.time())}.keras",
                )
                model.save(save_path)
                print(f"Final retrained model saved to {save_path}")

    # Contention log
    ct_path = os.path.join(config.OUTPUT_DIR, f"feedback_multi{tag}_contention.csv")
    pd.DataFrame(contention_log).to_csv(ct_path, index=False)
    print(f"Contention log saved to {ct_path}")

    # DTMC transition log (all vehicles)
    dtmc_rows = []
    for v in range(num_vehicles):
        for rat_enum, dtmc in vehicle_qsims[v].dtmc_sizers.items():
            for step, state, pdr, action in dtmc.history:
                dtmc_rows.append({
                    "vehicle_id": v,
                    "rat": rat_enum.value,
                    "step": step,
                    "state": state,
                    "packet_size": int(dtmc.size_levels[state]),
                    "pdr": round(pdr, 6),
                    "action": action,
                })
    if dtmc_rows:
        dtmc_path = os.path.join(config.OUTPUT_DIR, f"feedback_multi{tag}_dtmc_transitions.csv")
        pd.DataFrame(dtmc_rows).to_csv(dtmc_path, index=False)
        print(f"DTMC transitions saved to {dtmc_path}")

    # Throughput
    duration_s = sim_time if sim_time > 0 else 1.0
    throughput_bps = successful_bytes / duration_s

    # Per-vehicle stats
    achieved_pdr = successful_packets / total_packets if total_packets else 0
    log_df = pd.DataFrame(row_log)
    rat_dist = log_df["selected_rat"].value_counts().to_dict() if len(log_df) else {}

    per_vehicle_rows = []
    for v in range(num_vehicles):
        vpdr = vehicle_success[v] / vehicle_packets[v] if vehicle_packets[v] else 0
        v_df = log_df[log_df["vehicle_id"] == v] if len(log_df) else pd.DataFrame()
        v_duration = duration_s  # all vehicles share same time window
        v_throughput = vehicle_bytes[v] / v_duration
        per_vehicle_rows.append({
            "vehicle_id": v,
            "total_packets": vehicle_packets[v],
            "successful_packets": vehicle_success[v],
            "achieved_pdr": round(vpdr, 4),
            "mean_sim_latency": round(v_df["sim_latency"].mean(), 4) if len(v_df) else 0,
            "rat_switches": vehicle_switches[v],
            "successful_bytes": vehicle_bytes[v],
            "throughput_bytes_per_s": round(v_throughput, 4),
        })

    # DTMC stats (aggregate across all vehicles)
    dtmc_stats = {}
    for v in range(num_vehicles):
        for rat_enum, dtmc in vehicle_qsims[v].dtmc_sizers.items():
            stats = dtmc.get_statistics()
            if stats:
                for k, val in stats.items():
                    key = f"dtmc_{rat_enum.value}_{k}"
                    dtmc_stats.setdefault(key, [])
                    dtmc_stats[key].append(val)
    dtmc_agg = {k: round(np.mean(v), 4) for k, v in dtmc_stats.items()}

    summary = {
        "input_file": input_csv,
        "model_type": model_type,
        "seed": seed,
        "num_vehicles": num_vehicles,
        "total_rows": len(df),
        "total_tx_steps": tx_step,
        "repeat_factor": round(repeat_factor, 4),
        "total_packets": total_packets,
        "successful_packets": successful_packets,
        "achieved_pdr": round(achieved_pdr, 4),
        "mean_sim_latency": round(log_df["sim_latency"].mean(), 4) if len(log_df) else 0,
        "mean_pred_latency": round(log_df["pred_latency"].mean(), 4) if len(log_df) else 0,
        "successful_bytes": successful_bytes,
        "throughput_bytes_per_s": round(throughput_bps, 4),
        "total_rat_switches": sum(vehicle_switches),
        "retrain_cycles_total": sum(retrain_count.values()),
        **{f"retrain_{k}": v for k, v in retrain_count.items()},
        **{f"rat_{k}": v for k, v in rat_dist.items()},
        **dtmc_agg,
    }

    summary_path = os.path.join(config.OUTPUT_DIR, f"feedback_multi{tag}_summary.csv")
    # Save summary + per-vehicle detail
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    pd.DataFrame(per_vehicle_rows).to_csv(
        os.path.join(config.OUTPUT_DIR, f"feedback_multi{tag}_per_vehicle.csv"), index=False,
    )
    print(f"Summary saved to {summary_path}")

    # Print summary
    print("\n" + "=" * 64)
    mode_parts = []
    mode_parts.append("CONTENTION" if enable_contention else "NO CONTENTION")
    mode_parts.append("DTMC" if enable_dtmc else "NO DTMC")
    mode_label = " + ".join(mode_parts)
    print(f"MULTI-VEHICLE FEEDBACK LOOP RESULTS — {mode_label} ({num_vehicles} vehicles)")
    print("=" * 64)
    print(f"  CSV rows:              {len(df)}")
    print(f"  TX steps:              {tx_step}  (repeat factor: {repeat_factor:.3f})")
    print(f"  Total transmissions:   {total_packets}")
    print(f"  Achieved PDR:          {achieved_pdr:.4f}")
    print(f"  Mean sim latency:      {summary['mean_sim_latency']:.2f} ms")
    print(f"  Mean pred latency:     {summary['mean_pred_latency']:.2f} ms")
    print(f"  Throughput:            {throughput_bps:.2f} bytes/s  ({successful_bytes} bytes)")
    print(f"  Total RAT switches:    {sum(vehicle_switches)}")
    print(f"  Retrain cycles:        {sum(retrain_count.values())} "
          f"(dsrc={retrain_count['dsrc']}, pc5={retrain_count['pc5']}, 5g={retrain_count['5g']})")
    print(f"  RAT distribution:      {rat_dist}")
    for k, v in dtmc_agg.items():
        print(f"  {k}: {v}")
    print("=" * 64)

    # Automatically run variant passes for comparison
    if enable_contention:
        run_multi_vehicle_loop(
            input_csv=input_csv,
            model_type=model_type,
            seed=seed,
            retrain_interval=retrain_interval,
            base_packet_size=base_packet_size,
            correction_exponent=correction_exponent,
            num_vehicles=num_vehicles,
            sim_tx_interval_ms=sim_tx_interval_ms,
            enable_contention=False,
            enable_dtmc=enable_dtmc,
        )
    if enable_dtmc:
        run_multi_vehicle_loop(
            input_csv=input_csv,
            model_type=model_type,
            seed=seed,
            retrain_interval=retrain_interval,
            base_packet_size=base_packet_size,
            correction_exponent=correction_exponent,
            num_vehicles=num_vehicles,
            sim_tx_interval_ms=sim_tx_interval_ms,
            enable_contention=enable_contention,
            enable_dtmc=False,
        )
    if not enable_dtmc:
        run_multi_vehicle_loop(
            input_csv=input_csv,
            model_type=model_type,
            seed=seed,
            retrain_interval=retrain_interval,
            base_packet_size=base_packet_size,
            correction_exponent=correction_exponent,
            num_vehicles=num_vehicles,
            sim_tx_interval_ms=sim_tx_interval_ms,
            enable_contention=False,
            enable_dtmc=False,
        )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop feedback simulation with RAT selection, DTMC, and retraining"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to super_merged CSV")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--retrain_interval", type=int, default=500,
                        help="Samples per RAT before retraining (default: 500)")
    parser.add_argument("--base_packet_size", type=int, default=1024)
    parser.add_argument("--correction_exponent", type=float, default=PDR_CORRECTION_EXPONENT)
    parser.add_argument("--num_vehicles", type=int, default=1,
                        help="Number of vehicles in platoon (default: 1 = single-vehicle mode)")
    parser.add_argument("--tx-interval", type=int, default=None,
                        help="Override TX interval in ms for all RATs (default: per-RAT from config)")
    parser.add_argument("--no-dtmc", action="store_true",
                        help="Disable DTMC adaptive packet sizing (use fixed base_packet_size)")
    args = parser.parse_args()

    if args.num_vehicles > 1:
        run_multi_vehicle_loop(
            input_csv=args.input,
            model_type=args.model_type,
            seed=args.seed,
            retrain_interval=args.retrain_interval,
            base_packet_size=args.base_packet_size,
            correction_exponent=args.correction_exponent,
            num_vehicles=args.num_vehicles,
            sim_tx_interval_ms=args.tx_interval,
            enable_dtmc=not args.no_dtmc,
        )
    else:
        run_feedback_loop(
            input_csv=args.input,
            model_type=args.model_type,
            seed=args.seed,
            retrain_interval=args.retrain_interval,
            base_packet_size=args.base_packet_size,
            correction_exponent=args.correction_exponent,
            sim_tx_interval_ms=args.tx_interval,
            enable_dtmc=not args.no_dtmc,
        )


if __name__ == "__main__":
    main()
