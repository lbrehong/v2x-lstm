# End-to-End Pipeline Guide

This project has 6 stages forming a closed loop: raw logs, trimmed data, matched data, trained models, RAT selection, queue simulation, and feedback/retraining.

```
Raw Logs --> Trim --> Match by GPS --> Train Models
                                           |
                                           v
             +--- Network State --> RAT Selection API
             |                          |
             |                    Predict latency/PDR
             |                    Select best RAT
             |                          |
             |                          v
             |                    Queue Simulator
             |                    DTMC sizes packet
             |                    PHY checks capacity
             |                          |
             |                          v
             |                     Transmit
             |                          |
             |                          v
             |                  Actual Outcome
             |                   /          \
             |          DTMC updates      Model retrains
             |          window PDR        every 500 samples
             |               \              /
             +---------- Next transmission <-+
```

---

## Running the Full Pipeline

The `run_pipeline.py` script (at project root) orchestrates all stages in a single invocation. It uses direct Python function imports rather than subprocess calls, so all stages run in the same process with shared state.

```bash
# Full pipeline from raw logs (all 6 stages)
python -m run_pipeline --raw_data /path/to/raw_logs \
                               --data /path/to/trimmed_logs \
                               --model lstm --seed 42

# Start from trimmed CSVs (stages 2-6)
python -m run_pipeline --data /path/to/trimmed_logs \
                               --model lstm --seed 42

# Use preprocessed NPZ + existing models (stages 4-6)
python -m run_pipeline --npz output/5g_lstm_data.npz \
                               --model lstm --load existing --seed 42

# Incremental learning with new field data
python -m run_pipeline --data /path/to/trimmed_logs \
                               --new_data /path/to/new_logs \
                               --model lstm

# Multi-vehicle feedback loop (20 vehicles)
python -m run_pipeline --data /path/to/trimmed_logs \
                               --model lstm --num_vehicles 20 --seed 42
```

Individual stages can be skipped with `--skip_trimming`, `--skip_matching`, `--skip_training`, `--skip_selection`, or `--skip_feedback`. At least one data source (`--raw_data`, `--data`, `--npz`, or `--merged_csv`) must be provided.

Run `python -m run_pipeline --help` for the full argument reference.

The rest of this document describes each stage in detail, which is useful both for understanding the pipeline runner and for running stages individually.

---

## Stage 1: Trim Raw Logs

Raw vehicular network logs (5G, PC5, DSRC) contain noisy measurements with clock drift and inconsistent formats. The trimmer standardizes them.

```bash
python -m scripts.trimmers --folder /path/to/raw_logs
```

**What it does:**

- Finds files matching `5g`, `pc5`, `dsrc` patterns in the folder
- **5G**: two-pass -- first computes clock drift coefficient via linear regression on latency vs. sequence number, then compensates and rescales latency to 16-45 ms
- **PC5**: extracts latency, seq num, timestamp, GPS from fixed field positions
- **DSRC**: extracts RSRP values; GPS is matched from PC5/5G data (exact timestamp, then +/-1s window)

**Outputs:** `trim_5g.csv`, `trim_pc5.csv`, `trim_dsrc.csv` with standardized columns:

```
tx_seq_num, tx_timestamp_ms, tx_latitude, tx_longitude, latency_ms, [sinr, rsrp, rsrp_1, rsrp_2]
```

---

## Stage 2: Match Data Across RATs by GPS

Since each RAT samples independently, measurements must be aligned spatially so the model can learn to compare them at the same location.

```bash
python -m scripts.prepare_data --input /path/to/trimmed_data
```

**What it does:**

- Uses 5G as the primary reference (highest sampling rate)
- Computes rolling PDR (10s window) for each RAT
- Removes outlier latencies (<16 ms for 5G, <300 ms for secondary)
- Deduplicates GPS points, then matches PC5/DSRC to 5G locations within ~1 m tolerance (progressively relaxed up to 50x if needed)

**Outputs:**

- `matched_5g.csv`, `matched_pc5.csv`, `matched_dsrc.csv` -- aligned by GPS
- `super.csv` -- combined reference with `tx_latitude, tx_longitude, latency_ms_5g, pdr_5g`

---

## Stage 3: Train Models

Train one model per RAT. Each model is a dual-output RNN predicting both latency and PDR simultaneously.

```bash
# Train all 3 architectures for each RAT
python -m learning.main --rat 5g --model lstm --data /path/to/matched_data
python -m learning.main --rat 5g --model gru  --data /path/to/matched_data
python -m learning.main --rat 5g --model rnn  --data /path/to/matched_data

# Repeat for pc5 and dsrc
python -m learning.main --rat pc5  --model lstm --data /path/to/matched_data
python -m learning.main --rat dsrc --model lstm --data /path/to/matched_data
```

**What happens internally:**

1. Finds `trim_{rat}*.csv` files, loads and concatenates them
2. Interpolates GPS gaps, computes rolling PDR
3. Normalizes features using fixed-bound scalers (not data-driven):
   - GPS: `[43.5547, 43.5683]` lat, `[1.4640, 1.4722]` lon
   - Latency: `[4, 50]` ms to `[0, 1]`
   - PDR: `[0, 1]` (identity)
   - SINR (5G): `[200, 375]`, RSRP (5G): `[-127, -67]`, RSRP (DSRC): `[-150, -45]`
4. Generates sliding-window sequences of length 10
5. Trains model architecture:
   ```
   Input(10 timesteps, n_features)
       -> RNN(64 units)
       -> Dense(32, ReLU)
       -> latency_ms output (1, linear)
       -> pdr output (1, linear)
   ```
   Loss: MSE with weights `{latency: 1.0, pdr: 1.5}`. Early stopping patience=5.
6. Caches preprocessed data as `{rat}_lstm_data.npz` (use `--npz` next time to skip preprocessing)

### RAT-specific feature sets

| RAT  | Features (n) |
|------|-------------|
| 5G   | lat, lon, latency, SINR, RSRP, PDR (6) |
| PC5  | lat, lon, latency, PDR (4) |
| DSRC | lat, lon, RSRP1, RSRP2, latency, PDR (6) |

### Outputs

- `models/{lstm|gru|rnn}_{5g|pc5|dsrc}_{timestamp}.keras`
- `{model}_{rat}_training_log.csv` (per-epoch losses)
- `{model}_{rat}_training_history.json`
- `{rat}_lstm_data.npz`

---

## Stage 4: RAT Selection (Predict QoS + Choose Best RAT)

With trained models, you can predict latency/PDR for all three RATs at a given location and select the best one.

### Option A: Batch mode via CLI (for offline analysis)

```bash
python -m selection.file_integration --mode batch \
  --input /path/to/super_merged.csv \
  --output rat_decisions.csv \
  --model_type lstm
```

### Option B: Programmatic API (for real-time or simulator coupling)

```python
from selection.api import RATSelectionAPI
from api_types import NetworkState

api = RATSelectionAPI(model_type="lstm")

state = NetworkState(
    timestamp_ms=1000,
    latitude=43.56, longitude=1.47,
    fiveg_latency_ms=25.0, fiveg_sinr=300.0, fiveg_rsrp=-80.0, fiveg_pdr=0.98,
    pc5_latency_ms=18.0, pc5_pdr=0.95,
    dsrc_latency_ms=12.0, dsrc_rsrp_1=-60.0, dsrc_rsrp_2=-55.0, dsrc_pdr=0.99,
)

decision = api.select_rat(state)
print(decision.selected_rat)          # e.g., RATType.DSRC
print(decision.predicted_latency_ms)  # e.g., 11.3
print(decision.predicted_pdr)         # e.g., 0.993
print(decision.all_predictions)       # predictions for all 3 RATs
```

### Selection algorithm

1. Predict latency + PDR for each RAT using the trained models
2. Filter RATs where predicted PDR >= 0.99 (reliability threshold)
3. Among reliable RATs, pick the one with lowest latency
4. Tie-breaking (within 1 ms margin): prefer 5G > PC5 > DSRC
5. If no RAT meets threshold: fall back to highest PDR, then 5G, then UNAVAILABLE
6. Compute confidence score (50% PDR margin + 30% latency advantage + 20% availability)

### Output CSV columns

```
timestamp_ms, latitude, longitude, selected_rat,
pred_latency, pred_pdr, confidence,
pred_latency_5g, pred_pdr_5g,
pred_latency_pc5, pred_pdr_pc5,
pred_latency_dsrc, pred_pdr_dsrc,
recommended_max_packet_size
```

---

## Stage 5: Queue Simulation + DTMC Packet Sizing

The queue simulator takes RAT decisions and simulates realistic packet transmission with adaptive packet sizing.

### Option A: CLI (standalone simulation)

```bash
python -m queuesim.queue_simulator --input rat_decisions.csv --output sim_results.csv

# Add --no-phy-limits to disable PHY constraints for comparison
```

### Option B: Programmatic (coupled with RAT selector)

```python
from selection.api import RATSelectionAPI, JointController
from queuesim.queue_simulator import QueueSimulator

api = RATSelectionAPI(model_type="lstm")
qsim = QueueSimulator(base_packet_size=1000, enforce_phy_limits=True)
controller = JointController(rat_selector=api, queue_sim=qsim)

# For each network measurement:
rat_decision, packet_decision = controller.process_transmission(state)

print(rat_decision.selected_rat)           # Which RAT to use
print(packet_decision.packet_size_bytes)   # How big the packet should be
print(packet_decision.priority_level)      # Urgency-based priority
```

### DTMC packet sizer

The sizer maintains 4 packet size levels: `[1024, 2048, 3072, 4096]` bytes. It transitions between states based on a moving-window PDR:

| PDR observed | Action |
|---|---|
| > 0.99 | Increase packet size (move up one level) |
| < 0.95 | Decrease packet size (move down one level) |
| 0.95 - 0.99 | Stay at current size |

**PDR correction for packet size:** Since models were trained on ~1 kB packets, predictions must be adjusted when the DTMC selects a different size:

```
PDR(target_size) = PDR_base ^ ((target_size / 1000) ^ 0.8)
```

Larger packets lead to exponentially lower PDR. This correction feeds back into the selection algorithm.

### PHY-layer constraints (enabled by default)

| RAT | Max TX per 20ms | Queue capacity | Base latency |
|-----|-----------------|----------------|-------------|
| 5G NR (20 MHz) | ~40 KB | ~80 KB | 15 ms |
| PC5 (10 MHz) | ~6 KB | ~12 KB | 8 ms |
| DSRC (10 MHz) | ~15 KB | ~30 KB | 5 ms |

Packets exceeding TX capacity are dropped. Queue overflow also causes drops.

---

## Stage 6: Feedback Loop (Incremental Retraining)

After transmission, actual outcomes feed back into both the DTMC sizer and the ML models.

### DTMC feedback (automatic, per-packet)

```python
from api_types import TransmissionOutcome, NetworkState

outcome = TransmissionOutcome(
    timestamp_ms=1500,
    rat_used=RATType.DSRC,
    packet_size_bytes=2048,
    actual_latency_ms=14.2,
    delivered=True,
    network_state=state,
)

# Update DTMC's moving-window PDR estimate
qsim.update_pdr_estimate(outcome)

# Update RAT selector's outcome buffer (retrains every 500 samples)
api.report_outcome(outcome)
```

The DTMC immediately uses the new delivery result in its next transition decision. The RAT selector buffers outcomes and triggers incremental retraining every 500 samples.

### Batch incremental learning (offline, with new data)

```bash
python -m learning.main --rat 5g --model lstm \
  --data /path/to/original_data \
  --new_data /path/to/new_field_data
```

**What `automatic_train` does:**

1. Splits new data: 85% training, 15% validation
2. Streams through training data in chunks of N=500
3. For each chunk: batch predict, compute MAE + squared error, retrain 1 epoch
4. Logs predictions to `prediction_log_{model}_{rat}.csv`
5. Saves retrained model as `models/retrained_{model}_{rat}_{timestamp}.keras`

---

## Interleaved File Exchange (Phase 2)

If the RAT selector and queue simulator run as separate processes:

```bash
python -m selection.file_integration --mode exchange \
  --input data.csv \
  --exchange_dir /tmp/exchange \
  --output log.csv
```

This uses file-based polling: `network_state.csv` -> `queue_context.csv` -> `rat_decision.csv`, synchronized via `sync.txt`.
