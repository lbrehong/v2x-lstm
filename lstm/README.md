# RAT Performance Prediction for Vehicular Networks

A machine learning framework for predicting network Quality of Service (QoS) metrics and enabling intelligent Radio Access Technology (RAT) selection in vehicular communication systems.

## Overview

This project implements LSTM, GRU, and SimpleRNN models to predict latency and Packet Delivery Rate (PDR) for three Radio Access Technologies:

- **5G SA** (5G Standalone)
- **PC5/C-V2X** (Cellular Vehicle-to-Everything sidelink)
- **DSRC** (Dedicated Short-Range Communications / 802.11p)

The predictions enable proactive RAT selection, allowing vehicles to switch to the optimal network technology before QoS degrades, rather than reacting after failures occur.

## Features

- **Dual-output neural networks**: Simultaneous prediction of latency and PDR
- **Multiple architectures**: LSTM, GRU, and SimpleRNN with identical interfaces
- **Incremental learning**: Online model updates as new data arrives
- **Clock drift compensation**: Automatic correction for unsynchronized clocks in 5G measurements
- **GPS-based data alignment**: Cross-RAT matching for fair performance comparison
- **Interactive visualization**: Folium maps showing RAT selection along vehicle routes
- **Comprehensive statistics**: Latency/PDR histograms, RMSE tables, confidence intervals

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

Install required packages:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib tqdm folium scipy tabulate python-dotenv simpy
```

For CPU-only TensorFlow (recommended for most setups):

```bash
pip install tensorflow-cpu
```

### Clone and Setup

```bash
git clone <repository-url>
cd lstm
```

## Project Structure

```
lstm/
|-- config.py                      # Centralized configuration: bounds, hyperparameters, scalers
|-- api_types.py                   # Data structures for queue simulator integration
|-- utils.py                       # Utility functions: file discovery, model loading
|
|-- learning/                      # ML training pipeline
|   |-- main.py                    # Training entry point, MetricsLogger callback, incremental learning
|   |-- model.py                   # Network architectures, dual-output heads, DataStreamGenerator
|   `-- data_preprocessing.py      # PDR rolling computation, MinMaxScaler, sequence generation
|
|-- queuesim/                      # Queue simulator + PHY layer
|   |-- queue_simulator.py         # SimPy-based queue simulator with DTMC packet sizing
|   |-- phy_layer.py               # PHY-layer capacity calculations per RAT
|   |-- dtmc_sizer.py              # DTMC adaptive packet sizer
|   `-- sim_types.py               # TransmissionRecord, SimulationMetrics
|
|-- selection/                     # RAT selection API + integration
|   |-- api.py                     # RATSelectionAPI and JointController classes
|   |-- rat_selection.py           # RAT selection algorithms, Folium visualization, statistics
|   `-- file_integration.py        # File-based integration for batch/interleaved processing
|
|-- run_pipeline.py                # End-to-end pipeline runner (direct Python imports, no subprocesses)
|
|-- scripts/                       # Data prep & visualization CLI tools
|   |-- feedback_loop.py           # Feedback loop with incremental retraining
|   |-- trimmers.py                # Raw V2X/5G log processing, latency drift compensation
|   |-- prepare_data.py            # Cross-RAT data matching by GPS coordinates
|   `-- plot_history.py            # Training history visualization
|
|-- legacy/                        # Deprecated files (archived)
|   |-- trimmers.py                # Old trimmer version
|   `-- linaer.py                  # One-off drift compensation script
|
|-- tests/                         # Unit tests for all modules
|-- models/                        # Saved Keras models (.keras files)
|-- output/                        # Prediction logs, training logs, visualizations
`-- CLAUDE.md                      # AI assistant guidelines
```

## Usage

### Full Pipeline (Recommended)

The `run_pipeline.py` script orchestrates all stages (trim, match, train, select, feedback) in a single command. This is the recommended way to run the project end-to-end.

```bash
# Full pipeline from raw logs
python -m run_pipeline --raw_data /path/to/raw_logs \
                               --data /path/to/trimmed_logs \
                               --model lstm --seed 42

# Skip trimming, start from existing trimmed CSVs
python -m run_pipeline --data /path/to/trimmed_logs \
                               --model lstm --seed 42

# Use preprocessed NPZ files with existing models
python -m run_pipeline --npz output/5g_lstm_data.npz \
                               --model lstm --load existing --seed 42

# Incremental learning with new field data
python -m run_pipeline --data /path/to/trimmed_logs \
                               --new_data /path/to/new_logs \
                               --model lstm

# Multi-vehicle feedback loop (20 vehicles)
python -m run_pipeline --data /path/to/trimmed_logs \
                               --model lstm --num_vehicles 20 --seed 42

# RAT selection only (skip training, use existing models)
python -m run_pipeline --data /path/to/trimmed_logs \
                               --model lstm --skip_training
```

**Key options:**

| Argument | Description |
|----------|-------------|
| `--raw_data` | Raw log folder (triggers trimming step) |
| `--data` | Trimmed CSV folder (training + matching) |
| `--npz` | Preprocessed NPZ archive (skips CSV processing) |
| `--new_data` | New data folder for incremental learning |
| `--model` | Architecture: `lstm`, `gru`, or `rnn` (default: `lstm`) |
| `--rats` | RATs to train (default: `5g pc5 dsrc`) |
| `--num_vehicles` | Number of vehicles for feedback loop (default: 1) |
| `--seed` | Random seed for reproducibility |
| `--skip_trimming` | Skip the trimming stage |
| `--skip_matching` | Skip GPS matching stage |
| `--skip_training` | Skip model training stage |
| `--skip_selection` | Skip RAT selection stage |
| `--skip_feedback` | Skip feedback loop stage |

### Individual Pipeline Stages

The stages below can also be run independently. See `doc/PIPELINE.md` for detailed documentation of each stage.

### Data Preprocessing

#### Step 1: Trim Raw Logs

Process raw V2X and 5G log files into standardized CSVs:

```bash
python -m scripts.trimmers --folder /path/to/raw_logs
```

This generates `trim_5g.csv`, `trim_pc5.csv`, and `trim_dsrc.csv`.

#### Step 2: Match Cross-RAT Data

Align data from different RATs by GPS coordinates:

```bash
python -m scripts.prepare_data --input /path/to/trimmed_data
```

Outputs `matched_5g.csv`, `matched_pc5.csv`, `matched_dsrc.csv`, and `super.csv`.

### Model Training

#### Train New Models

Train LSTM, GRU, and RNN models for a specific RAT:

```bash
python -m learning.main --rat 5g --model lstm --data /path/to/training_logs --new_data /path/to/new_logs
```

#### Using Preprocessed Data

Skip CSV processing by loading from NPZ archive:

```bash
python -m learning.main --rat pc5 --model gru --npz /path/to/pc5_lstm_data.npz
```

#### Training Options

| Argument | Description |
|----------|-------------|
| `--rat` | RAT type: `5g`, `pc5`, or `dsrc` (required) |
| `--model` | Architecture: `lstm`, `gru`, or `rnn` (required) |
| `--data` | Path to training CSV folder |
| `--new_data` | Path to new data for incremental learning |
| `--npz` | Path to preprocessed NPZ archive |
| `--load` | Model path to load, or `none` to train fresh |
| `--epochs` | Number of training epochs (default: 100) |

### RAT Selection and Analysis

#### Generate Predictions for All RATs

```bash
python -m selection.rat_selection --input /path/to/matched_data --model_type all
```

This runs predictions for all model types across all RATs and generates `bestRAT_super.csv`.

#### Visualize RAT Selection on Map

```bash
python -m selection.rat_selection --input /path/to/bestRAT_super.csv --mode view
```

Generates interactive HTML maps showing which RAT was selected at each location.

#### Generate Statistics and Plots

```bash
python -m selection.rat_selection --input /path/to/bestRAT_super.csv --mode data
```

Outputs:
- Latency and PDR histograms
- RMSE comparison tables
- Summary statistics with confidence intervals

### Visualize Training History

```bash
python -m scripts.plot_history
# Enter RAT when prompted: 5g, pc5, or dsrc
```

## Configuration

Key parameters in `config.py`:

### Training Hyperparameters

```python
TIMESTEPS = 10              # Input sequence length
EPOCHS = 100                # Maximum training epochs
BATCH_SIZE = 32             # Training batch size
VALIDATION_SPLIT = 0.15     # Fraction for validation
EARLY_STOPPING_PATIENCE = 5 # Epochs before early stop
```

### Data Collection Parameters

```python
TX_INTERVAL_MS = 100        # Packet transmission interval (ms)
PDR_WINDOW = 10             # Rolling PDR window (seconds)
```

### RAT Selection Thresholds

```python
PDR_RELIABILITY_THRESHOLD = 0.99   # Minimum PDR for reliable RAT
PDR_AVAILABILITY_THRESHOLD = 0.1   # Minimum PDR to consider available
LATENCY_TIE_MARGIN_MS = 1.0        # Margin for tie-breaking
```

### GPS Bounds (Toulouse, France Test Area)

```python
MIN_LAT, MAX_LAT = 43.554669, 43.568290
MIN_LON, MAX_LON = 1.463952, 1.472176
```

## Model Architecture (`learning/model.py`)

The dual-output architecture predicts both latency and PDR simultaneously:

```
Input (batch, 10 timesteps, n_features)
    |
    v
RNN Layer (64 units) - LSTM/GRU/SimpleRNN
    |
    v
Dense (32 units, ReLU activation)
    |
    +---> Dense (1) --> latency_ms output
    |
    +---> Dense (1) --> pdr output
```

**Loss function**: Weighted MSE with `latency=1.0`, `pdr=1.5` (PDR weighted higher for reliability)

### RAT-Specific Features

| RAT  | Features | Count |
|------|----------|-------|
| 5G   | lat, lon, latency, SINR, RSRP, PDR | 6 |
| PC5  | lat, lon, latency, PDR | 4 |
| DSRC | lat, lon, RSRP1, RSRP2, latency, PDR | 6 |

## Output Files

### Models

- Initial: `models/{lstm|gru|rnn}_{5g|pc5|dsrc}_{timestamp}.keras`
- Retrained: `models/retrained_{model}_{rat}_{timestamp}.keras`

### Training Logs

- Epoch metrics: `output/{model}_{rat}_training_log.csv`
- Training history: `output/{model}_{rat}_training_history.json`

### Predictions

- Prediction log: `output/prediction_log_{model}_{rat}.csv`
- Final predictions: `output/final_log_{model}_{rat}.csv`

### Preprocessed Data

- NPZ archive: `output/{rat}_lstm_data.npz`

### Visualizations

- RAT selection maps: `output/rat_map_{model}.html`
- Histograms: `output/latency_histogram.png`, `output/pdr_histogram.png`
- RMSE plots: `output/rmse_pred_plot.png`

## RAT Selection Algorithm

The predictive QoS-based algorithm:

1. **Reliability filter**: Select RATs with predicted PDR >= 99%
2. **Latency optimization**: Among reliable RATs, choose lowest latency
3. **Tie-breaking**: If latencies within 1ms, prefer 5G > PC5 > DSRC
4. **Fallback**: If no RAT meets threshold, select highest actual PDR

The opportunistic (baseline) algorithm uses current measurements with sticky switching to reduce handovers.

## Data Pipeline

```
Raw Logs (5G, PC5, DSRC)
         |
         v
scripts/trimmers.py     -->  Trimmed CSVs (drift-compensated, GPS-enriched)
         |
         v
scripts/prepare_data.py -->  Matched CSVs (aligned by GPS)
         |
         v
learning/main.py        -->  LSTM Sequences (normalized, windowed)
         |
         v
Training/Inference      -->  Predictions + Retrained Models
         |
         v
selection/rat_selection.py -->  RAT Selection + Visualization + Statistics
```

