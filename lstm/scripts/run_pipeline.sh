#!/usr/bin/env bash
#
# Full RAT-selection pipeline: data prep -> training -> RAT selection ->
# feedback loop (single or multi-vehicle).
#
# Usage examples:
#
#   # Full pipeline from raw logs (trim + match + train + select + feedback)
#   ./scripts/run_pipeline.sh --raw_data /path/to/raw_logs \
#                             --data /path/to/trimmed_logs \
#                             --model lstm --seed 42
#
#   # Skip trimming, use existing trimmed CSVs (train + select + feedback)
#   ./scripts/run_pipeline.sh --data /path/to/trimmed_logs \
#                             --model lstm --seed 42
#
#   # Use preprocessed NPZ (skip CSV processing) + existing models
#   ./scripts/run_pipeline.sh --npz output/5g_lstm_data.npz \
#                             --model lstm --load existing --seed 42
#
#   # Incremental learning with new data
#   ./scripts/run_pipeline.sh --data /path/to/trimmed_logs \
#                             --new_data /path/to/new_logs \
#                             --model lstm
#
#   # Multi-vehicle feedback loop (20 vehicles)
#   ./scripts/run_pipeline.sh --data /path/to/trimmed_logs \
#                             --model lstm --num_vehicles 20 --seed 42
#
#   # RAT selection only (skip training, use existing models)
#   ./scripts/run_pipeline.sh --data /path/to/trimmed_logs \
#                             --model lstm --skip_training
#
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="lstm"
RATS="5g pc5 dsrc"
SEED=""
NUM_VEHICLES=1
RETRAIN_INTERVAL=500
EPOCHS=""
BASE_PACKET_SIZE=1000
CORRECTION_EXP=0.8
LOAD_MODEL=""                # "" = auto (latest), "none" = train from scratch, path = specific model

RAW_DATA=""                  # Path to raw log folders (triggers trimming)
DATA=""                      # Path to trimmed CSV folder (training data)
NEW_DATA=""                  # Path to new data folder (incremental learning)
NPZ=""                       # Path to preprocessed NPZ archive
MERGED_CSV=""                # Path to super_merged CSV (for feedback loop)

SKIP_TRIMMING=false
SKIP_MATCHING=false
SKIP_TRAINING=false
SKIP_SELECTION=false
SKIP_FEEDBACK=false

# ── Parse arguments ─────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
Usage: run_pipeline.sh [OPTIONS]

Data sources (at least one required):
  --raw_data DIR        Raw log folder (triggers trimming step)
  --data DIR            Trimmed CSV folder (training + matching)
  --new_data DIR        New data folder (incremental learning)
  --npz FILE            Preprocessed NPZ archive (skips CSV processing)
  --merged_csv FILE     Pre-built super_merged CSV (skips selection, goes to feedback)

Model options:
  --model TYPE          Model architecture: lstm, gru, rnn (default: lstm)
  --rats "r1 r2 ..."    RATs to train: "5g pc5 dsrc" (default: all three)
  --load MODE           Model loading: "none" = train fresh, "existing" = latest,
                        or a path to a specific .keras file (default: auto-detect)
  --epochs N            Training epochs (default: 100)

Feedback loop options:
  --num_vehicles N      Number of vehicles (default: 1 = single-vehicle)
  --retrain_interval N  Samples before retraining (default: 500)
  --seed N              Random seed for reproducibility
  --base_packet_size N  Base packet size in bytes (default: 1024)
  --correction_exp F    PDR correction exponent (default: 0.8)

Skip stages:
  --skip_trimming       Skip raw log trimming
  --skip_matching       Skip GPS-based cross-RAT matching
  --skip_training       Skip model training (use existing models)
  --skip_selection      Skip RAT selection (go straight to feedback)
  --skip_feedback       Skip feedback loop

General:
  -h, --help            Show this help message
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw_data)         RAW_DATA="$2";          shift 2 ;;
        --data)             DATA="$2";              shift 2 ;;
        --new_data)         NEW_DATA="$2";          shift 2 ;;
        --npz)              NPZ="$2";               shift 2 ;;
        --merged_csv)       MERGED_CSV="$2";        shift 2 ;;
        --model)            MODEL="$2";             shift 2 ;;
        --rats)             RATS="$2";              shift 2 ;;
        --load)             LOAD_MODEL="$2";        shift 2 ;;
        --epochs)           EPOCHS="$2";            shift 2 ;;
        --num_vehicles)     NUM_VEHICLES="$2";      shift 2 ;;
        --retrain_interval) RETRAIN_INTERVAL="$2";  shift 2 ;;
        --seed)             SEED="$2";              shift 2 ;;
        --base_packet_size) BASE_PACKET_SIZE="$2";  shift 2 ;;
        --correction_exp)   CORRECTION_EXP="$2";    shift 2 ;;
        --skip_trimming)    SKIP_TRIMMING=true;     shift ;;
        --skip_matching)    SKIP_MATCHING=true;     shift ;;
        --skip_training)    SKIP_TRAINING=true;     shift ;;
        --skip_selection)   SKIP_SELECTION=true;    shift ;;
        --skip_feedback)    SKIP_FEEDBACK=true;     shift ;;
        -h|--help)          usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Validation ──────────────────────────────────────────────────────────────
if [[ -z "$RAW_DATA" && -z "$DATA" && -z "$NPZ" && -z "$MERGED_CSV" ]]; then
    echo "Error: provide at least one of --raw_data, --data, --npz, or --merged_csv"
    exit 1
fi

cd "$PROJECT_DIR"

banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    echo ""
}

# ── Stage 1: Trim raw logs ─────────────────────────────────────────────────
if [[ -n "$RAW_DATA" ]] && [[ "$SKIP_TRIMMING" == false ]]; then
    banner "Stage 1: Trimming raw logs"
    python -m scripts.trimmers --folder "$RAW_DATA"
    # If --data not explicitly set, use the same folder (trimmed CSVs land there)
    if [[ -z "$DATA" ]]; then
        DATA="$RAW_DATA"
    fi
else
    echo "-- Skipping trimming (no --raw_data or --skip_trimming)"
fi

# ── Stage 2: GPS-match cross-RAT data ──────────────────────────────────────
if [[ -n "$DATA" ]] && [[ "$SKIP_MATCHING" == false ]]; then
    # Only run matching if ALL required matched files exist
    all_matched=true
    for f in matched_5g.csv matched_dsrc.csv matched_pc5.csv super.csv; do
        if [[ ! -f "$DATA/$f" ]]; then
            all_matched=false
            break
        fi
    done

    if [[ "$all_matched" == false ]]; then
        banner "Stage 2: Matching cross-RAT data by GPS"
        python -m scripts.prepare_data --input "$DATA"
    else
        echo "-- All matched CSVs present in $DATA, skipping matching"
    fi
else
    echo "-- Skipping matching (no --data or --skip_matching)"
fi

# ── Stage 3: Train models ──────────────────────────────────────────────────
if [[ "$SKIP_TRAINING" == false ]]; then
    banner "Stage 3: Training models ($MODEL for: $RATS)"

    for rat in $RATS; do
        echo "---- Training $MODEL for $rat ----"

        CMD=(python -m learning.main --rat "$rat" --model "$MODEL")

        # Data source
        if [[ -n "$NPZ" ]]; then
            # NPZ is per-RAT; construct the expected path
            npz_dir="$(dirname "$NPZ")"
            rat_npz="$npz_dir/${rat}_lstm_data.npz"
            if [[ -f "$rat_npz" ]]; then
                CMD+=(--npz "$rat_npz")
            else
                echo "  Warning: $rat_npz not found, falling back to CSV"
                [[ -n "$DATA" ]] && CMD+=(--data "$DATA") || { echo "  Error: no data for $rat"; continue; }
            fi
        elif [[ -n "$DATA" ]]; then
            CMD+=(--data "$DATA")
        else
            echo "Warning: no training data source for $rat, skipping"
            continue
        fi

        # Incremental learning
        [[ -n "$NEW_DATA" ]] && CMD+=(--new_data "$NEW_DATA")

        # Model loading strategy
        if [[ -n "$LOAD_MODEL" && "$LOAD_MODEL" != "existing" ]]; then
            CMD+=(--load "$LOAD_MODEL")
        fi

        # Epochs override
        [[ -n "$EPOCHS" ]] && CMD+=(--epochs "$EPOCHS")

        echo "  > ${CMD[*]}"
        "${CMD[@]}"
        echo ""
    done
else
    echo "-- Skipping training (--skip_training)"
fi

# ── Stage 4: RAT selection + super CSV ──────────────────────────────────────
if [[ "$SKIP_SELECTION" == false ]] && [[ -n "$DATA" ]]; then
    banner "Stage 4: RAT selection on matched data"
    python -m selection.rat_selection --input "$DATA" --model_type "$MODEL"
else
    echo "-- Skipping selection (--skip_selection or no --data)"
fi

# Look for merged CSV for feedback loop (prefer super_merged which has signal columns)
if [[ -z "$MERGED_CSV" ]] && [[ -n "$DATA" ]]; then
    if [[ -f "$DATA/super_merged.csv" ]]; then
        MERGED_CSV="$DATA/super_merged.csv"
    fi
fi

# ── Stage 5: Feedback loop ──────────────────────────────────────────────────
if [[ "$SKIP_FEEDBACK" == false ]]; then
    if [[ -z "$MERGED_CSV" ]]; then
        echo "-- Skipping feedback loop: no super_merged CSV found"
        echo "   Provide --merged_csv or ensure RAT selection produced one"
    else
        if [[ "$NUM_VEHICLES" -gt 1 ]]; then
            banner "Stage 5: Multi-vehicle feedback loop ($NUM_VEHICLES vehicles)"
        else
            banner "Stage 5: Feedback loop (single vehicle)"
        fi

        CMD=(python -m scripts.feedback_loop
            --input "$MERGED_CSV"
            --model_type "$MODEL"
            --retrain_interval "$RETRAIN_INTERVAL"
            --base_packet_size "$BASE_PACKET_SIZE"
            --correction_exponent "$CORRECTION_EXP"
            --num_vehicles "$NUM_VEHICLES"
        )
        [[ -n "$SEED" ]] && CMD+=(--seed "$SEED")

        echo "  > ${CMD[*]}"
        "${CMD[@]}"
    fi
else
    echo "-- Skipping feedback loop (--skip_feedback)"
fi

# ── Done ────────────────────────────────────────────────────────────────────
banner "Pipeline complete"
echo "Output files are in: $PROJECT_DIR/output/"
echo "Models are in:       $PROJECT_DIR/models/"
if [[ -n "$MERGED_CSV" ]]; then
    echo "Merged CSV used:     $MERGED_CSV"
fi
