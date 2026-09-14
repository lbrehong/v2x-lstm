"""
Full RAT-selection pipeline: data prep -> training -> RAT selection ->
feedback loop (single or multi-vehicle).

Usage examples:

    # Full pipeline from raw logs (trim + match + train + select + feedback)
    python -m run_pipeline --raw_data /path/to/raw_logs --model lstm --seed 42

    # Same, but write trimmed output to a separate folder
    python -m run_pipeline --raw_data /path/to/raw_logs \
                           --data /path/to/trimmed_output \
                           --model lstm --seed 42

    # Skip trimming, use existing trimmed CSVs (train + select + feedback)
    python -m run_pipeline --data /path/to/trimmed_logs \
                           --model lstm --seed 42

    # Use preprocessed NPZ (skip CSV processing) + existing models
    python -m run_pipeline --npz output/5g_lstm_data.npz \
                           --model lstm --load existing --seed 42

    # Incremental learning with new data
    python -m run_pipeline --data /path/to/trimmed_logs \
                           --new_data /path/to/new_logs \
                           --model lstm

    # Multi-vehicle feedback loop (20 vehicles)
    python -m run_pipeline --data /path/to/trimmed_logs \
                           --model lstm --num_vehicles 20 --seed 42

    # RAT selection only (skip training, use existing models)
    python -m run_pipeline --data /path/to/trimmed_logs \
                           --model lstm --skip_training
"""
import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path for direct imports
_PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_DIR))

import config


def banner(msg: str) -> None:
    print()
    print("=" * 64)
    print(f"  {msg}")
    print("=" * 64)
    print()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end RAT-selection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data = p.add_argument_group("Data sources (at least one required)")
    data.add_argument("--raw_data", default="",
                      help="Raw log folder (triggers trimming step)")
    data.add_argument("--data", default="",
                      help="Trimmed CSV folder (training + matching). "
                           "When used with --raw_data, trimmed output is "
                           "written here instead of into the raw folder")
    data.add_argument("--new_data", default="",
                      help="New data folder (incremental learning)")
    data.add_argument("--npz", default="",
                      help="Preprocessed NPZ archive (skips CSV processing)")
    data.add_argument("--merged_csv", default="",
                      help="Pre-built super_merged CSV (skips selection, goes to feedback)")

    model = p.add_argument_group("Model options")
    model.add_argument("--model", default="lstm",
                       choices=["lstm", "gru", "rnn"],
                       help="Model architecture (default: lstm)")
    model.add_argument("--rats", nargs="+", default=list(config.DEFAULT_RATS),
                       choices=config.ALL_RATS,
                       help="RATs involved in every stage: matching, training, "
                            f"selection and feedback (default: {' '.join(config.DEFAULT_RATS)})")
    model.add_argument("--load", default="none",
                       help='Model loading: "none"=train fresh (default), '
                            '"existing"=load latest, or path to .pt')
    model.add_argument("--epochs", type=int, default=0,
                       help="Training epochs (default: 100 from config)")

    fb = p.add_argument_group("Feedback loop options")
    fb.add_argument("--num_vehicles", type=int, default=1,
                    help="Number of vehicles (default: 1 = single-vehicle)")
    fb.add_argument("--retrain_interval", type=int, default=500,
                    help="Samples before retraining (default: 500)")
    fb.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility")
    fb.add_argument("--base_packet_size", type=int, default=1024,
                    help="Base packet size in bytes (default: 1024)")
    fb.add_argument("--correction_exp", type=float, default=config.PDR_CORRECTION_EXPONENT,
                    help="PDR correction exponent (default: 1.0)")
    fb.add_argument("--tx-interval", type=int, default=None,
                    help="Override TX interval in ms for all RATs (default: per-RAT from config)")
    fb.add_argument("--no-dtmc", action="store_true",
                    help="Disable DTMC adaptive packet sizing (use fixed base_packet_size)")
    fb.add_argument("--no-pqos", action="store_true",
                    help="Disable pQoS model inference (opportunistic baseline)")

    skip = p.add_argument_group("Skip stages")
    skip.add_argument("--skip_trimming", action="store_true")
    skip.add_argument("--skip_matching", action="store_true")
    skip.add_argument("--skip_training", action="store_true")
    skip.add_argument("--skip_selection", action="store_true")
    skip.add_argument("--skip_feedback", action="store_true")

    args = p.parse_args(argv)

    if not any([args.raw_data, args.data, args.npz, args.merged_csv]):
        p.error("provide at least one of --raw_data, --data, --npz, or --merged_csv")

    args.rats = config.validate_rats(args.rats)

    return args


def stage_trim(args) -> None:
    """Stage 1: Ingest raw logs into cohda-compatible df_*.json, then trimmed CSVs."""
    if not args.raw_data or args.skip_trimming:
        print("-- Skipping trimming (no --raw_data or --skip_trimming)")
        return

    banner("Stage 1: Trimming raw logs")

    from scripts.ingest_logs import ingest_dataset
    from scripts.convert_json_to_trim import convert_all

    # Write outputs to --data if provided, otherwise next to the raw logs
    out = args.data if args.data else os.path.join(args.raw_data, "trimmed")
    print(f"Raw logs:       {args.raw_data}")
    print(f"Trimmed output: {out}")

    # Per-tour df_*.json plus merged top-level files, then trim_*.csv from the merge
    ingest_dataset(args.raw_data, out)
    convert_all(out, out)

    args.data = out


def stage_match(args) -> None:
    """Stage 2: GPS-match cross-RAT data."""
    if not args.data or args.skip_matching:
        print("-- Skipping matching (no --data or --skip_matching)")
        return

    required = ["matched_5g.csv", "super.csv"] + [
        f"matched_{rat}.csv" for rat in args.rats if rat != "5g"
    ]
    all_present = all(
        Path(args.data, f).is_file() for f in required
    )

    if not all_present:
        banner("Stage 2: Matching cross-RAT data by GPS")
        from scripts.prepare_data import match_data
        match_data(args.data, rats=args.rats)
    else:
        print(f"-- All matched CSVs present in {args.data}, skipping matching")


def stage_train(args) -> None:
    """Stage 3: Train models using direct function calls."""
    if args.skip_training:
        print("-- Skipping training (--skip_training)")
        return

    banner(f"Stage 3: Training models ({args.model} for: {' '.join(args.rats)})")

    from learning.main import load_csv_data, prepare_data, train_single_model_torch
    from learning.model import automatic_train_torch, load_torch_model
    from config import (
        TIMESTEPS, EPOCHS, PDR_WINDOW, FEATURES_COUNT,
        get_tx_interval_ms, MODEL_DIR, OUTPUT_DIR,
    )
    from utils import find_files_with_string, get_latest_model, ensure_dir_exists

    ensure_dir_exists(MODEL_DIR)
    ensure_dir_exists(OUTPUT_DIR)

    MODEL_TYPE = args.model
    epochs = args.epochs if args.epochs else EPOCHS

    for rat in args.rats:
        print(f"---- Training {MODEL_TYPE} for {rat} ----")

        # Resolve data source
        data_npz = None
        data_path = None
        if args.npz:
            npz_dir = str(Path(args.npz).parent)
            rat_npz = os.path.join(npz_dir, f"{rat}_lstm_data.npz")
            if Path(rat_npz).is_file():
                data_npz = rat_npz
            else:
                print(f"  Warning: {rat_npz} not found, falling back to CSV")
                if args.data:
                    data_path = args.data
                else:
                    print(f"  Error: no data for {rat}")
                    continue
        elif args.data:
            data_path = args.data
        else:
            print(f"Warning: no training data source for {rat}, skipping")
            continue

        tx_interval = get_tx_interval_ms(rat)
        FEATURES = FEATURES_COUNT[rat]

        # Find trimmed CSV files
        files = []
        if data_path:
            files = find_files_with_string(data_path, f"trim_{rat}")
            if not files and not data_npz:
                print(f"  Warning: no trim_{rat} files in {data_path}, skipping {rat}")
                continue
        new_file = None
        if args.new_data:
            if not os.path.exists(args.new_data):
                raise ValueError(f"New data path not found: {args.new_data}")
            new_files = find_files_with_string(args.new_data, f"trim_{rat}")
            if not new_files:
                print(f"  Warning: no matching files for trim_{rat} in {args.new_data}")
                new_file = None
            else:
                new_file = new_files[0]

        # Determine model loading strategy
        if args.load == "none":
            model_path = None
            do_load = False
        elif args.load and args.load != "existing":
            model_path = os.path.join(MODEL_DIR, os.path.basename(args.load))
            do_load = True
        else:
            existing = get_latest_model(MODEL_TYPE, rat)
            if existing:
                model_path = existing
                do_load = True
            else:
                model_path = None
                do_load = False

        # Load and preprocess raw CSV data (skip if NPZ provided)
        df, df_new = None, None
        if data_path and not data_npz:
            df = load_csv_data(data_path, files, PDR_WINDOW, tx_interval)
            if args.new_data and new_file:
                df_new = load_csv_data(args.new_data, [new_file], PDR_WINDOW, tx_interval)

        # Prepare training data
        X_train, y_train, X_new_data, y_new_data = prepare_data(
            df, df_new, rat, data_npz, OUTPUT_DIR)

        # Convert targets to dictionary format for multi-output model
        y_train_dict = {
            "latency_ms": y_train[:, 0],
            "pdr": y_train[:, 1]
        }

        # Load existing model or train new one
        if do_load and model_path and os.path.exists(model_path):
            if MODEL_TYPE not in model_path:
                raise ValueError(f"Model file '{model_path}' does not match type '{MODEL_TYPE}'")

            print(f"Loading existing {MODEL_TYPE} model for {rat}...")
            model = load_torch_model(model_path)
            print(f"{MODEL_TYPE.upper()} model loaded: {model_path}")
        else:
            if not do_load:
                print(f"No model specified, building new {MODEL_TYPE} model...")
            elif not model_path or not os.path.exists(model_path):
                print(f"No existing model found, building new {MODEL_TYPE} model...")

            model, history = train_single_model_torch(
                MODEL_TYPE, TIMESTEPS, FEATURES, X_train, y_train_dict, rat, epochs)

            with open(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_{rat}_training_history.json"), "w") as f:
                json.dump(history, f)

        # Incremental learning
        if X_new_data is not None:
            print("=" * 60)
            print(f"Starting automatic incremental retraining for {MODEL_TYPE}...")
            print("=" * 60)
            automatic_train_torch(model, X_new_data, y_new_data, 32, 500, 0.15,
                                  os.path.join(OUTPUT_DIR, f"prediction_log_{MODEL_TYPE}_{rat}.csv"), rat, MODEL_TYPE)
        else:
            print("No new data provided. Skipping incremental retraining.")

        print()


def stage_selection(args) -> None:
    """Stage 4: RAT selection using direct function calls."""
    if args.skip_selection or not args.data:
        print("-- Skipping selection (--skip_selection or no --data)")
        return

    banner("Stage 4: RAT selection on matched data")

    from selection.rat_selection import (merge_csvs, get_predictions,
                                         select_best_rat, opportunistic_best_rat)
    from learning.model import load_torch_model
    from config import MODELS
    from utils import get_latest_model

    INPUT = args.data
    MODEL_TYPE = args.model
    model_types = MODELS if MODEL_TYPE == "all" else [MODEL_TYPE]

    print(f"_ Processing all input CSVs with model type(s): {model_types}, RATs: {list(args.rats)}")

    # Build super_merged from matched CSVs
    print("_ Merging into the super-CSV.")
    super_df = merge_csvs(INPUT, rats=args.rats)

    # Predict directly on super_merged GPS — no intermediate files
    gps_data = super_df[["tx_latitude", "tx_longitude"]].copy()

    for mt in model_types:
        for rat in args.rats:
            model_path = get_latest_model(mt, rat)
            if model_path is None:
                print(f"  Warning: no {mt} model found for {rat}, skipping")
                continue
            model_f = load_torch_model(model_path)
            latency, pdr = get_predictions(model_f, rat, gps_data)
            super_df[f"pred_latency_ms_{rat}_{mt}"] = latency
            super_df[f"pred_pdr_{rat}_{mt}"] = pdr
            print(f"  {rat}: {mt} predictions done.")

    print("_ Predictions added.")
    print("_ Sending to the selection algorithm.")
    for mt in model_types:
        super_df[f"Best_RAT_{mt}"] = super_df.apply(select_best_rat, args=(mt, args.rats), axis=1)
    print("_ Adding opportunistic algorithm.")
    super_df = opportunistic_best_rat(super_df, rats=args.rats)
    print("_ Algorithms done.")
    out_path = os.path.join(INPUT, "bestRAT_super.csv")
    print(f"_ Saving. {out_path}")
    super_df.to_csv(out_path, index=False)
    print("_ All done.")


def stage_feedback(args, merged_csv: str) -> None:
    """Stage 5: Feedback loop."""
    if args.skip_feedback:
        print("-- Skipping feedback loop (--skip_feedback)")
        return

    if not merged_csv:
        print("-- Skipping feedback loop: no super_merged CSV found")
        print("   Provide --merged_csv or ensure RAT selection produced one")
        return

    if args.num_vehicles > 1:
        banner(f"Stage 5: Multi-vehicle feedback loop ({args.num_vehicles} vehicles)")
        from scripts.feedback_loop import run_multi_vehicle_loop
        run_multi_vehicle_loop(
            input_csv=merged_csv,
            model_type=args.model,
            seed=args.seed,
            retrain_interval=args.retrain_interval,
            base_packet_size=args.base_packet_size,
            correction_exponent=args.correction_exp,
            num_vehicles=args.num_vehicles,
            sim_tx_interval_ms=args.tx_interval,
            enable_dtmc=not args.no_dtmc,
            enable_pqos=not args.no_pqos,
            rats=args.rats,
        )
    else:
        banner("Stage 5: Feedback loop (single vehicle)")
        from scripts.feedback_loop import run_feedback_loop
        run_feedback_loop(
            input_csv=merged_csv,
            model_type=args.model,
            seed=args.seed,
            retrain_interval=args.retrain_interval,
            base_packet_size=args.base_packet_size,
            correction_exponent=args.correction_exp,
            sim_tx_interval_ms=args.tx_interval,
            enable_dtmc=not args.no_dtmc,
            enable_pqos=not args.no_pqos,
            rats=args.rats,
        )


def main(argv=None) -> None:
    args = parse_args(argv)

    # Work from the project root
    os.chdir(_PROJECT_DIR)

    # Create timestamped run directory under output/
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_stamp}_{args.model}_{'-'.join(args.rats)}"
    if not args.skip_feedback: #and args.num_vehicles > 1:
        run_name += f"_{args.num_vehicles}v"
    run_dir = os.path.join("output", run_name)
    os.makedirs(run_dir, exist_ok=True)
    config.OUTPUT_DIR = run_dir
    print(f"Run output directory: {_PROJECT_DIR / run_dir}")

    # ── Stage 1: Trim raw logs ────────────────────────────────────────────
    stage_trim(args)

    # ── Stage 2: GPS-match cross-RAT data ─────────────────────────────────
    stage_match(args)

    # ── Stage 3: Train models ─────────────────────────────────────────────
    stage_train(args)

    # ── Stage 4: RAT selection + super CSV ────────────────────────────────
    stage_selection(args)

    # Auto-detect merged CSV for feedback loop
    merged_csv = args.merged_csv
    if not merged_csv and args.data:
        candidate = Path(args.data, "super_merged.csv")
        if candidate.is_file():
            merged_csv = str(candidate)

    # ── Stage 5: Feedback loop ────────────────────────────────────────────
    stage_feedback(args, merged_csv)

    # ── Done ──────────────────────────────────────────────────────────────
    banner("Pipeline complete")
    print(f"Output files are in: {_PROJECT_DIR / run_dir}/")
    print(f"Models are in:       {_PROJECT_DIR / 'models'}/")
    if merged_csv:
        print(f"Merged CSV used:     {merged_csv}")


if __name__ == "__main__":
    main()
