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
    model.add_argument("--rats", nargs="+", default=["5g", "pc5", "dsrc"],
                       help="RATs to train (default: 5g pc5 dsrc)")
    model.add_argument("--load", default="none",
                       help='Model loading: "none"=train fresh (default), '
                            '"existing"=load latest, or path to .keras')
    model.add_argument("--epochs", type=int, default=0,
                       help="Training epochs (default: 100 from config)")

    fb = p.add_argument_group("Feedback loop options")
    fb.add_argument("--num_vehicles", type=int, default=1,
                    help="Number of vehicles (default: 1 = single-vehicle)")
    fb.add_argument("--retrain_interval", type=int, default=500,
                    help="Samples before retraining (default: 500)")
    fb.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility")
    fb.add_argument("--base_packet_size", type=int, default=1000,
                    help="Base packet size in bytes (default: 1000)")
    fb.add_argument("--correction_exp", type=float, default=0.8,
                    help="PDR correction exponent (default: 0.8)")

    skip = p.add_argument_group("Skip stages")
    skip.add_argument("--skip_trimming", action="store_true")
    skip.add_argument("--skip_matching", action="store_true")
    skip.add_argument("--skip_training", action="store_true")
    skip.add_argument("--skip_selection", action="store_true")
    skip.add_argument("--skip_feedback", action="store_true")

    args = p.parse_args(argv)

    if not any([args.raw_data, args.data, args.npz, args.merged_csv]):
        p.error("provide at least one of --raw_data, --data, --npz, or --merged_csv")

    return args


def stage_trim(args) -> None:
    """Stage 1: Trim raw logs using direct function calls."""
    if not args.raw_data or args.skip_trimming:
        print("-- Skipping trimming (no --raw_data or --skip_trimming)")
        return

    banner("Stage 1: Trimming raw logs")

    from scripts.trimmers import (find_files_with_string, trim_sa, trim_pc5,
                                   trim_dsrc, get_first_timestamp, append_files)

    RAW = args.raw_data
    # Write trimmed output to --data if provided, otherwise into --raw_data
    OUT = args.data if args.data else RAW

    if OUT != RAW:
        os.makedirs(OUT, exist_ok=True)
        print(f"Raw logs:       {RAW}")
        print(f"Trimmed output: {OUT}")

    # Raw logs are not CSVs; exclude pipeline outputs that may be in the same dir
    files_dsrc = [f for f in find_files_with_string(RAW, "dsrc") if not f.endswith(".csv")]
    files_pc5 = [f for f in find_files_with_string(RAW, "pc5") if not f.endswith(".csv")]
    files_5g = [f for f in find_files_with_string(RAW, "5g") if not f.endswith(".csv")]
    print("Matching DSRC files: ", files_dsrc)
    print("Matching PC5 files: ", files_pc5)
    print("Matching 5G files: ", files_5g)

    sa_data = trim_sa(files_5g, os.path.join(OUT, "trim_5g.csv"), RAW)

    combined_out_pc5 = 0
    combined_out_dsrc = 0
    trimmed_files_pc5 = []
    trimmed_files_dsrc = []

    for i, file in enumerate(files_pc5):
        output_file_pc5 = f"trim_pc5_{i}.csv"
        output_file_dsrc = f"trim_dsrc_{i}.csv"
        print("Trimming PC5 file ", file)
        out_pc5, pc5_data = trim_pc5(os.path.join(RAW, file),
                                      os.path.join(OUT, output_file_pc5))
        print(f"Generated {out_pc5} lines for PC5")
        combined_out_pc5 += out_pc5

        rsu_id = file.split("_")[2]
        try:
            log_dsrc = [f for f in files_dsrc if rsu_id in f][0]
        except IndexError:
            print("No matching DSRC file found for ", file)
            trimmed_files_pc5.append(os.path.join(OUT, output_file_pc5))
            continue

        print("Trimming DSRC file ", log_dsrc)
        out_dsrc = trim_dsrc(os.path.join(RAW, log_dsrc),
                              os.path.join(OUT, output_file_dsrc),
                              pc5_data, sa_data)
        print(f"Generated {out_dsrc} lines for DSRC")
        combined_out_dsrc += out_dsrc
        trimmed_files_pc5.append(os.path.join(OUT, output_file_pc5))
        trimmed_files_dsrc.append(os.path.join(OUT, output_file_dsrc))

    trimmed_files_pc5.sort(key=get_first_timestamp)
    append_files(os.path.join(OUT, "trim_pc5.csv"), trimmed_files_pc5)
    print(f"Combined trimmed PC5 files into trim_pc5.csv. Total lines: {combined_out_pc5}")

    trimmed_files_dsrc.sort(key=get_first_timestamp)
    append_files(os.path.join(OUT, "trim_dsrc.csv"), trimmed_files_dsrc)
    print(f"Combined trimmed DSRC files into trim_dsrc.csv. Total lines: {combined_out_dsrc}")

    # Clean up intermediate per-RSU trimmed files
    for f in trimmed_files_pc5 + trimmed_files_dsrc:
        os.remove(f)

    if not args.data:
        args.data = RAW


def stage_match(args) -> None:
    """Stage 2: GPS-match cross-RAT data."""
    if not args.data or args.skip_matching:
        print("-- Skipping matching (no --data or --skip_matching)")
        return

    required = ["matched_5g.csv", "matched_dsrc.csv",
                 "matched_pc5.csv", "super.csv"]
    all_present = all(
        Path(args.data, f).is_file() for f in required
    )

    if not all_present:
        banner("Stage 2: Matching cross-RAT data by GPS")
        from scripts.prepare_data import match_data
        match_data(args.data)
    else:
        print(f"-- All matched CSVs present in {args.data}, skipping matching")


def stage_train(args) -> None:
    """Stage 3: Train models using direct function calls."""
    if args.skip_training:
        print("-- Skipping training (--skip_training)")
        return

    banner(f"Stage 3: Training models ({args.model} for: {' '.join(args.rats)})")

    from learning.main import load_csv_data, prepare_data, train_single_model
    from learning.model import automatic_train, rmse
    from keras.models import load_model
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
            loaded_path = get_latest_model(MODEL_TYPE, rat)
            model = load_model(loaded_path, custom_objects={'rmse': rmse})
            print(f"{MODEL_TYPE.upper()} model loaded: {loaded_path}")
        else:
            if not do_load:
                print(f"No model specified, building new {MODEL_TYPE} model...")
            elif not model_path or not os.path.exists(model_path):
                print(f"No existing model found, building new {MODEL_TYPE} model...")

            model, history = train_single_model(
                MODEL_TYPE, TIMESTEPS, FEATURES, X_train, y_train_dict, rat, epochs)

            with open(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_{rat}_training_history.json"), "w") as f:
                json.dump(history.history, f)

        # Incremental learning
        if X_new_data is not None:
            print("=" * 60)
            print(f"Starting automatic incremental retraining for {MODEL_TYPE}...")
            print("=" * 60)
            automatic_train(model, X_new_data, y_new_data, 32, 500, 0.15,
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

    from selection.rat_selection import (merge_csvs, add_predictions,
                                         select_best_rat, opportunistic_best_rat)
    from learning.data_preprocessing import preprocess_lstm_input
    from learning.model import automatic_train, rmse
    from keras.models import load_model
    from config import TIMESTEPS, TARGET_COLS, RATS, MODELS, OUTPUT_DIR
    from utils import get_latest_model

    INPUT = args.data
    MODEL_TYPE = args.model
    model_types = MODELS if MODEL_TYPE == "all" else [MODEL_TYPE]

    print(f"_ Processing all input CSVs with model type(s): {model_types}")

    # Generate predictions for each RAT x model_type
    for rat in RATS:
        matched_csv = os.path.join(INPUT, f"matched_{rat}.csv")
        if not os.path.exists(matched_csv):
            print(f"  Warning: {matched_csv} not found, skipping {rat}")
            continue
        dfl = pd.read_csv(matched_csv)
        (X_new, y_new, scalers) = preprocess_lstm_input(
            dfl, new=True, rat=rat, target_cols=TARGET_COLS, seq_length=TIMESTEPS)
        for mt in model_types:
            model_path = get_latest_model(mt, rat)
            if model_path is None:
                print(f"  Warning: no {mt} model found for {rat}, skipping")
                continue
            model_f = load_model(model_path, custom_objects={'rmse': rmse})
            automatic_train(model_f, X_new, y_new, 32, 200, 0.15,
                            os.path.join(OUTPUT_DIR, f"final_log_{mt}_{rat}.csv"), rat, mt)
            print(f"  {rat}: {mt} predictions done.")

    print("_ Processing done.")
    print("_ Merging into the super-CSV.")
    super_df = merge_csvs(INPUT)
    print("_ Adding predictions into the super-CSV.")
    super_pred_df = add_predictions(super_df, INPUT, model_types=model_types)
    print("_ Merging done.")
    print("_ Sending to the selection algorithm.")
    for mt in model_types:
        super_df[f"Best_RAT_{mt}"] = super_pred_df.apply(select_best_rat, args=(mt,), axis=1)
    print("_ Adding opportunistic algorithm.")
    super_df = opportunistic_best_rat(super_df)
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
        )


def main(argv=None) -> None:
    args = parse_args(argv)

    # Work from the project root
    os.chdir(_PROJECT_DIR)

    # Create timestamped run directory under output/
    import config
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_stamp}_{args.model}"
    if not args.skip_feedback and args.num_vehicles > 1:
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
