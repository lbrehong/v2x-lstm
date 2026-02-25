"""
Batch pipeline runner: train once per model, then run feedback loops
for multiple vehicle counts.

Uses separate datasets for training vs inference:
  --train_data  : dataset for training (e.g. logs_cohda/full)
  --infer_data  : dataset for feedback/inference (e.g. logs_cohda/full_new)

If only --train_data is given, it is reused for inference.
If only --infer_data is given, training is skipped (--load existing).

Usage:
    # Train on full, infer on full_new
    python -m run_batch --train_data /path/to/full --infer_data /path/to/full_new

    # Same dataset for both
    python -m run_batch --train_data /path/to/logs

    # Infer only (skip training, load existing models)
    python -m run_batch --infer_data /path/to/full_new

    # Single model, custom vehicle counts
    python -m run_batch --train_data /path/to/full --infer_data /path/to/full_new \
        --models lstm --vehicles 1 5 20
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Default vehicle counts for the sweep
DEFAULT_VEHICLES = [1, 2, 5, 10, 20, 50, 100]
DEFAULT_MODELS = ["lstm", "gru", "rnn"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch runner: train models then sweep vehicle counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data = p.add_argument_group("Data sources")
    data.add_argument("--raw_data", default="",
                      help="Raw log folder (triggers trimming before training)")
    data.add_argument("--train_data", default="",
                      help="Trimmed CSV folder for training")
    data.add_argument("--infer_data", default="",
                      help="Trimmed CSV folder for inference/feedback "
                           "(defaults to --train_data if omitted)")
    data.add_argument("--npz", default="",
                      help="Preprocessed NPZ archive (training)")
    data.add_argument("--merged_csv", default="",
                      help="Pre-built super_merged CSV (skip selection)")

    sweep = p.add_argument_group("Sweep parameters")
    sweep.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                       choices=["lstm", "gru", "rnn"],
                       help=f"Model architectures (default: {DEFAULT_MODELS})")
    sweep.add_argument("--vehicles", nargs="+", type=int,
                       default=DEFAULT_VEHICLES,
                       help=f"Vehicle counts to sweep (default: {DEFAULT_VEHICLES})")
    sweep.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    opts = p.add_argument_group("Pipeline options")
    opts.add_argument("--epochs", type=int, default=0,
                      help="Training epochs (0 = use config default)")
    opts.add_argument("--retrain_interval", type=int, default=500,
                      help="Feedback retraining interval (default: 500)")
    opts.add_argument("--base_packet_size", type=int, default=1000,
                      help="Base packet size in bytes (default: 1000)")
    opts.add_argument("--correction_exp", type=float, default=0.8,
                      help="PDR correction exponent (default: 0.8)")
    opts.add_argument("--tx_interval", type=int, default=None,
                      help="Override TX interval in ms for all RATs (default: per-RAT from config)")

    skip = p.add_argument_group("Skip stages (applied to training run)")
    skip.add_argument("--skip_trimming", action="store_true")
    skip.add_argument("--skip_matching", action="store_true")

    args = p.parse_args()

    if not args.train_data and not args.infer_data:
        p.error("provide at least one of --train_data or --infer_data")

    # Default: infer_data falls back to train_data
    if not args.infer_data:
        args.infer_data = args.train_data

    return args


def fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def set_terminal_title(title: str) -> None:
    """Set the terminal window/tab title via ANSI escape (works over SSH/tmux)."""
    sys.stdout.write(f"\033]2;{title}\007")
    sys.stdout.flush()


def progress_banner(
    step: int, total: int, label: str, argv: list[str], batch_start: float,
) -> None:
    """Print a progress header and update terminal title before each run."""
    elapsed = time.time() - batch_start
    pct = step / total * 100

    # ETA from average duration of completed steps
    if step > 0:
        avg = elapsed / step
        remaining = avg * (total - step)
        eta_str = f"ETA {fmt_duration(remaining)}"
    else:
        eta_str = "ETA --"

    bar_width = 30
    filled = int(bar_width * step / total)
    bar = "=" * filled + ">" + "." * (bar_width - filled - 1)

    # Persistent: terminal title stays visible regardless of output scrolling
    set_terminal_title(
        f"[{step}/{total} {pct:.0f}%] {label} | "
        f"elapsed {fmt_duration(elapsed)} | {eta_str}"
    )

    print()
    print("#" * 68)
    print(f"#  [{bar}] {step}/{total} ({pct:.0f}%)  "
          f"elapsed {fmt_duration(elapsed)}  {eta_str}")
    print(f"#  {label}")
    print(f"#  argv: python -m run_pipeline {' '.join(argv)}")
    print("#" * 68)
    print()


def run_pipeline(
    argv: list[str],
    label: str,
    step: int = 0,
    total: int = 0,
    batch_start: float = 0.0,
) -> bool:
    """Run a single pipeline invocation. Returns True on success."""
    from run_pipeline import main as pipeline_main

    progress_banner(step, total, label, argv, batch_start)

    t0 = time.time()
    try:
        pipeline_main(argv)
        elapsed = time.time() - t0
        print(f"\n>> [{step + 1}/{total}] {label} completed in {fmt_duration(elapsed)}")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n>> [{step + 1}/{total}] {label} FAILED after {fmt_duration(elapsed)}: {e}")
        return False


def main() -> None:
    args = parse_args()
    batch_start = time.time()
    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    skip_training = not args.train_data

    # Training data args (Phase 1)
    train_data_args = []
    if args.raw_data:
        train_data_args += ["--raw_data", args.raw_data]
    if args.train_data:
        train_data_args += ["--data", args.train_data]
    if args.npz:
        train_data_args += ["--npz", args.npz]

    # Inference data args (Phase 2)
    infer_data_args = ["--data", args.infer_data]

    # Common feedback args
    fb_args = [
        "--seed", str(args.seed),
        "--retrain_interval", str(args.retrain_interval),
        "--base_packet_size", str(args.base_packet_size),
        "--correction_exp", str(args.correction_exp),
    ]
    if args.tx_interval is not None:
        fb_args += ["--tx-interval", str(args.tx_interval)]

    skip_args = []
    if args.skip_trimming:
        skip_args.append("--skip_trimming")
    if args.skip_matching:
        skip_args.append("--skip_matching")

    epoch_args = ["--epochs", str(args.epochs)] if args.epochs else []

    # Build the full run matrix
    n_train = 0 if skip_training else len(args.models)
    n_feedback = len(args.models) * len(args.vehicles)
    total_runs = n_train + n_feedback
    results: list[tuple[str, bool, float]] = []

    print("=" * 68)
    print(f"  Batch run started at {batch_stamp}")
    print(f"  Models:     {args.models}")
    print(f"  Vehicles:   {args.vehicles}")
    print(f"  Seed:       {args.seed}")
    if args.train_data:
        print(f"  Train data: {args.train_data}")
    print(f"  Infer data: {args.infer_data}")
    if args.tx_interval:
        print(f"  TX interval: {args.tx_interval}ms (all RATs)")
    print(f"  Total pipeline invocations: {total_runs}")
    if n_train:
        print(f"    - {n_train} training runs")
    print(f"    - {n_feedback} feedback runs")
    print("=" * 68)

    step = 0  # overall progress counter

    # ── Phase 1: Train each model (trim + match + train + select) ─────────
    if not skip_training:
        for model in args.models:
            label = f"TRAIN {model.upper()}"
            t0 = time.time()
            argv = [
                *train_data_args,
                "--model", model,
                *skip_args,
                *epoch_args,
                "--skip_feedback",
            ]
            ok = run_pipeline(argv, label, step, total_runs, batch_start)
            results.append((label, ok, time.time() - t0))
            step += 1

            # After first model trains, skip trimming+matching for subsequent
            if "--skip_trimming" not in skip_args:
                skip_args.append("--skip_trimming")
            if "--skip_matching" not in skip_args:
                skip_args.append("--skip_matching")

    # ── Phase 2: Feedback loops for each model × vehicle count ────────────
    # Each run loads the latest non-retrained model (get_latest_model
    # excludes retrained_* files by default), so retrained models saved
    # during one feedback run cannot leak into subsequent runs.

    # Resolve merged CSV: check infer_data first, then train_data
    merged_csv_args = []
    if args.merged_csv:
        merged_csv_args = ["--merged_csv", args.merged_csv]
    else:
        for data_dir in [args.infer_data, args.train_data]:
            if not data_dir:
                continue
            candidate = Path(data_dir) / "super_merged.csv"
            if candidate.is_file():
                merged_csv_args = ["--merged_csv", str(candidate)]
                break

    for model in args.models:
        for nv in args.vehicles:
            label = f"FEEDBACK {model.upper()} {nv}v"
            t0 = time.time()
            argv = [
                *infer_data_args,
                *merged_csv_args,
                "--model", model,
                "--load", "existing",
                "--skip_trimming",
                "--skip_matching",
                "--skip_training",
                "--skip_selection",
                "--num_vehicles", str(nv),
                *fb_args,
            ]
            ok = run_pipeline(argv, label, step, total_runs, batch_start)
            results.append((label, ok, time.time() - t0))
            step += 1

    # ── Summary ───────────────────────────────────────────────────────────
    total_elapsed = time.time() - batch_start
    print()
    print("=" * 68)
    print("  BATCH SUMMARY")
    print("=" * 68)
    for label, ok, elapsed in results:
        status = "OK" if ok else "FAIL"
        print(f"  [{status:>4}] {label:<30} {fmt_duration(elapsed):>10}")
    print("-" * 68)

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print(f"  {n_ok}/{len(results)} succeeded, {n_fail} failed")
    print(f"  Total elapsed: {fmt_duration(total_elapsed)}")
    print("=" * 68)

    status = "DONE" if n_fail == 0 else f"DONE ({n_fail} failed)"
    set_terminal_title(f"Batch {status} in {fmt_duration(total_elapsed)}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()