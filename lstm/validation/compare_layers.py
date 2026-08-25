"""
Validation de type 2 : analyse détaillée de la divergence entre TensorFlow
et PyTorch le long de la chaîne du réseau :

entrée → états cachés à chaque timestep → dernier état caché
→ couche dense → pré-activation / post-activation de la fonction sigmoid
→ prédiction.

Cette validation comprend également une comparaison, pour chaque timestep,
de l'état caché, de l'état de cellule (pour le LSTM) et de la sortie, sur
un petit batch de séquences réelles.

Utilisation :

    python -m validation.compare_layers --rat dsrc --model lstm --seed 42 \
        --npz output/dsrc_lstm_data.npz --n-sequences 10 --seq-length 5
"""
import os
import argparse
import numpy as np

import config
from utils import ensure_dir_exists
from validation.common import build_model_pair, instrumented_forward, rel_error
from validation.plotting import plot_chain_divergence, plot_timestep_divergence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation type 2: TF vs PyTorch layer-by-layer divergence")
    parser.add_argument("--rat", required=True, choices=["5g", "pc5", "dsrc"])
    parser.add_argument("--model", required=True, choices=["lstm", "gru", "rnn"], dest="model_type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--npz", type=str, help="Preprocessed NPZ archive with x_train")
    parser.add_argument("--n-sequences", type=int, default=10)
    parser.add_argument("--seq-length", type=int, default=None,
                         help="Truncate sequences to this many timesteps (default: full TIMESTEPS)")
    parser.add_argument("--start-index", type=int, default=None,
                         help="Dataset index to start the slice from (default: ~1/3 into the "
                              "dataset, to avoid the rolling-window warm-up region at the very start)")
    return parser.parse_args()


def _load_sequences(args, timesteps, features):
    seq_len = args.seq_length or timesteps
    if args.npz and os.path.exists(args.npz):
        data = np.load(args.npz)
        X_full = data["x_train"].astype(np.float32)
        start = args.start_index
        if start is None:
            start = max(0, len(X_full) // 3)
        end = start + args.n_sequences
        if end > len(X_full):
            raise ValueError(
                f"--start-index {start} + --n-sequences {args.n_sequences} exceeds dataset size {len(X_full)}")
        return X_full[start:end, :seq_len, :], start

    rng = np.random.default_rng(args.seed)
    return rng.random((args.n_sequences, seq_len, features)).astype(np.float32), None


def _check_representativeness(chain_tf, chain_pt, threshold=1e-6):
    for label, chain in (("TF", chain_tf), ("PT", chain_pt)):
        variance = float(np.var(chain["hidden_states"]))
        if variance < threshold:
            print(f"WARNING: {label} hidden states have near-zero variance ({variance:.2e}). "
                  "This batch may be degenerate (e.g. saturated/dead neurons on both sides), "
                  "which can mask a real divergence rather than confirm fidelity. "
                  "Consider changing --start-index.")


def main():
    args = parse_args()
    out_dir = os.path.join(config.OUTPUT_DIR, "validation")
    ensure_dir_exists(out_dir)

    timesteps = config.TIMESTEPS
    features = config.FEATURES_COUNT[args.rat]
    tf_model, torch_model = build_model_pair(args.model_type, args.rat, timesteps, features, args.seed)

    X, start = _load_sequences(args, timesteps, features)
    chain_tf, chain_pt = instrumented_forward(tf_model, torch_model, args.model_type, X)

    _check_representativeness(chain_tf, chain_pt)

    stages = ["input", "hidden_states", "last_hidden", "dense_out",
              "latency_pre", "latency_post", "pdr_pre", "pdr_post", "prediction"]
    if chain_tf["cell_states"] is not None:
        stages.insert(2, "cell_states")

    stage_names, mean_errors, max_errors = [], [], []
    report_lines = [f"Validation type 2 - {args.rat}/{args.model_type} - "
                     f"n_sequences={len(X)}, seq_length={X.shape[1]}, start_index={start}", ""]

    for stage in stages:
        err = rel_error(chain_tf[stage], chain_pt[stage])
        stage_names.append(stage)
        mean_errors.append(float(err.mean()))
        max_errors.append(float(err.max()))
        report_lines.append(f"{stage}: mean={err.mean():.3e} max={err.max():.3e}")

    # 'input' is the same array fed to both models, so its rel_error is
    # trivially 0 - comparing against it as a baseline would always report a
    # false "jump" at the very first real stage. The jump check only makes
    # sense between two stages that both involve an actual TF vs PT
    # computation, so it starts one stage later.
    noise_floor = 1e-6
    for i in range(2, len(mean_errors)):
        prev = max(mean_errors[i - 1], noise_floor)
        if mean_errors[i] / prev > 100:
            report_lines.append(
                f"WARNING: mean relative error jumps by >100x between "
                f"'{stage_names[i - 1]}' and '{stage_names[i]}' - investigate this stage.")

    report = "\n".join(report_lines)
    print(report)
    with open(os.path.join(out_dir, f"{args.rat}_{args.model_type}_chain_report.txt"), "w") as f:
        f.write(report + "\n")

    plot_chain_divergence(stage_names, mean_errors, max_errors,
                           os.path.join(out_dir, f"{args.rat}_{args.model_type}_chain_divergence.png"))

    hidden_err_per_seq_t = rel_error(chain_tf["hidden_states"], chain_pt["hidden_states"]).mean(axis=-1)
    cell_err_per_seq_t = None
    if chain_tf["cell_states"] is not None:
        cell_err_per_seq_t = rel_error(chain_tf["cell_states"], chain_pt["cell_states"]).mean(axis=-1)

    plot_timestep_divergence(hidden_err_per_seq_t, cell_err_per_seq_t,
                              os.path.join(out_dir, f"{args.rat}_{args.model_type}_timestep_divergence.png"))

    print(f"Reports and figures written to {out_dir}")


if __name__ == "__main__":
    main()
