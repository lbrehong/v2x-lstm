"""
Validation de type 1 : injecte des poids identiques dans les modèles
TensorFlow et PyTorch, puis compare l'erreur de prédiction obtenue entre
les deux modèles à l'aide de l'erreur relative.

Cette validation permet de vérifier si, avec exactement les mêmes poids
et les mêmes données d'entrée, les deux frameworks produisent des
prédictions identiques ou très proches.

Utilisation :

    python -m validation.compare_predictions --rat dsrc --model lstm --seed 42 --stage init

    python -m validation.compare_predictions --rat dsrc --model lstm --seed 42 \
        --npz output/dsrc_lstm_data.npz --stage trained --epochs 20
"""
import os
import argparse
import numpy as np

import config
from utils import ensure_dir_exists
from validation.common import build_model_pair, rel_error
from validation.plotting import (
    plot_loss_curves, plot_weight_divergence, plot_prediction_comparison,
    plot_prediction_divergence_over_epochs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validation type 1: TF vs PyTorch prediction error comparison")
    parser.add_argument("--rat", required=True, choices=["5g", "pc5", "dsrc"])
    parser.add_argument("--model", required=True, choices=["lstm", "gru", "rnn"], dest="model_type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--npz", type=str, help="Preprocessed NPZ archive with x_train/y_train")
    parser.add_argument("--stage", choices=["init", "trained"], default="init")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--batch-eval-size", type=int, default=256,
                         help="Number of samples to run inference on for the comparison report")
    return parser.parse_args()


def _load_eval_batch(args):
    if args.npz and os.path.exists(args.npz):
        data = np.load(args.npz)
        X_full = data["x_train"].astype(np.float32)
        y_full = data["y_train"].astype(np.float32)
        rng = np.random.default_rng(args.seed)
        n = min(len(X_full), args.batch_eval_size)
        idx = rng.choice(len(X_full), size=n, replace=False)
        y_eval = {"latency_ms": y_full[idx, 0], "pdr": y_full[idx, 1]}
        return X_full[idx], y_eval, X_full, y_full

    timesteps = config.TIMESTEPS
    features = config.FEATURES_COUNT[args.rat]
    rng = np.random.default_rng(args.seed)
    X_eval = rng.random((args.batch_eval_size, timesteps, features)).astype(np.float32)
    return X_eval, None, None, None


def _predict_tf(tf_model, X):
    preds = tf_model.predict(X, verbose=0)
    return {"latency_ms": np.asarray(preds[0]).flatten(), "pdr": np.asarray(preds[1]).flatten()}


def _predict_torch(torch_model, X):
    import torch
    torch_model.eval()
    with torch.no_grad():
        lat, pdr = torch_model(torch.from_numpy(X.astype(np.float32)))
    return {"latency_ms": lat.numpy().flatten(), "pdr": pdr.numpy().flatten()}


def _report_rel_error(pred_tf, pred_pt, out_dir, rat, model_type, stage):
    lines = [f"Validation type 1 - {rat}/{model_type} - stage={stage}", ""]
    csv_path = os.path.join(out_dir, f"{rat}_{model_type}_{stage}_prediction_errors.csv")
    with open(csv_path, "w") as f:
        f.write("output,mean_rel_error,max_rel_error,p50,p95,p99\n")
        for key in ("latency_ms", "pdr"):
            err = rel_error(pred_tf[key], pred_pt[key])
            row = (key, err.mean(), err.max(), np.percentile(err, 50),
                   np.percentile(err, 95), np.percentile(err, 99))
            f.write(",".join(str(v) for v in row) + "\n")
            lines.append(f"{key}: mean={row[1]:.3e} max={row[2]:.3e} "
                         f"p50={row[3]:.3e} p95={row[4]:.3e} p99={row[5]:.3e}")

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(out_dir, f"{rat}_{model_type}_{stage}_summary.txt"), "w") as f:
        f.write(summary + "\n")


def _train_synced(tf_model, torch_model, model_type, X_full, y_full, epochs, batch_size, seed, val_split):
    """
    Entraîne indépendamment les modèles TensorFlow et PyTorch à partir des
    mêmes poids initiaux transférés, avec un ordre de mélange des données
    synchronisé manuellement à chaque epoch.

    Les mécanismes de mélange propres à Keras et à PyTorch utilisent des
    générateurs aléatoires (RNG) différents. Ainsi, utiliser uniquement la
    même seed ne garantit pas que les deux frameworks utiliseront le même
    ordre des batches.

    La fonction reproduit également le comportement de `validation_split`
    de Keras : la dernière fraction `val_split` des tableaux, en conservant
    leur ordre d'origine, est réservée à la validation.
    """
    import torch as torch_lib
    from learning.weight_transfer import weight_rel_error

    X = X_full.astype(np.float32)
    y_lat = y_full[:, 0].astype(np.float32).reshape(-1, 1)
    y_pdr = y_full[:, 1].astype(np.float32).reshape(-1, 1)

    n_val = int(len(X) * val_split)
    X_val, y_val_lat, y_val_pdr = X[-n_val:], y_lat[-n_val:], y_pdr[-n_val:]
    X_tr, y_tr_lat, y_tr_pdr = X[:-n_val], y_lat[:-n_val], y_pdr[:-n_val]
    n_samples = len(X_tr)

    rng = np.random.default_rng(seed)
    hist_keys = ["loss", "latency_ms_loss", "pdr_loss",
                 "val_loss", "val_latency_ms_loss", "val_pdr_loss"]
    history_tf = {k: [] for k in hist_keys}
    history_pt = {k: [] for k in hist_keys}
    weight_divergence = {}
    pred_divergence = {"latency_ms_mean": [], "latency_ms_p99": [], "pdr_mean": [], "pdr_p99": []}

    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        tf_losses, tf_lat_losses, tf_pdr_losses = [], [], []
        pt_losses, pt_lat_losses, pt_pdr_losses = [], [], []

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb_lat, yb_pdr = X_tr[idx], y_tr_lat[idx], y_tr_pdr[idx]
            # forcer TF et PyTorch à voir exactement les mêmes échantillons, dans le même ordre, à chaque batch, pour isoler la divergence due au framework lui-même (optimiseur, flottant) plutôt qu'à un ordre de mélange différent.
            tf_result = tf_model.train_on_batch(
                xb, {"latency_ms": yb_lat, "pdr": yb_pdr}, return_dict=True)
            tf_losses.append(tf_result["loss"])
            tf_lat_losses.append(tf_result["latency_ms_loss"])
            tf_pdr_losses.append(tf_result["pdr_loss"])

            torch_model.optimizer.zero_grad()
            pred_lat, pred_pdr = torch_model(torch_lib.from_numpy(xb))
            loss_lat = torch_lib.mean((pred_lat - torch_lib.from_numpy(yb_lat)) ** 2)
            loss_pdr = torch_lib.mean((pred_pdr - torch_lib.from_numpy(yb_pdr)) ** 2)
            loss = loss_lat + 1.5 * loss_pdr
            loss.backward()
            torch_model.optimizer.step()
            pt_losses.append(loss.item())
            pt_lat_losses.append(loss_lat.item())
            pt_pdr_losses.append(loss_pdr.item())

        tf_val = tf_model.evaluate(
            X_val, {"latency_ms": y_val_lat, "pdr": y_val_pdr}, verbose=0, return_dict=True)
        tf_pred_lat_val, tf_pred_pdr_val = tf_model.predict(X_val, verbose=0)

        torch_model.eval()
        with torch_lib.no_grad():
            pred_lat, pred_pdr = torch_model(torch_lib.from_numpy(X_val))
            pt_val_loss_lat = torch_lib.mean((pred_lat - torch_lib.from_numpy(y_val_lat)) ** 2).item()
            pt_val_loss_pdr = torch_lib.mean((pred_pdr - torch_lib.from_numpy(y_val_pdr)) ** 2).item()
            pt_val_loss = pt_val_loss_lat + 1.5 * pt_val_loss_pdr

        # Per-sample divergence between the two independently-trained models
        # on the held-out validation set, tracked every epoch. This is what
        # actually answers "does TF-vs-PT disagreement on individual samples
        # grow or shrink with more training?" - the weight-space and
        # aggregate-loss comparisons above cannot answer that on their own,
        # since a model can agree almost perfectly in aggregate while still
        # disagreeing sharply on a handful of rare/hard samples.
        lat_err = rel_error(tf_pred_lat_val.flatten(), pred_lat.numpy().flatten())
        pdr_err = rel_error(tf_pred_pdr_val.flatten(), pred_pdr.numpy().flatten())
        pred_divergence["latency_ms_mean"].append(float(lat_err.mean()))
        pred_divergence["latency_ms_p99"].append(float(np.percentile(lat_err, 99)))
        pred_divergence["pdr_mean"].append(float(pdr_err.mean()))
        pred_divergence["pdr_p99"].append(float(np.percentile(pdr_err, 99)))

        history_tf["loss"].append(float(np.mean(tf_losses)))
        history_tf["latency_ms_loss"].append(float(np.mean(tf_lat_losses)))
        history_tf["pdr_loss"].append(float(np.mean(tf_pdr_losses)))
        history_tf["val_loss"].append(float(tf_val["loss"]))
        history_tf["val_latency_ms_loss"].append(float(tf_val["latency_ms_loss"]))
        history_tf["val_pdr_loss"].append(float(tf_val["pdr_loss"]))

        history_pt["loss"].append(float(np.mean(pt_losses)))
        history_pt["latency_ms_loss"].append(float(np.mean(pt_lat_losses)))
        history_pt["pdr_loss"].append(float(np.mean(pt_pdr_losses)))
        history_pt["val_loss"].append(pt_val_loss)
        history_pt["val_latency_ms_loss"].append(pt_val_loss_lat)
        history_pt["val_pdr_loss"].append(pt_val_loss_pdr)

        for name, value in weight_rel_error(tf_model, torch_model, model_type).items():
            weight_divergence.setdefault(name, []).append(value)

        print(f"[trained-stage] epoch {epoch + 1}/{epochs} - "
              f"TF loss {history_tf['loss'][-1]:.4f} - PT loss {history_pt['loss'][-1]:.4f}")

    y_val_eval = {"latency_ms": y_val_lat.flatten(), "pdr": y_val_pdr.flatten()}
    return history_tf, history_pt, weight_divergence, pred_divergence, X_val, y_val_eval


def _write_series_csv(series_dict, out_path):
    """Write a dict of {name: [value_per_epoch]} as one column per name."""
    import csv
    names = list(series_dict.keys())
    n_epochs = max((len(v) for v in series_dict.values()), default=0)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + names)
        for epoch in range(n_epochs):
            writer.writerow([epoch + 1] + [series_dict[name][epoch] if epoch < len(series_dict[name]) else ""
                                            for name in names])


def main():
    args = parse_args()
    out_dir = os.path.join(config.OUTPUT_DIR, "validation")
    ensure_dir_exists(out_dir)

    timesteps = config.TIMESTEPS
    features = config.FEATURES_COUNT[args.rat]
    tf_model, torch_model = build_model_pair(args.model_type, args.rat, timesteps, features, args.seed)

    if args.stage == "trained":
        if not args.npz or not os.path.exists(args.npz):
            raise ValueError("--stage trained requires --npz pointing to preprocessed training data.")
        data = np.load(args.npz)
        history_tf, history_pt, weight_divergence, pred_divergence, X_eval, y_eval = _train_synced(
            tf_model, torch_model, args.model_type, data["x_train"], data["y_train"],
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
            val_split=config.VALIDATION_SPLIT)

        prefix = os.path.join(out_dir, f"{args.rat}_{args.model_type}")
        plot_loss_curves(history_tf, history_pt, f"{prefix}_loss_curves.png")
        plot_weight_divergence(weight_divergence, f"{prefix}_weight_divergence.png")
        plot_prediction_divergence_over_epochs(pred_divergence, f"{prefix}_prediction_divergence.png")

        _write_series_csv(weight_divergence, f"{prefix}_weight_divergence.csv")
        _write_series_csv(pred_divergence, f"{prefix}_prediction_divergence.csv")
        _write_series_csv({f"tf_{k}": v for k, v in history_tf.items()}
                           | {f"pt_{k}": v for k, v in history_pt.items()},
                           f"{prefix}_loss_history.csv")
    else:
        X_eval, y_eval, _, _ = _load_eval_batch(args)

    pred_tf = _predict_tf(tf_model, X_eval)
    pred_pt = _predict_torch(torch_model, X_eval)

    _report_rel_error(pred_tf, pred_pt, out_dir, args.rat, args.model_type, args.stage)

    if y_eval is not None:
        plot_prediction_comparison(
            y_eval, pred_tf, pred_pt,
            os.path.join(out_dir, f"{args.rat}_{args.model_type}_{args.stage}_predictions.png"))

    print(f"Reports and figures written to {out_dir}")


if __name__ == "__main__":
    main()
