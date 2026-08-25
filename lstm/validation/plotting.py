"""
Matplotlib figure generation for cross-framework validation reports.

All figures are saved as PNG files - nothing is shown interactively, so
these functions work in headless environments.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(history_tf, history_pt, save_path):
    """
    Overlay TF vs PyTorch training/validation loss curves per epoch, for the
    total loss and each per-output loss (latency_ms, pdr).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("loss", "val_loss", "Total loss"),
        ("latency_ms_loss", "val_latency_ms_loss", "Latency loss"),
        ("pdr_loss", "val_pdr_loss", "PDR loss"),
    ]
    for ax, (train_key, val_key, title) in zip(axes, panels):
        if train_key in history_tf:
            ax.plot(history_tf[train_key], label="TF train", color="tab:blue")
        if val_key in history_tf:
            ax.plot(history_tf[val_key], label="TF val", color="tab:blue", linestyle="--")
        if train_key in history_pt:
            ax.plot(history_pt[train_key], label="PT train", color="tab:orange")
        if val_key in history_pt:
            ax.plot(history_pt[val_key], label="PT val", color="tab:orange", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_weight_divergence(weight_divergence, save_path):
    """
    Plot the relative error between TF and PyTorch weights, per layer, over
    training epochs (stage='trained' comparison only).

    Args:
        weight_divergence: dict {layer_name: [rel_error_epoch0, rel_error_epoch1, ...]}
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for layer_name, values in weight_divergence.items():
        ax.plot(values, label=layer_name, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Median relative error (TF vs PT weights)")
    ax.set_yscale("log")
    ax.set_title("Weight divergence over training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_prediction_divergence_over_epochs(pred_divergence, save_path):
    """
    Plot the TF-vs-PT relative error on the held-out validation set's
    predictions, per epoch - mean and p99, for each output. Answers whether
    per-sample disagreement between the two independently-trained models
    grows, shrinks, or plateaus with more training (distinct from both the
    weight-space divergence and the aggregate loss curves).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [("latency_ms_mean", "latency_ms_p99", "latency_ms"),
              ("pdr_mean", "pdr_p99", "pdr")]
    for ax, (mean_key, p99_key, title) in zip(axes, panels):
        ax.plot(pred_divergence[mean_key], label="mean rel. error", marker="o", markersize=3)
        ax.plot(pred_divergence[p99_key], label="p99 rel. error", marker="s", markersize=3)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Relative error (TF vs PT predictions)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_prediction_comparison(y_true, pred_tf, pred_pt, save_path):
    """
    Scatter plots comparing predicted vs actual (both frameworks) and TF vs
    PT predictions directly, for each output.

    Args:
        y_true, pred_tf, pred_pt: dicts with 'latency_ms' and 'pdr' 1D arrays
    """
    outputs = [k for k in ("latency_ms", "pdr") if k in y_true]
    fig, axes = plt.subplots(2, len(outputs), figsize=(6 * len(outputs), 9))
    if len(outputs) == 1:
        axes = axes.reshape(2, 1)

    for col, key in enumerate(outputs):
        yt = np.asarray(y_true[key]).flatten()
        pt_tf = np.asarray(pred_tf[key]).flatten()
        pt_pt = np.asarray(pred_pt[key]).flatten()

        ax = axes[0, col]
        ax.scatter(yt, pt_tf, s=8, alpha=0.5, label="TF", color="tab:blue")
        ax.scatter(yt, pt_pt, s=8, alpha=0.5, label="PT", color="tab:orange")
        lims = [min(yt.min(), pt_tf.min(), pt_pt.min()), max(yt.max(), pt_tf.max(), pt_pt.max())]
        ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{key}: predicted vs actual")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=8)

        ax2 = axes[1, col]
        ax2.scatter(pt_tf, pt_pt, s=8, alpha=0.5, color="tab:green")
        lims2 = [min(pt_tf.min(), pt_pt.min()), max(pt_tf.max(), pt_pt.max())]
        ax2.plot(lims2, lims2, color="gray", linestyle="--", linewidth=1)
        ax2.set_title(f"{key}: TF vs PT prediction")
        ax2.set_xlabel("TF prediction")
        ax2.set_ylabel("PT prediction")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_chain_divergence(stage_names, mean_errors, max_errors, save_path):
    """
    Line plot of relative error (mean and max across the batch) at each stage
    of the network chain, to visually spot a sudden jump between two stages.
    """
    fig, ax = plt.subplots(figsize=(max(6, len(stage_names) * 1.2), 4.5))
    x = np.arange(len(stage_names))
    ax.plot(x, mean_errors, marker="o", label="mean rel. error")
    ax.plot(x, max_errors, marker="s", label="max rel. error")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names, rotation=30, ha="right")
    ax.set_ylabel("Relative error (TF vs PT)")
    ax.set_yscale("log")
    ax.set_title("Divergence along the network chain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_timestep_divergence(hidden_rel_error, cell_rel_error, save_path):
    """
    Per-timestep relative error of the hidden state (and cell state, if
    provided) - one line per sequence plus the mean across sequences, to
    visualize whether divergence accumulates timestep after timestep.

    Args:
        hidden_rel_error: array (n_sequences, timesteps)
        cell_rel_error: array (n_sequences, timesteps) or None (LSTM only)
    """
    n_panels = 2 if cell_rel_error is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    axes = np.atleast_1d(axes)

    def _plot(ax, data, title):
        for seq in data:
            ax.plot(seq, color="tab:gray", alpha=0.3, linewidth=1)
        ax.plot(data.mean(axis=0), color="tab:red", linewidth=2, label="mean")
        ax.set_yscale("log")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Relative error (TF vs PT)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    _plot(axes[0], hidden_rel_error, "Hidden state divergence per timestep")
    if cell_rel_error is not None:
        _plot(axes[1], cell_rel_error, "Cell state divergence per timestep")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
