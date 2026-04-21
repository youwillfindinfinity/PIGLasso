"""
plot_diffusion.py
-----------------
Two diffusion plots for PIGLasso/SSGLasso real burn data results:

  1. Delta bar chart  (delta_bar.pdf/png)
     Top N genes by |delta| (burn mean − ctrl mean), coloured positive/negative.
     Shows the input signal before network propagation.

  2. Diffusion heatmap  (diffusion_heatmap.pdf/png)
     Top N genes (by |delta|) × early time points.
     Reveals how signal spreads through the network over time.

Usage:
    cd BurnInjuries/
    python PIGLasso/pipeline_src/diffusion/plot_diffusion.py
    python PIGLasso/pipeline_src/diffusion/plot_diffusion.py --dataset GSE37069 --model SSGLasso
"""

import argparse
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       9,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

TOP_N         = 20   # genes shown in bar chart
HEATMAP_N     = 30   # genes shown in heatmap
HEATMAP_T_MAX = 40   # number of time columns to show (early dynamics are most informative)

POS_COLOR = "#E8527A"   # rose-red  — upregulated in burn
NEG_COLOR = "#4D9078"   # teal      — downregulated in burn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset: str, model: str):
    diff_sig_dir = os.path.join(_HERE, "results", dataset, model, "diff_sig")
    net_diff_dir = os.path.join(_HERE, "results", dataset, model, "net_diff")

    delta = pd.read_csv(os.path.join(diff_sig_dir, "delta.tsv"),
                        sep="\t", index_col=0).squeeze("columns")
    delta.name = "delta"

    diff = pd.read_csv(os.path.join(net_diff_dir, "diffused_signal_genes_x_t.tsv"),
                       sep="\t", index_col=0)

    return delta, diff


# ---------------------------------------------------------------------------
# Plot 1 — Delta bar chart
# ---------------------------------------------------------------------------

def plot_delta_bar(delta: pd.Series, out_stem: str, dataset: str,
                   model: str, dpi: int = 200):
    top = delta.abs().sort_values(ascending=False).head(TOP_N)
    genes  = top.index.tolist()
    values = delta.loc[genes].values
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in values]

    # Sort by value (descending absolute, but keep sign for visual order)
    order  = np.argsort(np.abs(values))[::-1]
    genes  = [genes[i] for i in order]
    values = values[order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    bars = ax.barh(range(len(genes)), values, color=colors, edgecolor="none",
                   height=0.7)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=8.5)
    ax.invert_yaxis()

    ax.axvline(0, color="black", lw=0.8, zorder=3)
    ax.set_xlabel("Δ expression  (burn mean − ctrl mean)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", lw=0.4, alpha=0.4)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=POS_COLOR, label="Upregulated in burn"),
        Patch(facecolor=NEG_COLOR, label="Downregulated in burn"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, frameon=True,
              framealpha=0.9, edgecolor="#cccccc", loc="lower right")

    ax.set_title(f"Top {TOP_N} differentially expressed genes", fontsize=12,
                 fontweight="bold", pad=4)
    ax.text(0.5, 1.01,
            f"{dataset}  ·  acute phase  ·  model = {model}  ·  "
            f"Δ = burn mean − GSE37069 ctrl",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color="black")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = f"{out_stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — Diffusion heatmap
# ---------------------------------------------------------------------------

def plot_diffusion_heatmap(delta: pd.Series, diff: pd.DataFrame,
                           out_stem: str, dataset: str, model: str,
                           dpi: int = 200):
    # Select top genes by |delta|
    top_genes = delta.abs().sort_values(ascending=False).head(HEATMAP_N).index.tolist()
    top_genes = [g for g in top_genes if g in diff.index]

    # Restrict to early time columns
    n_cols = min(HEATMAP_T_MAX, diff.shape[1])
    sub = diff.loc[top_genes, diff.columns[:n_cols]]

    # Parse numeric t values for axis labels
    t_vals = np.array([float(c.replace("t=", "")) for c in sub.columns])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=dpi)

    vmax = np.abs(sub.values).max()
    im = ax.imshow(sub.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # Gene labels on y-axis
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=8)

    # Time labels on x-axis — show ~6 ticks
    n_ticks = 6
    tick_idx = np.linspace(0, n_cols - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{t_vals[i]:.2f}" for i in tick_idx], fontsize=8)
    ax.set_xlabel("Diffusion time  t", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Diffused signal", fontsize=9)

    ax.set_title("Network diffusion signal over time", fontsize=12,
                 fontweight="bold", pad=4)
    ax.text(0.5, 1.01,
            f"{dataset}  ·  acute phase  ·  model = {model}  ·  "
            f"top {HEATMAP_N} genes by |Δ|  ·  t ∈ [0, {t_vals[n_cols-1]:.2f}]",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color="black")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = f"{out_stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="GSE182616")
    parser.add_argument("--model",   default="PIGLasso",
                        help="PIGLasso or SSGLasso")
    parser.add_argument("--dpi",     type=int, default=200)
    args = parser.parse_args()

    out_dir = os.path.join(_HERE, "results", args.dataset, args.model, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data  ({args.dataset} / {args.model}) …")
    delta, diff = load_data(args.dataset, args.model)

    print("Plotting delta bar chart …")
    plot_delta_bar(delta,
                   os.path.join(out_dir, "delta_bar"),
                   args.dataset, args.model, args.dpi)

    print("Plotting diffusion heatmap …")
    plot_diffusion_heatmap(delta, diff,
                           os.path.join(out_dir, "diffusion_heatmap"),
                           args.dataset, args.model, args.dpi)


if __name__ == "__main__":
    main()
