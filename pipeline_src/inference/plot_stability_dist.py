"""
plot_stability_dist.py
----------------------
Plots the distribution of per-edge stability scores from the PIGLasso
inference on GSE182616 (acute burn phase).

Shows the bimodal separation between unstable (noise) edges near 0
and stable (signal) edges near 0.5–1.0, justifying the stability threshold.

Usage:
    cd BurnInjuries/
    python PIGLasso/pipeline_src/inference/plot_stability_dist.py
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       11,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

_HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "results", "network_inference", "GSE182616",
                        "stability_scores.csv")
FIG_DIR  = os.path.join(_HERE, "results", "network_inference", "GSE182616", "figures")

THRESHOLD  = 0.5          # stability selection threshold
COL_SEL    = "#B4436C"    # selected edges
COL_NOSEL  = "#AAAAAA"    # non-selected edges
BINS       = 40


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Not found: {CSV_PATH}\n"
            "Run extract_stability_scores.py on Snellius first, then transfer the CSV."
        )

    df = pd.read_csv(CSV_PATH)
    all_scores = df["stability"].values
    above = df.loc[df["stability"] >= THRESHOLD, "stability"].values
    below = df.loc[df["stability"] <  THRESHOLD, "stability"].values

    print(f"Total edges:        {len(df)}")
    print(f"Above threshold:    {len(above)}  (median {np.median(above):.3f})")
    print(f"Below threshold:    {len(below)}  (median {np.median(below):.3f})")

    bin_edges_full = np.linspace(0, 1, BINS + 1)
    bin_edges_zoom = np.linspace(THRESHOLD, 1, BINS // 2 + 1)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.14, wspace=0.35)

    # ── Panel A: full distribution, log y-scale, coloured by threshold ───────
    ax_a.hist(below, bins=bin_edges_full, color=COL_NOSEL, alpha=0.80,
              label=f"τ < {THRESHOLD}  (n = {len(below):,})", zorder=2)
    ax_a.hist(above, bins=bin_edges_full, color=COL_SEL, alpha=0.85,
              label=f"τ ≥ {THRESHOLD}  (n = {len(above):,})", zorder=3)
    ax_a.axvline(THRESHOLD, color="#222222", linewidth=1.3, linestyle="--",
                 zorder=4, label=f"Threshold  τ = {THRESHOLD}")

    ax_a.set_yscale("log")
    ax_a.set_xlabel("Stability score  (fraction of subsamples)")
    ax_a.set_ylabel("Number of edges  (log scale)")
    ax_a.set_xlim(0, 1)
    ax_a.set_title("A   All candidate edges", loc="left",
                   fontsize=11, fontweight="bold")
    ax_a.legend(frameon=False, loc="upper right", ncol=1,
                handlelength=1.2, fontsize=9.5)

    # ── Panel B: above-threshold edges only, linear scale ────────────────────
    ax_b.hist(above, bins=bin_edges_zoom, color=COL_SEL, alpha=0.85, zorder=2)

    med = np.median(above)
    ax_b.axvline(med, color="#222222", linewidth=1.3, linestyle="--", zorder=3)

    ax_b.set_xlabel("Stability score  (fraction of subsamples)")
    ax_b.set_ylabel("Number of edges")
    ax_b.set_xlim(THRESHOLD, 1)
    ax_b.set_title(f"B   Network edges  (τ ≥ {THRESHOLD},  n = {len(above):,})",
                   loc="left", fontsize=11, fontweight="bold")

    # Add median annotation after layout is stable
    fig.canvas.draw()
    ymax_b = ax_b.get_ylim()[1]
    ax_b.text(med + 0.008, ymax_b * 0.95,
              f"median = {med:.2f}", fontsize=9, va="top", color="#222222")

    os.makedirs(FIG_DIR, exist_ok=True)
    out_pdf = os.path.join(FIG_DIR, "stability_score_dist.pdf")
    out_png = os.path.join(FIG_DIR, "stability_score_dist.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
