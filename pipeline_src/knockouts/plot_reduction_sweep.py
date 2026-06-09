"""
plot_reduction_sweep.py
-----------------------
Two-panel sensitivity figure for the knockout reduction-factor sweep.

  Panel A: Spearman ρ vs reduction factor
    Gene-ranking correlation against the r=0.10 reference, showing how
    sensitive the perturbation ranking is to the edge-weight reduction.

  Panel B: Top-10 rank trajectories
    The 10 highest-impact genes at r=0.10 tracked across all reduction
    factors.  Y-axis inverted so rank 1 sits at the top.

Usage:
    cd BurnInjuries/
    python PIGLasso/pipeline_src/knockouts/run_reduction_sweep.py  # first
    python PIGLasso/pipeline_src/knockouts/plot_reduction_sweep.py
    python PIGLasso/pipeline_src/knockouts/plot_reduction_sweep.py --out path/to/fig.pdf
"""

import argparse
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       11,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

_HERE       = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR   = os.path.join(_HERE, "results", "GSE182616", "PIGLasso", "reduction_sweep")
FIGURES_DIR = os.path.join(_HERE, "results", "GSE182616", "PIGLasso", "figures")

REDUCTIONS  = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
REFERENCE   = 0.10
TOP_N       = 10

PALETTE = [
    "#B4436C", "#2C7BB6", "#4D9078", "#F78154",
    "#7B5EA7", "#C89A3A", "#2E86AB", "#A23B72",
    "#56B870", "#E05C5C",
]


# ── I/O ─────────────────────────────────────────────────────────────────────

def reduction_tag(r: float) -> str:
    return f"r{round(r * 100):03d}"


def tsv_path(r: float) -> str:
    tag = reduction_tag(r)
    return os.path.join(SWEEP_DIR, tag,
                        f"perturbative_gene_impacts_reduction_{r:.3f}.tsv")


def load_all() -> dict:
    data = {}
    for r in REDUCTIONS:
        path = tsv_path(r)
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            df = df.sort_values("impact_max_l2", ascending=False).reset_index(drop=True)
            df["rank"] = df.index + 1
            data[r] = df
        else:
            print(f"[WARN] Missing: {path}")
    return data


# ── Computations ─────────────────────────────────────────────────────────────

def compute_spearman(data: dict) -> tuple:
    ref = data[REFERENCE].set_index("gene")["impact_max_l2"]
    xs, ys = [], []
    for r in REDUCTIONS:
        if r not in data:
            continue
        cur    = data[r].set_index("gene")["impact_max_l2"]
        common = ref.index.intersection(cur.index)
        rho, _ = spearmanr(ref.loc[common], cur.loc[common])
        xs.append(r)
        ys.append(rho)
    return xs, ys


# ── Figure ────────────────────────────────────────────────────────────────────

def build_figure(data: dict) -> plt.Figure:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.subplots_adjust(left=0.07, right=0.82, top=0.89, bottom=0.14, wspace=0.38)

    reductions_present = [r for r in REDUCTIONS if r in data]
    r_labels = [str(r) for r in reductions_present]

    # ── Panel A ──────────────────────────────────────────────────────────────
    xs, ys = compute_spearman(data)

    ax_a.plot(range(len(xs)), ys,
              color="#2C7BB6", marker="o", markersize=6,
              linewidth=2.0, zorder=3, clip_on=False)

    for i, (x, y) in enumerate(zip(xs, ys)):
        is_ref = abs(x - REFERENCE) < 1e-9
        color  = "#D7191C" if is_ref else "#2C7BB6"
        ax_a.scatter([i], [y], color=color, zorder=5, s=55)
        ax_a.annotate(f"{y:.3f}", xy=(i, y),
                      xytext=(0, 9), textcoords="offset points",
                      ha="center", fontsize=8.5, color=color)

    ax_a.axhline(1.0, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_a.axvline(xs.index(REFERENCE), color="#D7191C",
                 linewidth=0.9, linestyle=":", alpha=0.55, zorder=1)

    ax_a.set_xticks(range(len(xs)))
    ax_a.set_xticklabels(r_labels)
    ax_a.set_xlabel("Reduction factor")
    ax_a.set_ylabel(f"Spearman ρ  (vs r = {REFERENCE})")
    ax_a.set_ylim(max(0, min(ys) - 0.05), 1.03)
    ax_a.set_title("A   Ranking stability across reduction factors",
                   loc="left", fontsize=11, fontweight="bold")

    # ── Panel B ──────────────────────────────────────────────────────────────
    ref_top = data[REFERENCE].head(TOP_N)["gene"].tolist()
    gene_colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(ref_top)}

    for g in ref_top:
        rx, ry = [], []
        for r in reductions_present:
            row = data[r][data[r]["gene"] == g]
            if not row.empty:
                rx.append(reductions_present.index(r))
                ry.append(int(row["rank"].values[0]))

        color = gene_colors[g]
        ax_b.plot(rx, ry, color=color, marker="o",
                  markersize=4.5, linewidth=1.6, alpha=0.9, zorder=3)

    # Stagger right-side labels to avoid overlap
    if reductions_present:
        last_idx = len(reductions_present) - 1
        label_data = []
        for g in ref_top:
            row = data[reductions_present[-1]][data[reductions_present[-1]]["gene"] == g]
            if not row.empty:
                label_data.append((int(row["rank"].values[0]), g))
        label_data.sort(key=lambda t: t[0])

        used_y = []
        MIN_SEP = 0.55
        for rank_val, g in label_data:
            y_pos = float(rank_val)
            for prev in used_y:
                if abs(y_pos - prev) < MIN_SEP:
                    y_pos = prev + MIN_SEP
            used_y.append(y_pos)
            ax_b.annotate(
                g,
                xy=(last_idx, rank_val),
                xytext=(last_idx + 0.35, y_pos),
                fontsize=8.5, va="center",
                color=gene_colors[g], fontweight="bold",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=gene_colors[g],
                                lw=0.6, alpha=0.5),
            )

    ax_b.set_xlim(-0.3, last_idx + 0.2)
    ax_b.axvline(reductions_present.index(REFERENCE),
                 color="#888888", linewidth=1.0, linestyle=":", alpha=0.6, zorder=1)

    ax_b.set_xticks(range(len(reductions_present)))
    ax_b.set_xticklabels(r_labels)
    ax_b.set_xlabel("Reduction factor")
    ax_b.set_ylabel("Rank  (1 = highest impact)")
    ax_b.invert_yaxis()
    ax_b.set_title(f"B   Top-{TOP_N} gene rank trajectories",
                   loc="left", fontsize=11, fontweight="bold")

    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None,
                        help="Output path (default: figures/reduction_sweep.pdf)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    print("Loading reduction sweep results …")
    data = load_all()

    if REFERENCE not in data:
        raise FileNotFoundError(
            f"Reference reduction r={REFERENCE} not found in {SWEEP_DIR}.\n"
            "Run run_reduction_sweep.py first."
        )

    present = sorted(data.keys())
    print(f"  {len(data)}/{len(REDUCTIONS)} reductions loaded: {present}")

    print("Building figure …")
    fig = build_figure(data)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_pdf = args.out or os.path.join(FIGURES_DIR, "reduction_sweep.pdf")
    out_png = out_pdf.replace(".pdf", ".png")

    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_png, dpi=150,      bbox_inches="tight")
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
