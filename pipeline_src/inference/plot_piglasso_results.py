"""
plot_piglasso_results.py
------------------------
Two figures for the PIGLasso-with-prior real burn data run (GSE182616, acute phase):

  1. Lambda stability path  (lambda_path.pdf/png)
     Number of stable edges at three thresholds (0.5 / 0.8 / 1.0) across the
     full lambda range.  Prior run (solid) overlaid on no-prior run (dashed)
     to show the effect of the prior on edge stability.

  2. Network figure  (piglasso_network.pdf/png)
     Dark-background network styled after NODIS burns_network.py.
     Edges: stability = 1.0 at lambda = 0.30.
     Nodes: coloured by biological category, sized by degree.
     Layout: shell stratified by degree (hubs centre, peripherals outer ring).

Usage:
    cd BurnInjuries/
    python inference/plotting/plot_piglasso_results.py
    python inference/plotting/plot_piglasso_results.py --no-network   # path only
    python inference/plotting/plot_piglasso_results.py --no-path      # network only
"""

import argparse
import os
import pickle
import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
# Paths
# ---------------------------------------------------------------------------
_HERE        = os.path.dirname(os.path.abspath(__file__))
PIGLASSO_DIR = os.path.join(_HERE, "results", "piglasso", "GSE182616")
FIGURES_DIR  = os.path.join(_HERE, "results", "figures")

PKL_PRIOR   = os.path.join(
    PIGLASSO_DIR,
    "PHASE__Acute__n513__zscored__filtered__Q200__bperc0.65__lam0.05-0.3x20"
    "__seed42__pw0.5__piglasso_results.pkl",
)
PKL_NOPRIOR = os.path.join(
    PIGLASSO_DIR,
    "PHASE__Acute__n513__zscored__filtered__Q200__bperc0.65__lam0.05-0.3x20"
    "__piglasso_results.pkl",
)

# Lambda to extract the network from (most penalising = sparsest)
NETWORK_LAMBDA = 0.30
# Stability threshold for the network edges
NETWORK_STAB_THRESH = 1.0

# ---------------------------------------------------------------------------
# Biological categories — node fill colour
# ---------------------------------------------------------------------------
BIO_CATEGORIES = {
    "immune": {
        "color": "#CC4778",  # pink
        "genes": {
            "IL10", "EMR3", "CD86", "KLRD1", "KIR2DL2", "KIR3DL1",
            "TREM1", "TIGIT", "PRF1", "MERTK", "PLA2G7", "FCRL1",
            "S1PR5", "FASLG", "TNFRSF10D", "PIK3R5", "TRPV2",
        },
    },
    "signalling": {
        "color": "#4C81D9",  # blue
        "genes": {
            "IKBKB", "RAF1", "RPS6KA5", "PPP1R12B", "PPP1R15A",
            "STAT5A", "SOCS3", "TRIB1", "DUSP2", "PIM3", "HIF1A",
            "PIK3IP1", "RAPGEF2", "RGL1", "IRS2", "MARCKS",
        },
    },
    "apoptosis": {
        "color": "#F78154",  # orange
        "genes": {
            "BTG1", "BTG2", "CABLES2", "DNAJB1", "FOSB", "CREM",
            "SERTAD2", "NR4A2", "CASC2", "TNFRSF10D",
        },
    },
    "metabolism": {
        "color": "#F2C14E",  # yellow
        "genes": {
            "ABCG1", "ACSL3", "GCLM", "GRAMD1C", "L2HGDH", "TKTL1",
            "SLC1A3", "SLC45A4", "VNN3", "HAL", "ADPRH",
        },
    },
    "epigenetic": {
        "color": "#59CC47",  # green
        "genes": {
            "JARID2", "NCOA2", "PCGF3", "PHF13", "UBN1", "AUTS2",
            "PRDM8", "TOX4", "HMBOX1", "AKNA",
        },
    },
}

ALWAYS_LABEL = {
    "IL10", "HIF1A", "SOCS3", "TREM1", "PRF1", "TIGIT",
    "FOSB", "DUSP2", "RAF1", "IKBKB", "STAT5A", "RPS6KA5",
}


def _gene_color(gene: str) -> str:
    for cat in BIO_CATEGORIES.values():
        if gene in cat["genes"]:
            return cat["color"]
    return "#5BC9C4"  # light blue — uncharacterised


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pkl(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _stability_matrix(d: dict) -> np.ndarray:
    """Return (p, p, lamlen) stability array in [0, 1]."""
    return d["edge_counts_all"] / d["Q"]


def _edge_counts_per_lambda(stab: np.ndarray, thresholds) -> dict:
    """For each threshold, return array of edge counts across lambda axis."""
    p = stab.shape[0]
    idx = np.triu_indices(p, k=1)
    result = {}
    for t in thresholds:
        counts = [int(np.sum(stab[:, :, li][idx] > t)) for li in range(stab.shape[2])]
        result[t] = np.array(counts)
    return result


# ---------------------------------------------------------------------------
# Plot 1 — Lambda stability path
# ---------------------------------------------------------------------------

def plot_lambda_path(out_stem: str, dpi: int = 200):
    d_prior   = _load_pkl(PKL_PRIOR)
    d_noprior = _load_pkl(PKL_NOPRIOR)

    lam = d_prior["lambda_range"]
    s_prior   = _stability_matrix(d_prior)
    s_noprior = _stability_matrix(d_noprior)

    thresholds = [0.5, 0.8, 1.0]
    thr_colors = {0.5: "#B4436C", 0.8: "#5CAD6E", 1.0: "#F78154"}

    counts_prior   = _edge_counts_per_lambda(s_prior,   thresholds)
    counts_noprior = _edge_counts_per_lambda(s_noprior, thresholds)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)

    for t in thresholds:
        c = thr_colors[t]
        label_p  = f"stability > {t}" if t < 1.0 else "stability = 1.0"
        label_np = f"stability > {t} (no prior)" if t < 1.0 else "stability = 1.0 (no prior)"
        ax.plot(lam, counts_prior[t],   color=c, lw=1.8,  ls="-",  label=f"{label_p}")
        ax.plot(lam, counts_noprior[t], color=c, lw=1.4,  ls="--", label=f"{label_np}")

    # Vertical line at chosen network lambda
    ax.axvline(NETWORK_LAMBDA, color="#4C72B0", lw=1.0, ls=":", alpha=0.8,
               label=f"network λ = {NETWORK_LAMBDA}")

    ax.set_xlabel("Regularisation parameter λ", fontsize=10)
    ax.set_ylabel("Number of stable edges", fontsize=10)
    ax.set_title(
        "GSE182616  ·  acute phase  ·  n = 513, p = 164  ·  "
        "solid = with prior (pw = 0.5)  ·  dashed = no prior",
        fontsize=8, color="black", pad=2,
    )
    fig.suptitle("PIGLasso stability path", fontsize=12, fontweight="bold", y=0.995)

    # Clean legend: deduplicate and order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7.5, frameon=True,
              framealpha=0.9, edgecolor="#cccccc", ncol=2,
              loc="upper right", labelspacing=0.4)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = f"{out_stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — Network figure
# ---------------------------------------------------------------------------

def _compute_layout(genes: list, degree: np.ndarray) -> np.ndarray:
    n = len(genes)
    mu, sd   = degree.mean(), degree.std()
    thr_hub  = mu + 2 * sd
    thr_high = mu + sd

    hubs       = [i for i in range(n) if degree[i] >= thr_hub]
    high       = [i for i in range(n) if thr_high  <= degree[i] < thr_hub]
    connected  = [i for i in range(n) if mu         <= degree[i] < thr_high]
    peripheral = [i for i in range(n) if degree[i]  < mu]

    pos = np.zeros((n, 2))
    rng = np.random.default_rng(42)

    def _ring(indices, radius, jitter=0.0):
        k = len(indices)
        if k == 0:
            return
        indices = sorted(indices, key=lambda i: -degree[i])
        offset  = rng.uniform(0, 2 * np.pi)
        for rank, i in enumerate(indices):
            angle      = offset + 2 * np.pi * rank / k
            r          = radius + rng.uniform(-jitter, jitter)
            pos[i, 0]  = r * np.cos(angle)
            pos[i, 1]  = r * np.sin(angle)

    _ring(hubs,       radius=0.30, jitter=0.06)
    _ring(high,       radius=0.65, jitter=0.08)
    _ring(connected,  radius=1.10, jitter=0.10)
    _ring(peripheral, radius=1.65, jitter=0.12)
    return pos


def plot_network(out_stem: str, dpi: int = 200):
    d = _load_pkl(PKL_PRIOR)

    genes   = d["genes"]
    lam     = d["lambda_range"]
    stab    = _stability_matrix(d)

    # Find lambda index closest to NETWORK_LAMBDA
    lam_idx = int(np.argmin(np.abs(lam - NETWORK_LAMBDA)))
    lam_val = float(lam[lam_idx])
    print(f"  Using lambda[{lam_idx}] = {lam_val:.4f}")

    # Build edge list: (i, j, stability) where stability >= threshold
    stab_at_lam = stab[:, :, lam_idx]
    p           = len(genes)
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            s = stab_at_lam[i, j]
            if s >= NETWORK_STAB_THRESH:
                edges.append((i, j, float(s)))

    # Degree from this edge set
    degree = np.zeros(p, dtype=int)
    for i, j, _ in edges:
        degree[i] += 1
        degree[j] += 1

    pos          = _compute_layout(genes, degree)
    node_colors  = [_gene_color(g) for g in genes]
    node_sizes   = [60 + d ** 1.85 for d in degree]
    mu, sd       = degree.mean(), degree.std()
    thr_hub      = mu + 2 * sd
    thr_high     = mu + sd

    BG = "#0f0f1a"
    fig, ax = plt.subplots(figsize=(14, 11), dpi=dpi)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # Edge alpha / width scaled by stability.
    # When all edges share the same value (stability=1.0) we fall back to a
    # fixed mid-range alpha so edges are clearly visible.
    stab_vals = np.array([s for _, _, s in edges])
    smin, smax = stab_vals.min(), stab_vals.max()
    s_range = smax - smin

    # --- Edges ---
    for i, j, s in edges:
        if s_range > 1e-9:
            t = (s - smin) / s_range
        else:
            t = 0.65   # all-equal case: fixed comfortable alpha
        alpha = 0.10 + 0.45 * t
        lw    = 0.30 + 1.40 * t
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color="#aaaacc", alpha=alpha, linewidth=lw, zorder=1)

    # --- Nodes ---
    for i, g in enumerate(genes):
        c  = node_colors[i]
        s  = node_sizes[i]
        dg = degree[i]
        zo = 5 if dg >= thr_hub else (4 if dg >= thr_high else 3)
        ew = 1.5 if dg >= thr_hub else (0.8 if dg >= thr_high else 0.3)
        ec = "white" if dg >= thr_hub else ("white" if dg >= thr_high else c)
        ax.scatter(pos[i, 0], pos[i, 1], s=s, c=c,
                   edgecolors=ec, linewidths=ew, zorder=zo, alpha=0.92)

    # --- Labels ---
    deg_series  = pd.Series(degree, index=genes)
    top6_hubs   = set(deg_series.nlargest(6).index)
    to_label    = top6_hubs | {g for g in ALWAYS_LABEL if g in genes}
    gene_idx    = {g: i for i, g in enumerate(genes)}

    for g in to_label:
        if g not in gene_idx:
            continue
        i  = gene_idx[g]
        dg = degree[i]
        fw = "bold"  if dg >= thr_hub  else "normal"
        fs = 7.5     if dg >= thr_hub  else 6.0
        fc = "white" if dg >= thr_hub  else "#dddddd"
        bc = _gene_color(g)
        ax.annotate(
            g,
            xy=(pos[i, 0], pos[i, 1]),
            xytext=(pos[i, 0] + 0.025, pos[i, 1] + 0.025),
            fontsize=fs, fontweight=fw, color=fc, zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", fc=bc, ec="none", alpha=0.75),
        )

    # --- Category legend ---
    handles = [
        mpatches.Patch(facecolor=cat["color"], edgecolor="white",
                       linewidth=0.5, label=name.capitalize())
        for name, cat in BIO_CATEGORIES.items()
    ]
    handles.append(
        mpatches.Patch(facecolor="#5BC9C4", edgecolor="white",
                       linewidth=0.5, label="Uncharacterised")
    )
    leg = ax.legend(handles=handles, loc="lower left", fontsize=12,
                    frameon=True, framealpha=0.25, edgecolor="#555555",
                    facecolor="#111122", labelcolor="white",
                    handlelength=1.6, handleheight=1.4, labelspacing=0.7,
                    title="Category", title_fontsize=13)
    leg.get_title().set_color("white")

    # --- Node size legend ---
    deg_examples = [(5, "deg = 5"), (15, "deg = 15"), (30, "deg = 30 (hub)")]
    for dex, lbl in deg_examples:
        ax.scatter([], [], s=60 + dex ** 1.85, c="white", alpha=0.7, label=lbl)
    size_leg = ax.legend(loc="lower right", fontsize=11, frameon=True,
                         framealpha=0.25, edgecolor="#555555",
                         facecolor="#111122", labelcolor="white",
                         title="Node size", title_fontsize=12,
                         scatterpoints=1, labelspacing=0.8)
    size_leg.get_title().set_color("white")
    ax.add_artist(leg)

    # --- Stats box ---
    n_isolated = int(np.sum(degree == 0))
    ax.text(
        0.01, 0.99,
        (
            f"n = 513 samples  ·  p = {p} genes\n"
            f"Edges: {len(edges)} (stability = 1.0 at λ = {lam_val:.2f})\n"
            f"Mean degree: {mu:.1f}  ·  Isolated nodes: {n_isolated}\n"
            f"Prior weight: 0.5  ·  Q = 200 subsamples"
        ),
        transform=ax.transAxes, va="top", ha="left",
        fontsize=13, color="#aaaaaa",
        bbox=dict(boxstyle="round,pad=0.6", fc="#111122",
                  ec="#444444", alpha=0.8),
    )

    ax.set_axis_off()
    ax.set_title(
        "PIGLasso Co-expression Network  ·  GSE182616 (Acute Phase)\n"
        f"Stability-based GGM with prior  ·  Stability = 1.0  ·  λ = {lam_val:.2f}  ·  n/p = 3.13",
        color="white", pad=10, fontsize=13.5, fontweight="bold",
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p_out = f"{out_stem}.{ext}"
        fig.savefig(p_out, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {p_out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-network", action="store_true",
                        help="Skip the network figure")
    parser.add_argument("--no-path",    action="store_true",
                        help="Skip the lambda stability path figure")
    parser.add_argument("--dpi",        type=int, default=200)
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    if not args.no_path:
        print("Plotting lambda stability path …")
        plot_lambda_path(os.path.join(FIGURES_DIR, "lambda_path"), dpi=args.dpi)

    if not args.no_network:
        print("Plotting network figure …")
        plot_network(os.path.join(FIGURES_DIR, "piglasso_network"), dpi=args.dpi)


if __name__ == "__main__":
    main()
