"""
plot_knockouts.py
-----------------
Two knockout plots for PIGLasso/SSGLasso real burn data results:

  1. Knockout impact bar chart  (knockout_bar.pdf/png)
     Top N genes by perturbative impact score, coloured by biological category.

  2. Delta vs knockout scatter  (delta_vs_knockout.pdf/png)
     X: |delta| (direct expression signal), Y: knockout impact score.
     Highlights genes that are both strongly expressed AND network-influential.

Usage:
    cd BurnInjuries/
    python PIGLasso/pipeline_src/knockouts/plot_knockouts.py
    python PIGLasso/pipeline_src/knockouts/plot_knockouts.py --dataset GSE37069 --model SSGLasso
"""

import argparse
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
    "font.size":       9,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DIFF = os.path.join(_HERE, "..", "diffusion")

TOP_N      = 20    # genes shown in bar chart
LABEL_N    = 15    # genes labelled in scatter

REDUCTION  = 0.1   # must match the reduction used in run_knockout.sh

BIO_CATEGORIES = {
    "immune": {
        "color": "#B4436C",
        "genes": {
            "IL10", "EMR3", "CD86", "KLRD1", "KIR2DL2", "KIR3DL1",
            "TREM1", "TIGIT", "PRF1", "MERTK", "PLA2G7", "FCRL1",
            "S1PR5", "FASLG", "TNFRSF10D", "PIK3R5", "TRPV2",
            "AREG",
        },
    },
    "signalling": {
        "color": "#4C72B0",
        "genes": {
            "IKBKB", "RAF1", "RPS6KA5", "PPP1R12B", "PPP1R15A",
            "STAT5A", "SOCS3", "TRIB1", "DUSP2", "PIM3", "HIF1A",
            "PIK3IP1", "RAPGEF2", "RGL1", "IRS2", "MARCKS",
            "IGF2R",
        },
    },
    "apoptosis": {
        "color": "#F78154",
        "genes": {
            "BTG1", "BTG2", "CABLES2", "DNAJB1", "FOSB", "CREM",
            "SERTAD2", "NR4A2", "CASC2", "TNFRSF10D",
            "RELL1", "KIAA0087",
        },
    },
    "metabolism": {
        "color": "#F2C14E",
        "genes": {
            "ABCG1", "ACSL3", "GCLM", "GRAMD1C", "L2HGDH", "TKTL1",
            "SLC1A3", "SLC45A4", "VNN3", "HAL", "ADPRH",
            "CHSY1",
        },
    },
    "epigenetic": {
        "color": "#5CAD6E",
        "genes": {
            "JARID2", "NCOA2", "PCGF3", "PHF13", "UBN1", "AUTS2",
            "PRDM8", "TOX4", "HMBOX1", "AKNA",
            "ZNF254",
        },
    },
}

UNCHARACTERISED_COLOR = "#264653"


def _gene_color(gene: str) -> str:
    for cat in BIO_CATEGORIES.values():
        if gene in cat["genes"]:
            return cat["color"]
    return UNCHARACTERISED_COLOR


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset: str, model: str):
    ko_dir   = os.path.join(_HERE, "results", dataset, model)
    diff_dir = os.path.join(_DIFF, "results", dataset, model, "diff_sig")

    ko_file = os.path.join(ko_dir,
                           f"perturbative_gene_impacts_reduction_{REDUCTION:.3f}.tsv")
    ko = pd.read_csv(ko_file, sep="\t")

    delta = pd.read_csv(os.path.join(diff_dir, "delta.tsv"),
                        sep="\t", index_col=0).squeeze("columns")
    delta.name = "delta"

    return ko, delta


# ---------------------------------------------------------------------------
# Plot 1 — Knockout impact bar chart
# ---------------------------------------------------------------------------

def plot_knockout_bar(ko: pd.DataFrame, out_stem: str, dataset: str,
                      model: str, dpi: int = 200):
    top    = ko.head(TOP_N).copy()
    genes  = top["gene"].tolist()
    scores = top["impact_max_l2"].values
    colors = [_gene_color(g) for g in genes]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    ax.barh(range(len(genes)), scores, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=8.5)
    ax.invert_yaxis()

    ax.set_xlabel("Perturbative impact  (max‖ΔS(t)‖₂)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", lw=0.4, alpha=0.4)

    # Category legend
    handles = [
        mpatches.Patch(facecolor=cat["color"], edgecolor="none",
                       label=name.capitalize())
        for name, cat in BIO_CATEGORIES.items()
    ]
    handles.append(mpatches.Patch(facecolor=UNCHARACTERISED_COLOR,
                                  edgecolor="none", label="Uncharacterised"))
    ax.legend(handles=handles, fontsize=7.5, frameon=True,
              framealpha=0.9, edgecolor="#cccccc", loc="lower right",
              labelspacing=0.4)

    fig.suptitle(f"Top {TOP_N} genes by knockout impact", fontsize=12,
                 fontweight="bold", y=0.995)
    ax.set_title(f"{dataset}  ·  acute phase  ·  model = {model}  ·  "
                 f"reduction = {REDUCTION}",
                 fontsize=8, color="black", pad=2)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = f"{out_stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — Delta vs knockout scatter
# ---------------------------------------------------------------------------

def plot_delta_vs_knockout(ko: pd.DataFrame, delta: pd.Series,
                           out_stem: str, dataset: str, model: str,
                           dpi: int = 200):
    merged = ko.set_index("gene").join(delta.abs().rename("abs_delta"),
                                       how="inner")
    merged.columns = ["impact", "abs_delta"]

    genes  = merged.index.tolist()
    x      = merged["abs_delta"].values
    y      = merged["impact"].values
    colors = [_gene_color(g) for g in genes]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)

    ax.scatter(x, y, c=colors, s=40, alpha=0.80, edgecolors="none", zorder=3)

    # Label top LABEL_N genes by combined rank (impact + abs_delta normalised)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
    combined = x_norm + y_norm
    top_idx  = np.argsort(combined)[::-1][:LABEL_N]

    for i in top_idx:
        ax.annotate(
            genes[i],
            xy=(x[i], y[i]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#222222", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="none", alpha=0.7),
        )

    ax.set_xlabel("|Δ expression|  (|burn mean − ctrl mean|)", fontsize=10)
    ax.set_ylabel("Knockout impact  (max‖ΔS(t)‖₂)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(lw=0.4, alpha=0.3)

    # Category legend
    handles = [
        mpatches.Patch(facecolor=cat["color"], edgecolor="none",
                       label=name.capitalize())
        for name, cat in BIO_CATEGORIES.items()
    ]
    handles.append(mpatches.Patch(facecolor=UNCHARACTERISED_COLOR,
                                  edgecolor="none", label="Uncharacterised"))
    ax.legend(handles=handles, fontsize=7.5, frameon=True,
              framealpha=0.9, edgecolor="#cccccc",
              loc="upper left", labelspacing=0.4)

    fig.suptitle("|Δ expression| vs knockout impact", fontsize=12,
                 fontweight="bold", y=0.995)
    ax.set_title(f"{dataset}  ·  acute phase  ·  model = {model}",
                 fontsize=8, color="black", pad=2)

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
    ko, delta = load_data(args.dataset, args.model)

    print("Plotting knockout bar chart …")
    plot_knockout_bar(ko,
                      os.path.join(out_dir, "knockout_bar"),
                      args.dataset, args.model, args.dpi)

    print("Plotting delta vs knockout scatter …")
    plot_delta_vs_knockout(ko, delta,
                           os.path.join(out_dir, "delta_vs_knockout"),
                           args.dataset, args.model, args.dpi)


if __name__ == "__main__":
    main()
