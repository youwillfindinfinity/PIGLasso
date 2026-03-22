"""
Filter expression matrix to top N most variable genes and subset prior accordingly.

Input
-----
data/burn/expression.tsv    — genes × samples (rows = genes, cols = samples)
data/burn/genes.txt         — gene list (one per line, same order as expression rows)
data/burn/prior_burns.npy   — (p, p) prior matrix aligned to genes.txt

Output
------
data/burn/expression_top{N}.tsv    — filtered genes × samples
data/burn/genes_top{N}.txt         — selected gene list
data/burn/prior_burns_top{N}.npy   — (N, N) subsetted prior matrix

Usage
-----
python scripts/filter_top_genes.py --n-genes 5000 --out-dir data/burn/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter to top N most variable genes.")
    parser.add_argument("--n-genes", type=int, default=5000,
                        help="Number of top-variance genes to retain (default: 5000)")
    parser.add_argument("--expr", type=Path, default=Path("data/burn/expression.tsv"),
                        help="Path to expression TSV (genes x samples)")
    parser.add_argument("--genes", type=Path, default=Path("data/burn/genes.txt"),
                        help="Path to gene list (one per line)")
    parser.add_argument("--prior", type=Path, default=Path("data/burn/prior_burns.npy"),
                        help="Path to prior .npy (p x p, aligned to --genes)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/burn/"),
                        help="Output directory (default: data/burn/)")
    args = parser.parse_args()

    N = args.n_genes
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load expression (genes × samples)
    print(f"Loading expression: {args.expr}")
    expr = pd.read_csv(args.expr, sep="\t", index_col=0)
    n_genes_orig, n_samples = expr.shape
    print(f"  Shape: {n_genes_orig} genes × {n_samples} samples")

    # Compute per-gene variance (across samples)
    variances = expr.var(axis=1)

    # Select top N by variance
    top_idx = variances.nlargest(N).index
    expr_filtered = expr.loc[top_idx]
    print(f"  Retained top {N} genes by variance")

    # Write filtered expression TSV
    out_expr = out_dir / f"expression_top{N}.tsv"
    expr_filtered.to_csv(out_expr, sep="\t")
    print(f"  Written: {out_expr}")

    # Write gene list
    out_genes = out_dir / f"genes_top{N}.txt"
    out_genes.write_text("\n".join(top_idx.tolist()) + "\n")
    print(f"  Written: {out_genes}")

    # Subset prior
    print(f"Loading prior: {args.prior}")
    prior = np.load(args.prior)
    print(f"  Prior shape: {prior.shape}")

    # Map selected gene names to row indices in the original gene list
    orig_genes = args.genes.read_text().splitlines()
    gene_to_idx = {g: i for i, g in enumerate(orig_genes)}

    selected_genes = top_idx.tolist()
    missing = [g for g in selected_genes if g not in gene_to_idx]
    if missing:
        print(f"  WARNING: {len(missing)} selected genes not found in genes.txt — skipping")
        selected_genes = [g for g in selected_genes if g in gene_to_idx]

    idx = [gene_to_idx[g] for g in selected_genes]
    prior_sub = prior[np.ix_(idx, idx)]
    print(f"  Subsetted prior: {prior_sub.shape}")

    out_prior = out_dir / f"prior_burns_top{N}.npy"
    np.save(out_prior, prior_sub)
    print(f"  Written: {out_prior}")

    print(f"\nDone. Use expression_top{N}.tsv + prior_burns_top{N}.npy for PIGLasso.")


if __name__ == "__main__":
    main()
