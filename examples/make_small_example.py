"""
Generate small synthetic data for the PIGLasso example pipeline.

Writes to examples/data/small_example/:
  expression.tsv      — 20 genes × 25 samples (genes as rows)
  genes.txt           — 20 gene names
  prior.npy           — (20, 20) prior matrix
  adjacency.csv       — (20, 20) binary adjacency (for diffusion / knockout)
  stability.csv       — (20, 20) stability scores
  delta.tsv           — 20-gene delta expression vector

Run:
    python examples/make_small_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_SAMPLES = 25
N_GENES = 20
OUT_DIR = Path(__file__).parent / "data" / "small_example"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    genes = [f"GENE{i:02d}" for i in range(N_GENES)]
    samples = [f"S{i:02d}" for i in range(N_SAMPLES)]

    # ---------- expression (genes × samples) ----------
    # Two blocks of correlated genes to mimic real co-expression structure
    Omega = np.eye(N_GENES) * 2.0
    for i in range(9):
        Omega[i, i + 1] = Omega[i + 1, i] = 0.5
    for i in range(10, 19):
        Omega[i, i + 1] = Omega[i + 1, i] = 0.5
    Sigma = np.linalg.inv(Omega)
    X = rng.multivariate_normal(np.zeros(N_GENES), Sigma, size=N_SAMPLES)
    expr_df = pd.DataFrame(X.T, index=genes, columns=samples)
    expr_df.to_csv(OUT_DIR / "expression.tsv", sep="\t")
    print(f"expression.tsv  : {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")

    # ---------- gene list ----------
    (OUT_DIR / "genes.txt").write_text("\n".join(genes) + "\n")
    print(f"genes.txt       : {N_GENES} genes")

    # ---------- prior (sparse, block-structured) ----------
    prior = np.zeros((N_GENES, N_GENES))
    for i in range(9):
        prior[i, i + 1] = prior[i + 1, i] = 0.7
    for i in range(10, 19):
        prior[i, i + 1] = prior[i + 1, i] = 0.6
    np.fill_diagonal(prior, 0.0)
    np.save(OUT_DIR / "prior.npy", prior)
    print(f"prior.npy       : shape {prior.shape}, density {np.mean(prior > 0):.3f}")

    # ---------- adjacency + stability (derived from true precision) ----------
    adj = (np.abs(Omega) > 0.4).astype(float)
    np.fill_diagonal(adj, 0.0)
    adj_df = pd.DataFrame(adj, index=genes, columns=genes)
    adj_df.to_csv(OUT_DIR / "adjacency.csv")
    print(f"adjacency.csv   : {int(adj.sum() / 2)} edges")

    stab = adj * rng.uniform(0.6, 0.95, size=adj.shape)
    stab = (stab + stab.T) / 2
    np.fill_diagonal(stab, 0.0)
    stab_df = pd.DataFrame(stab, index=genes, columns=genes)
    stab_df.to_csv(OUT_DIR / "stability.csv")
    print(f"stability.csv   : stability scores in [{stab[stab > 0].min():.2f}, {stab.max():.2f}]")

    # ---------- delta expression vector ----------
    # Simulate burn-vs-control delta: genes in block 1 are up-regulated
    delta = np.zeros(N_GENES)
    delta[:10] = rng.normal(loc=1.5, scale=0.5, size=10)
    delta[10:] = rng.normal(loc=0.0, scale=0.3, size=10)
    delta_df = pd.DataFrame({"delta": delta}, index=genes)
    delta_df.index.name = "gene"
    delta_df.to_csv(OUT_DIR / "delta.tsv", sep="\t")
    print(f"delta.tsv       : {N_GENES} genes, mean delta block1={delta[:10].mean():.2f}")

    print(f"\nAll files written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
