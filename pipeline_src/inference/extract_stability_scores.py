"""
extract_stability_scores.py
---------------------------
Extracts per-edge stability scores from the raw PIGLasso pkl (source_piglasso_pkl)
and saves them as a flat CSV for local plotting.

Stability score for edge (i,j) = fraction of subsamples where that edge was
selected (non-zero) at the knee lambda.

Usage (on Snellius):
    cd /gpfs/home2/zblei/Documents/BurnInjuries/
    source .venv/bin/activate
    python PIGLasso/pipeline_src/inference/extract_stability_scores.py

Output:
    PIGLasso/pipeline_src/inference/results/network_inference/GSE182616/stability_scores.csv
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

PKL = Path(
    "/gpfs/home2/zblei/Documents/BurnInjuries/PIGLasso/pipeline_src/inference/"
    "results/piglasso/GSE182616/"
    "PHASE__Acute__n513__zscored__filtered__Q200__bperc0.65__lam0.05-1.0x20__seed42__pw0.5__piglasso_results.pkl"
)
INFERRED_PKL = Path(
    "/gpfs/home2/zblei/Documents/BurnInjuries/PIGLasso/pipeline_src/inference/"
    "results/network_inference/GSE182616/"
    "PHASE__Acute__n513__zscored__filtered__Q200__bperc0.65__lam0.05-1.0x20__seed42__pw0.5__inferred.pkl"
)
OUT = PKL.parent.parent.parent / "network_inference" / "GSE182616" / "stability_scores.csv"

print(f"Loading raw PIGLasso pkl: {PKL.name}")
with open(PKL, "rb") as f:
    pig = pickle.load(f)

print("Keys:", list(pig.keys()))

edge_counts_all = pig["edge_counts_all"]   # (p, p, n_lambda)
Q              = int(pig["Q"])
genes          = list(map(str, pig["genes"]))
lambda_range   = np.array(pig["lambda_range"], dtype=float)
p, _, n_lam    = edge_counts_all.shape
print(f"  p={p}, Q={Q}, n_lambda={n_lam}, lambda range [{lambda_range[0]:.3f}, {lambda_range[-1]:.3f}]")

# Load inferred pkl to get the knee/slice info
print(f"Loading inferred pkl for knee info: {INFERRED_PKL.name}")
with open(INFERRED_PKL, "rb") as f:
    inf = pickle.load(f)

knee      = inf["knee"]
l_lo      = knee["slice_lo"]
l_hi      = knee["slice_hi"]
main_lam  = knee["main_lambda"]
main_idx  = knee["main_idx"]
print(f"  Knee slice: idx [{l_lo},{l_hi}), lambda [{lambda_range[l_lo]:.3f}, {lambda_range[l_hi-1]:.3f}]")
print(f"  Main knee lambda: {main_lam} (idx {main_idx})")

# Stability scores at the main knee lambda
counts_at_knee = edge_counts_all[:, :, main_idx].astype(float)
stability      = counts_at_knee / Q   # fraction in [0, 1]

# Load adjacency for selected-edge flag
adj = inf["adjacency"]   # binary, shape (p, p)

# Extract upper triangle
rows = []
for i in range(p):
    for j in range(i + 1, p):
        rows.append({
            "gene_i":    genes[i],
            "gene_j":    genes[j],
            "stability": stability[i, j],
            "selected":  int(adj[i, j]),
        })

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"\nSaved {len(df)} edges → {OUT}")
print(f"  Selected edges: {df['selected'].sum()}")
print(f"  Stability range: {df['stability'].min():.3f} – {df['stability'].max():.3f}")
print(f"  Median stability (selected): {df.loc[df['selected']==1,'stability'].median():.3f}")
print(f"  Median stability (non-selected): {df.loc[df['selected']==0,'stability'].median():.3f}")
