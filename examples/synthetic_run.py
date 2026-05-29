"""
PIGLasso synthetic example
==========================

Demonstrates the full PIGLasso workflow on a small synthetic dataset
(no real data download required).

Requirements:
    pip install nodis
    pip install -e ..   # from repo root

Run:
    python examples/synthetic_run.py
"""

from __future__ import annotations

import numpy as np
from nodis.estimators.piglasso import PIGLassoEstimator
from nodis.estimators.prior_utils import build_corr_prior

# ------------------------------------------------------------------
# 1. Simulate expression data (50 samples × 30 genes)
# ------------------------------------------------------------------
rng = np.random.default_rng(42)
p = 30
n = 50

# True precision matrix: block-diagonal (two modules of 15 genes each)
Omega = np.eye(p) * 2.0
for i in range(14):
    Omega[i, i + 1] = Omega[i + 1, i] = 0.6
for i in range(15, 29):
    Omega[i, i + 1] = Omega[i + 1, i] = 0.6

Sigma = np.linalg.inv(Omega)
X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)

print(f"Simulated expression matrix: {X.shape[0]} samples × {X.shape[1]} genes")

# ------------------------------------------------------------------
# 2. Build a data-derived correlation prior (no ground-truth leak)
# ------------------------------------------------------------------
prior = build_corr_prior(X, gamma=2.0)
print(f"Prior matrix: shape={prior.shape}, density={np.mean(prior > 0.1):.3f}")

# ------------------------------------------------------------------
# 3. Fit PIGLasso with prior
# ------------------------------------------------------------------
est = PIGLassoEstimator(
    Q=20,           # subsamples (use 200+ for real data)
    b_perc=0.65,
    n_lambda=10,    # lambda grid points (use 20+ for real data)
    lambda_lo=0.05,
    lambda_hi=0.40,
    pi_thr=0.5,
    prior_weight=0.5,
    n_jobs=1,
    seed=42,
)
est.fit(X, prior=prior)

adj = est.get_adjacency()
scores = est.precision_

n_edges = int(adj.sum()) // 2
print(f"Edges selected (pi >= 0.5): {n_edges}")
print(f"Stability score matrix: min={scores.min():.3f}, max={scores.max():.3f}")

# ------------------------------------------------------------------
# 4. Compare with no-prior baseline
# ------------------------------------------------------------------
est_noprior = PIGLassoEstimator(Q=20, b_perc=0.65, n_lambda=10,
                                lambda_lo=0.05, lambda_hi=0.40,
                                pi_thr=0.5, n_jobs=1, seed=42)
est_noprior.fit(X)
adj_noprior = est_noprior.get_adjacency()
n_edges_noprior = int(adj_noprior.sum()) // 2

print(f"\nEdges without prior : {n_edges_noprior}")
print(f"Edges with prior    : {n_edges}")
print(f"Prior shifted edges : {int((adj - adj_noprior).clip(0).sum()) // 2} newly selected")

# ------------------------------------------------------------------
# 5. (Optional) pass to NODIS for statistical inference
# ------------------------------------------------------------------
try:
    from nodis.estimators.desparsified import DesparsifiedGGM
    print("\nNODIS DesparsifiedGGM available — you can run inference on top of this network.")
except ImportError:
    print("\nNODIS inference layer (DesparsifiedGGM) not yet implemented — coming in NODIS v0.1.")
