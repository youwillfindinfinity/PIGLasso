#!/usr/bin/env python3
"""
Prior construction pipeline for PIGLasso (Burns Transcriptomics).
Implements Steps 1–4 from priorPIGLASSO.md as individually callable steps.

Usage:
    python build_prior.py --step 1
    python build_prior.py --step 2a
    python build_prior.py --step 2b
    python build_prior.py --step 2c
    python build_prior.py --step 3
    python build_prior.py --step 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- Paths ----------
PIPELINE_DIR = Path(__file__).parent
EXPR_PATH = PIPELINE_DIR / (
    "preprocessing/"
    "burn_control/preprocessed/GSE37069_controls__genes_x_samples.tsv"
)
OUT_DIR = PIPELINE_DIR / "prior"
OUT_DIR.mkdir(exist_ok=True)

GENES_TXT     = OUT_DIR / "genes.txt"
PRIOR_STRING  = OUT_DIR / "prior_string.npy"
PRIOR_PATHWAY = OUT_DIR / "prior_pathway.npy"
PRIOR_COEXPR  = OUT_DIR / "prior_coexpr.npy"
PRIOR_FINAL   = OUT_DIR / "prior_piglasso.npy"
PRIOR_PLOT    = OUT_DIR / "prior_distribution.png"


# ============================================================
# Step 1 — Extract gene list
# ============================================================
def step1():
    print("\n=== Step 1: Extract gene list ===", flush=True)
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    genes = list(expr.index.astype(str))
    print(f"  Genes: {len(genes)}  |  Samples: {expr.shape[1]}", flush=True)
    print(f"  Sample genes: {genes[:5]}", flush=True)
    GENES_TXT.write_text("\n".join(genes) + "\n")
    print(f"  Saved: {GENES_TXT}", flush=True)


# ============================================================
# Step 2a — STRING PPI prior
# ============================================================
def _load_genes() -> list[str]:
    if not GENES_TXT.exists():
        sys.exit(f"ERROR: {GENES_TXT} not found — run step 1 first.")
    return GENES_TXT.read_text().strip().split("\n")


def _query_string_batch(
    batch: list[str],
    gene_idx: dict[str, int],
    P: np.ndarray,
    species: int = 9606,
    score_threshold: int = 400,
) -> int:
    string_api = "https://version-12-0.string-db.org/api/tsv-no-header/network"
    params = {
        "identifiers":     "%0d".join(batch),
        "species":         species,
        "required_score":  score_threshold,
        "caller_identity": "piglasso_burns_prior",
    }
    for attempt in range(3):
        try:
            resp = requests.post(string_api, data=params, timeout=120)
            resp.raise_for_status()
            break
        except Exception as e:
            print(f"    [WARN] attempt {attempt+1}/3: {e}", flush=True)
            time.sleep(5)
    else:
        print("    [ERROR] STRING batch failed after 3 attempts — skipping.", flush=True)
        return 0

    lines = [l.strip().split("\t") for l in resp.text.strip().split("\n") if l.strip()]
    edges = 0
    for l in lines:
        if len(l) < 12:
            continue
        g1, g2 = l[2], l[3]
        try:
            exp_score = float(l[10])
        except (ValueError, IndexError):
            continue
        if g1 in gene_idx and g2 in gene_idx:
            i, j = gene_idx[g1], gene_idx[g2]
            w = exp_score  # API v12 returns scores in [0, 1] directly
            if w > P[i, j]:
                P[i, j] = w
                P[j, i] = w
            edges += 1
    return edges


def step2a(batch_size: int = 1000):
    print("\n=== Step 2a: STRING PPI prior ===", flush=True)
    if PRIOR_STRING.exists():
        print(f"  FOUND {PRIOR_STRING} — already done, exiting.", flush=True)
        return

    genes = _load_genes()
    p = len(genes)
    gene_idx = {g: i for i, g in enumerate(genes)}
    P = np.zeros((p, p), dtype=np.float32)

    batches = [genes[i:i + batch_size] for i in range(0, p, batch_size)]
    print(f"  {len(batches)} batches of ≤{batch_size} genes...", flush=True)
    for bi, batch in enumerate(batches, 1):
        print(f"  Batch {bi}/{len(batches)} ({len(batch)} genes)...", end=" ", flush=True)
        edges = _query_string_batch(batch, gene_idx, P)
        print(f"{edges} edges", flush=True)
        time.sleep(1)

    total = int(P.astype(bool).sum()) // 2
    print(f"  Total STRING edges: {total}", flush=True)
    np.save(PRIOR_STRING, P)
    print(f"  Saved: {PRIOR_STRING}", flush=True)


# ============================================================
# Step 2b — KEGG pathway co-membership prior
# ============================================================
BURN_PATHWAYS = {
    "hsa04610": "Complement and coagulation cascades",
    "hsa04620": "Toll-like receptor signaling",
    "hsa04630": "JAK-STAT signaling",
    "hsa04060": "Cytokine-cytokine receptor interaction",
    "hsa04210": "Apoptosis",
    "hsa04064": "NF-kappa B signaling",
    "hsa04668": "TNF signaling",
    "hsa04066": "HIF-1 signaling",
    "hsa04151": "PI3K-Akt signaling",
    "hsa04380": "Osteoclast differentiation",
    "hsa04510": "Focal adhesion",
    "hsa04514": "Cell adhesion molecules",
}


def _fetch_kegg_genes(pathway_id: str) -> set[str]:
    url = f"https://rest.kegg.jp/get/{pathway_id}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    genes: set[str] = set()
    in_gene = False
    for line in resp.text.split("\n"):
        if line.startswith("GENE"):
            in_gene = True
        elif line.startswith(("COMPOUND", "REFERENCE", "DISEASE", "///", "REACTION")):
            in_gene = False
        if in_gene:
            parts = line.strip().split()
            if len(parts) >= 2:
                sym = parts[1].rstrip(";").split(",")[0]
                genes.add(sym)
    return genes


def step2b():
    print("\n=== Step 2b: KEGG pathway prior ===", flush=True)
    if PRIOR_PATHWAY.exists():
        print(f"  FOUND {PRIOR_PATHWAY} — already done, exiting.", flush=True)
        return

    genes = _load_genes()
    p = len(genes)
    gene_idx = {g: i for i, g in enumerate(genes)}
    P = np.zeros((p, p), dtype=np.float32)

    for pid, pname in BURN_PATHWAYS.items():
        try:
            pgenes = _fetch_kegg_genes(pid)
        except Exception as e:
            print(f"  [WARN] {pid} ({pname}): {e}", flush=True)
            continue
        members = [gene_idx[g] for g in pgenes if g in gene_idx]
        print(f"  {pname}: {len(pgenes)} pathway genes, {len(members)} in list", flush=True)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                P[members[a], members[b]] += 1
                P[members[b], members[a]] += 1
        time.sleep(0.3)

    P /= len(BURN_PATHWAYS)
    density = (P > 0).sum() / 2 / (p * (p - 1) / 2)
    print(f"  Pathway prior density: {density:.6f}", flush=True)
    np.save(PRIOR_PATHWAY, P)
    print(f"  Saved: {PRIOR_PATHWAY}", flush=True)


# ============================================================
# Step 2c — Spearman correlation prior
# ============================================================
def step2c(threshold: float = 0.3):
    print("\n=== Step 2c: Spearman correlation prior ===", flush=True)
    if PRIOR_COEXPR.exists():
        print(f"  FOUND {PRIOR_COEXPR} — already done, exiting.", flush=True)
        return

    print(f"  Loading expression matrix...", flush=True)
    expr = pd.read_csv(EXPR_PATH, sep="\t", index_col=0)
    print(f"  Computing Spearman correlation ({expr.shape[0]} genes × {expr.shape[1]} samples)...", flush=True)
    corr = expr.T.corr(method="spearman").abs().values.astype(np.float32)
    np.fill_diagonal(corr, 0)
    corr[corr < threshold] = 0
    if corr.max() > 0:
        corr /= corr.max()

    p = corr.shape[0]
    density = (corr > 0).sum() / 2 / (p * (p - 1) / 2)
    print(f"  Correlation prior density (threshold={threshold}): {density:.4f}", flush=True)
    np.save(PRIOR_COEXPR, corr)
    print(f"  Saved: {PRIOR_COEXPR}", flush=True)


# ============================================================
# Step 3 — Combine
# ============================================================
def step3(w_string: float = 0.5, w_pathway: float = 0.3, w_coexpr: float = 0.2):
    print("\n=== Step 3: Combine priors ===", flush=True)
    for path, name in [(PRIOR_STRING, "prior_string"), (PRIOR_PATHWAY, "prior_pathway"), (PRIOR_COEXPR, "prior_coexpr")]:
        if not path.exists():
            sys.exit(f"ERROR: {path} not found — run step 2a/2b/2c first.")

    P_string  = np.load(PRIOR_STRING).astype(np.float32)
    P_pathway = np.load(PRIOR_PATHWAY).astype(np.float32)
    P_coexpr  = np.load(PRIOR_COEXPR).astype(np.float32)

    P = w_string * P_string + w_pathway * P_pathway + w_coexpr * P_coexpr
    P = (P + P.T) / 2
    np.fill_diagonal(P, 0)
    P = np.clip(P, 0, 1)

    p = P.shape[0]
    density = (P > 0.05).sum() / 2 / (p * (p - 1) / 2)
    print(f"  Combined prior: shape={P.shape}, density@0.05={density:.4f}", flush=True)
    if density > 0.15:
        print("  [WARN] Density > 15%", flush=True)
    if density < 0.01:
        print("  [WARN] Density < 1%", flush=True)

    np.save(PRIOR_FINAL, P)
    print(f"  Saved: {PRIOR_FINAL}", flush=True)


# ============================================================
# Step 4 — Validate
# ============================================================
def step4():
    print("\n=== Step 4: Validation ===", flush=True)
    if not PRIOR_FINAL.exists():
        sys.exit(f"ERROR: {PRIOR_FINAL} not found — run step 3 first.")

    P = np.load(PRIOR_FINAL)
    assert P.shape[0] == P.shape[1], "Prior must be square"
    assert np.allclose(P, P.T, atol=1e-5), "Prior must be symmetric"
    assert np.all(np.diag(P) == 0), "Diagonal must be zero"
    assert P.min() >= 0 and P.max() <= 1, "Prior values must be in [0, 1]"
    print("  All structural checks passed", flush=True)

    p = P.shape[0]
    density = (P > 0.05).sum() / 2 / (p * (p - 1) / 2)
    print(f"  Shape:   {P.shape}", flush=True)
    print(f"  Min:     {P.min():.4f}", flush=True)
    print(f"  Max:     {P.max():.4f}", flush=True)
    print(f"  Mean:    {P.mean():.6f}", flush=True)
    print(f"  Density: {density:.4f}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nz = P[P > 0].flatten()
        plt.figure(figsize=(8, 4))
        plt.hist(nz, bins=50, color="steelblue", edgecolor="white")
        plt.xlabel("Prior weight")
        plt.ylabel("Edge count")
        plt.title("Distribution of non-zero prior weights")
        plt.tight_layout()
        plt.savefig(PRIOR_PLOT, dpi=150)
        plt.close()
        print(f"  Saved plot: {PRIOR_PLOT}", flush=True)
    except ImportError:
        print("  [WARN] matplotlib not available — skipping plot", flush=True)

    print(f"\nFinal prior: {PRIOR_FINAL}", flush=True)


# ============================================================
# Main
# ============================================================
STEPS = {
    "1":  step1,
    "2a": step2a,
    "2b": step2b,
    "2c": step2c,
    "3":  step3,
    "4":  step4,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=list(STEPS), help="Which step to run")
    args = parser.parse_args()

    t0 = time.time()
    STEPS[args.step]()
    print(f"\nStep {args.step} done in {(time.time() - t0) / 60:.1f} min", flush=True)
