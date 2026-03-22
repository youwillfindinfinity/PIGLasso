"""
Hub gene identification from PIGLasso burn network + prior circularity diagnostic.

Inputs
------
- PIGLasso adjacency:  results/burns/expression_top5000_adjacency.csv
- PIGLasso stability:  results/burns/expression_top5000_stability.csv
- Prior matrix:        data/burn/prior_burns_top5000.npy
- Gene list:           data/burn/genes_top5000.txt
- NODIS adjacency:     (optional) NODIS/results/burns/burns_nodis_adj_fdr05.csv

Outputs (written to --out)
--------------------------
  hub_rankings.csv          — genes ranked by degree, betweenness, eigenvector centrality
  hub_convergent.csv        — genes in top-k% on >=2 metrics (convergent evidence hubs)
  prior_circularity.txt     — Spearman rho between prior degree and inferred degree
  nodis_hub_overlap.csv     — (if --nodis-adj provided) hub edges validated by NODIS FDR

Usage
-----
    python scripts/hub_analysis_burns.py \\
        --adj     results/burns/expression_top5000_adjacency.csv \\
        --stab    results/burns/expression_top5000_stability.csv \\
        --prior   data/burn/prior_burns_top5000.npy \\
        --genes   data/burn/genes_top5000.txt \\
        --out     results/burns/hubs/

    # With NODIS validation layer:
    python scripts/hub_analysis_burns.py \\
        --adj       results/burns/expression_top5000_adjacency.csv \\
        --stab      results/burns/expression_top5000_stability.csv \\
        --prior     data/burn/prior_burns_top5000.npy \\
        --genes     data/burn/genes_top5000.txt \\
        --nodis-adj results/burns/burns_nodis_adj_fdr05.csv \\
        --out       results/burns/hubs/
"""
from __future__ import annotations

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def degree_centrality(adj: np.ndarray) -> np.ndarray:
    return adj.sum(axis=1)


def betweenness_centrality(adj: np.ndarray) -> np.ndarray:
    try:
        import networkx as nx
        G = nx.from_numpy_array(adj)
        bc = nx.betweenness_centrality(G, normalized=True)
        return np.array([bc[i] for i in range(len(adj))])
    except ImportError:
        warnings.warn("networkx not available; betweenness set to NaN.")
        return np.full(len(adj), np.nan)


def eigenvector_centrality(adj: np.ndarray) -> np.ndarray:
    try:
        import networkx as nx
        G = nx.from_numpy_array(adj)
        try:
            ec = nx.eigenvector_centrality_numpy(G)
            return np.array([ec[i] for i in range(len(adj))])
        except Exception:
            return np.full(len(adj), np.nan)
    except ImportError:
        return np.full(len(adj), np.nan)


# ---------------------------------------------------------------------------
# Prior circularity diagnostic
# ---------------------------------------------------------------------------

def prior_circularity_check(
    prior: np.ndarray,
    adj: np.ndarray,
    genes: list[str],
    out_path: pathlib.Path,
) -> float:
    prior_deg = prior.sum(axis=1)
    inferred_deg = adj.sum(axis=1)

    # Only meaningful for genes that have at least one edge in inferred network
    mask = inferred_deg > 0
    if mask.sum() < 10:
        warnings.warn("Fewer than 10 genes with inferred edges; prior diagnostic unreliable.")

    rho, pval = spearmanr(prior_deg[mask], inferred_deg[mask])

    lines = [
        "Prior Circularity Diagnostic",
        "=" * 40,
        f"Genes with >=1 inferred edge:  {mask.sum()}",
        f"Spearman rho (prior deg vs inferred deg): {rho:.4f}  (p={pval:.4g})",
        "",
        "Interpretation:",
        "  rho > 0.7 → CIRCULAR: hubs likely prior-driven, not data-driven",
        "  rho < 0.5 → OK: prior acts as noise filter, hubs are data-driven",
        "  0.5-0.7   → AMBIGUOUS: report sensitivity analysis across prior weights",
        "",
    ]

    if rho > 0.7:
        lines.append("STATUS: CIRCULAR — treat hub results with caution.")
        lines.append("ACTION: Run sensitivity analysis with prior_weight in {0.1, 0.5, 1.0}.")
    elif rho < 0.5:
        lines.append("STATUS: OK — hubs are data-driven.")
    else:
        lines.append("STATUS: AMBIGUOUS — run prior weight sensitivity analysis.")

    # Top-10 genes by inferred degree with their prior degree for inspection
    df_check = pd.DataFrame({
        "gene": genes,
        "prior_degree": prior_deg,
        "inferred_degree": inferred_deg,
    }).sort_values("inferred_degree", ascending=False).head(20)
    lines.append("\nTop-20 hub genes — prior degree vs inferred degree:")
    lines.append(df_check.to_string(index=False))

    text = "\n".join(lines)
    out_path.write_text(text)
    print(text)
    return float(rho)


# ---------------------------------------------------------------------------
# Convergent evidence hubs
# ---------------------------------------------------------------------------

def convergent_hubs(rankings: pd.DataFrame, top_k: float = 0.10) -> pd.DataFrame:
    """
    Genes in the top k% for at least 2 of: degree, betweenness, eigenvector.
    This reduces single-metric false positives.
    """
    metrics = ["degree", "betweenness", "eigenvector"]
    available = [m for m in metrics if m in rankings.columns and rankings[m].notna().any()]
    p = len(rankings)
    cutoff = max(1, int(np.ceil(top_k * p)))

    in_top = pd.DataFrame(index=rankings.index)
    for m in available:
        ranked = rankings[m].rank(ascending=False, method="min")
        in_top[m] = ranked <= cutoff

    rankings["n_metrics_top"] = in_top[available].sum(axis=1)
    convergent = rankings[rankings["n_metrics_top"] >= 2].copy()
    convergent = convergent.sort_values("degree", ascending=False)
    return convergent


# ---------------------------------------------------------------------------
# NODIS overlap
# ---------------------------------------------------------------------------

def nodis_hub_overlap(
    hub_genes: list[str],
    nodis_adj: pd.DataFrame,
    piglasso_adj: np.ndarray,
    genes: list[str],
) -> pd.DataFrame:
    """
    For each hub gene, count how many of its PIGLasso-selected edges
    are also FDR-validated by NODIS.
    """
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    rows = []
    nodis_genes = set(nodis_adj.index)

    for hub in hub_genes:
        if hub not in gene_to_idx:
            continue
        i = gene_to_idx[hub]
        piglasso_partners = [genes[j] for j in range(len(genes))
                             if piglasso_adj[i, j] == 1 and i != j]
        if hub not in nodis_genes:
            rows.append({"hub_gene": hub,
                         "piglasso_edges": len(piglasso_partners),
                         "nodis_validated": 0,
                         "validation_rate": np.nan,
                         "note": "hub not in NODIS gene set"})
            continue

        validated = sum(
            1 for p in piglasso_partners
            if p in nodis_genes and nodis_adj.loc[hub, p] == 1
        )
        in_nodis = sum(1 for p in piglasso_partners if p in nodis_genes)
        rows.append({
            "hub_gene": hub,
            "piglasso_edges": len(piglasso_partners),
            "nodis_validated": validated,
            "nodis_eligible": in_nodis,
            "validation_rate": validated / in_nodis if in_nodis > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hub gene identification from PIGLasso burn network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adj",    required=True, type=pathlib.Path,
                        help="PIGLasso adjacency CSV (genes × genes).")
    parser.add_argument("--stab",   required=True, type=pathlib.Path,
                        help="PIGLasso stability scores CSV (genes × genes).")
    parser.add_argument("--prior",  required=True, type=pathlib.Path,
                        help="Prior matrix .npy (genes × genes, same order as --genes).")
    parser.add_argument("--genes",  required=True, type=pathlib.Path,
                        help="Gene list .txt (one symbol per line, matches prior rows).")
    parser.add_argument("--nodis-adj", type=pathlib.Path, default=None,
                        help="(Optional) NODIS FDR-controlled adjacency CSV for validation.")
    parser.add_argument("--top-k", type=float, default=0.10,
                        help="Top-k fraction for convergent evidence criterion.")
    parser.add_argument("--out",    required=True, type=pathlib.Path,
                        help="Output directory.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("Loading PIGLasso adjacency ...")
    adj_df = pd.read_csv(args.adj, index_col=0)
    genes  = list(adj_df.index)
    adj    = adj_df.values.astype(float)

    print("Loading prior matrix ...")
    prior_genes = [l.strip() for l in args.genes.read_text().splitlines() if l.strip()]
    prior_full  = np.load(args.prior)

    # Align prior to adjacency gene order
    prior_idx = {g: i for i, g in enumerate(prior_genes)}
    align = [prior_idx[g] for g in genes if g in prior_idx]
    missing_in_prior = [g for g in genes if g not in prior_idx]
    if missing_in_prior:
        warnings.warn(f"{len(missing_in_prior)} adjacency genes not in prior gene list.")
    genes_aligned = [g for g in genes if g in prior_idx]
    adj_aligned   = adj[[genes.index(g) for g in genes_aligned], :][:, [genes.index(g) for g in genes_aligned]]
    prior_aligned = prior_full[np.ix_(align, align)]

    p = len(genes_aligned)
    n_edges = int(adj_aligned.sum()) // 2
    print(f"Network: {p} genes, {n_edges} edges")

    # 2. Centrality
    print("Computing centrality metrics ...")
    deg = degree_centrality(adj_aligned)
    bet = betweenness_centrality(adj_aligned)
    eig = eigenvector_centrality(adj_aligned)

    stab_df = pd.read_csv(args.stab, index_col=0)
    mean_stab = np.array([stab_df.loc[g].mean() if g in stab_df.index else np.nan
                          for g in genes_aligned])

    rankings = pd.DataFrame({
        "gene":         genes_aligned,
        "degree":       deg,
        "betweenness":  bet,
        "eigenvector":  eig,
        "mean_stability": mean_stab,
    }).set_index("gene").sort_values("degree", ascending=False)

    rankings.to_csv(args.out / "hub_rankings.csv")
    print(f"Hub rankings written: {args.out / 'hub_rankings.csv'}")
    print(rankings.head(20).to_string())

    # 3. Convergent evidence hubs
    convergent = convergent_hubs(rankings.copy(), top_k=args.top_k)
    convergent.to_csv(args.out / "hub_convergent.csv")
    print(f"\nConvergent evidence hubs (top {int(args.top_k*100)}%, ≥2 metrics): "
          f"{len(convergent)} genes")
    print(convergent.head(20).to_string())

    # 4. Prior circularity diagnostic
    print("\nRunning prior circularity diagnostic ...")
    prior_circularity_check(
        prior_aligned, adj_aligned, genes_aligned,
        args.out / "prior_circularity.txt",
    )

    # 5. NODIS validation (optional)
    if args.nodis_adj is not None:
        print(f"\nLoading NODIS adjacency: {args.nodis_adj}")
        nodis_adj = pd.read_csv(args.nodis_adj, index_col=0)
        hub_genes = list(convergent.index[:50])  # top-50 convergent hubs
        overlap_df = nodis_hub_overlap(hub_genes, nodis_adj, adj_aligned, genes_aligned)
        overlap_df.to_csv(args.out / "nodis_hub_overlap.csv", index=False)
        validated = overlap_df["nodis_validated"].sum()
        total_edges = overlap_df["piglasso_edges"].sum()
        print(f"NODIS validation: {validated}/{total_edges} hub edges FDR-validated "
              f"({100*validated/max(total_edges,1):.1f}%)")

    print(f"\nAll outputs in {args.out}")


if __name__ == "__main__":
    main()
