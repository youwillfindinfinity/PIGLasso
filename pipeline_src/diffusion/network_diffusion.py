#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import scipy.linalg


def load_network_from_adjacency(adj_path: Path) -> tuple[nx.Graph, pd.DataFrame]:
    sep = "," if adj_path.suffix == ".csv" else "\t"
    adj = pd.read_csv(adj_path, sep=sep, index_col=0)
    adj.index = adj.index.astype(str)
    adj.columns = adj.columns.astype(str)

    # ensure symmetry
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj.values, 0.0)

    G = nx.from_pandas_adjacency(adj)
    return G, adj


def largest_component_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return G
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()


def laplacian_from_adj(adj: pd.DataFrame, normalized: bool = False) -> np.ndarray:
    A = adj.values.astype(float)
    deg = A.sum(axis=1)
    if not normalized:
        D = np.diag(deg)
        return D - A

    with np.errstate(divide="ignore"):
        inv_sqrt = 1.0 / np.sqrt(deg)
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    Dm = np.diag(inv_sqrt)
    I = np.eye(A.shape[0])
    return I - (Dm @ A @ Dm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="../trauma_data/diffusion_inputs", help="diffusion_inputs dir")
    ap.add_argument("--adj", default="burn_network_adjacency.csv", help="adjacency file name in in_dir")
    ap.add_argument("--delta", default="delta.tsv", help="delta vector file name in in_dir")
    ap.add_argument("--common_genes", default="common_genes.txt", help="optional gene list to validate against")
    ap.add_argument("--strict_gene_match", action="store_true",
                    help="error if delta genes != adjacency genes (recommended for common-genes pipeline)")
    ap.add_argument("--out_dir", default=None, help="output dir (default: trauma_data/baseline_diffusion)")
    ap.add_argument("--tmin", type=float, default=1e-4)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--nt", type=int, default=80)
    ap.add_argument("--normalized_laplacian", action="store_true")
    ap.add_argument("--use_lcc", action="store_true", help="restrict to largest connected component")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    adj_path = in_dir / args.adj
    delta_path = in_dir / args.delta

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = in_dir.parent / "baseline_diffusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load network ---
    G, adj = load_network_from_adjacency(adj_path)

    degrees = np.array([d for _, d in G.degree()])
    n_iso = int((degrees == 0).sum())
    print(f"[NET] nodes={G.number_of_nodes()} edges={G.number_of_edges()} isolates={n_iso}")

    if args.use_lcc:
        G_lcc = largest_component_subgraph(G)
        keep = list(G_lcc.nodes())
        adj = adj.loc[keep, keep]
        G = G_lcc
        print(f"[NET] using LCC: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    node_order = list(adj.index)

    # --- optional: validate common_genes.txt matches adjacency ---
    common_path = in_dir / args.common_genes
    if common_path.exists():
        common_genes = [l.strip() for l in common_path.read_text().splitlines() if l.strip()]
        if set(common_genes) != set(node_order):
            print(f"[WARN] common_genes.txt set != adjacency set "
                  f"(common={len(common_genes)} adj={len(node_order)})")
        # stricter: ensure same order
        if common_genes == node_order:
            print("[INFO] common_genes.txt matches adjacency order ✓")
        else:
            print("[INFO] common_genes.txt does not match adjacency order (that can be OK).")

    # --- load delta and align ---
    delta_df = pd.read_csv(delta_path, sep="\t", index_col=0)
    delta_df.index = delta_df.index.astype(str)

    if delta_df.shape[1] == 1:
        delta_vec = delta_df.iloc[:, 0]
    else:
        if "delta" in delta_df.columns:
            delta_vec = delta_df["delta"]
        else:
            raise ValueError(f"delta.tsv has multiple columns: {list(delta_df.columns)}")

    # Strict consistency checks
    delta_genes = set(delta_vec.index)
    adj_genes = set(node_order)

    if args.strict_gene_match:
        missing_in_delta = [g for g in node_order if g not in delta_genes]
        extra_in_delta = [g for g in delta_vec.index if g not in adj_genes]
        if missing_in_delta or extra_in_delta:
            raise RuntimeError(
                f"Gene mismatch between adjacency and delta.\n"
                f"  missing_in_delta: {len(missing_in_delta)} (example {missing_in_delta[:5]})\n"
                f"  extra_in_delta:   {len(extra_in_delta)} (example {extra_in_delta[:5]})\n"
                f"Tip: regenerate diffusion_inputs with the common-genes script, or disable --strict_gene_match."
            )

    # Align (should be exact in strict mode; otherwise fills 0)
    delta_vec = delta_vec.reindex(node_order).fillna(0.0).astype(float)

    print(f"[DELTA] nonzero genes: {(delta_vec != 0).sum()} | std={delta_vec.std():.4g}")

    # --- diffusion ---
    L = laplacian_from_adj(adj, normalized=args.normalized_laplacian)

    t_values = np.linspace(args.tmin, args.tmax, args.nt)
    S = np.zeros((len(node_order), len(t_values)), dtype=float)

    for j, t in enumerate(t_values):
        K = scipy.linalg.expm(-t * L)
        S[:, j] = K @ delta_vec.values

    S_df = pd.DataFrame(S, index=node_order, columns=[f"t={t:.4g}" for t in t_values])
    S_df.to_csv(out_dir / "diffused_signal_genes_x_t.tsv", sep="\t")

    # summaries
    abs_delta = delta_vec.abs().sort_values(ascending=False).head(30)
    abs_delta.to_csv(out_dir / "top30_abs_delta.tsv", sep="\t")

    for t_pick in [t_values[0], t_values[len(t_values)//2], t_values[-1]]:
        col = f"t={t_pick:.4g}"
        top = S_df[col].abs().sort_values(ascending=False).head(30)
        top.to_csv(out_dir / f"top30_abs_diffused_{col.replace('=', '').replace('.', 'p')}.tsv", sep="\t")

    print(f"[SAVED] {out_dir / 'diffused_signal_genes_x_t.tsv'}")
    print(f"[SAVED] {out_dir / 'top30_abs_delta.tsv'}")
    print("[DONE]")


if __name__ == "__main__":
    main()