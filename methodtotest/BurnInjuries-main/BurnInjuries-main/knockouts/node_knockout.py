#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, diags, issparse
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm

# Optional MPI
try:
    from mpi4py import MPI
    _HAVE_MPI = True
except Exception:
    _HAVE_MPI = False


def read_delta(delta_path: Path) -> pd.Series:
    """
    delta.tsv: either
      - two-column TSV with index=gene, value=delta
      - or a 1-column TSV with gene index
    Returns: pd.Series indexed by gene.
    """
    df = pd.read_csv(delta_path, sep="\t", index_col=0)
    if df.shape[1] == 0:
        raise ValueError(f"{delta_path} has no columns?")
    if df.shape[1] == 1:
        s = df.iloc[:, 0]
    else:
        # If multiple columns exist, try to find 'delta'
        if "delta" in df.columns:
            s = df["delta"]
        else:
            s = df.iloc[:, 0]
    s.index = s.index.astype(str)
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return s


def load_network_as_sparse(
    path: Path,
    genes: list[str],
    sep: str = "\t",
) -> csr_matrix:
    """
    Supports:
      1) edge list TSV/CSV with columns: gene1, gene2, weight
      2) adjacency matrix CSV/TSV with gene names as both index and columns
    Returns: symmetric weighted adjacency W (CSR) in the provided gene order.
    """
    suffix = path.suffix.lower()

    # Heuristic: if it's "edgelist" in name or has 3 columns -> edge list
    # Otherwise treat as adjacency matrix.
    # You can force edge-list by passing a 3-col file.
    df_head = pd.read_csv(path, sep=sep, nrows=5)
    is_edge_list = df_head.shape[1] in (3, 4)

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    if is_edge_list:
        df = pd.read_csv(path, sep=sep)
        # Accept common column names
        cols = [c.lower() for c in df.columns]
        df.columns = cols

        # Try to identify columns
        if {"gene1", "gene2", "weight"}.issubset(set(cols)):
            c1, c2, cw = "gene1", "gene2", "weight"
        elif {"u", "v", "weight"}.issubset(set(cols)):
            c1, c2, cw = "u", "v", "weight"
        elif df.shape[1] >= 3:
            c1, c2, cw = df.columns[0], df.columns[1], df.columns[2]
        else:
            raise ValueError("Edge list must have at least 3 columns (gene1, gene2, weight).")

        u = df[c1].astype(str).to_numpy()
        v = df[c2].astype(str).to_numpy()
        w = pd.to_numeric(df[cw], errors="coerce").fillna(0.0).to_numpy()

        # Keep only edges within our gene set
        mask = np.array([(a in gene_to_idx) and (b in gene_to_idx) for a, b in zip(u, v)])
        u = u[mask]; v = v[mask]; w = w[mask]

        rows = np.array([gene_to_idx[a] for a in u], dtype=np.int32)
        cols = np.array([gene_to_idx[b] for b in v], dtype=np.int32)
        data = w.astype(np.float64)

        # Make symmetric (add both directions)
        rows_sym = np.concatenate([rows, cols])
        cols_sym = np.concatenate([cols, rows])
        data_sym = np.concatenate([data, data])

        W = coo_matrix((data_sym, (rows_sym, cols_sym)), shape=(n, n)).tocsr()
        W.sum_duplicates()
        return W

    # adjacency matrix
    adj = pd.read_csv(path, sep=sep, index_col=0)
    adj.index = adj.index.astype(str)
    adj.columns = adj.columns.astype(str)

    # Reindex to our gene order, fill missing with 0
    adj = adj.reindex(index=genes, columns=genes).fillna(0.0)
    W = csr_matrix(adj.to_numpy(dtype=np.float64))
    return W


def laplacian_from_adjacency(W: csr_matrix) -> csr_matrix:
    # L = D - W, where D is diagonal of strengths
    d = np.asarray(W.sum(axis=1)).ravel()
    L = diags(d, 0, format="csr") - W
    return L


def build_edge_index_lists(W: csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    """
    Convert W to COO arrays (rows, cols, data). Also build, for each node i,
    the list of data indices k where edge k is incident to i (i is row or col).
    This lets us knockdown node i by scaling data[idx_list[i]].
    """
    W_coo = W.tocoo()
    rows = W_coo.row.astype(np.int32)
    cols = W_coo.col.astype(np.int32)
    data = W_coo.data.astype(np.float64)

    n = W.shape[0]
    idx_lists: list[list[int]] = [[] for _ in range(n)]
    for k in range(data.shape[0]):
        i = int(rows[k]); j = int(cols[k])
        idx_lists[i].append(k)
        if j != i:
            idx_lists[j].append(k)
    return rows, cols, data, idx_lists


def diffuse_signal(L: csr_matrix, delta_vec: np.ndarray, t_max: float, t_num: int) -> np.ndarray:
    """
    Computes S(t) = exp(-t L) delta for t in linspace(0, t_max, t_num).
    Returns: array shape (t_num, n).
    """
    tvals = np.linspace(0.0, float(t_max), int(t_num))
    # expm_multiply supports a time grid via start/stop/num
    S = expm_multiply((-L), delta_vec, start=0.0, stop=float(t_max), num=int(t_num), endpoint=True)
    # S shape is (t_num, n)
    return tvals, S


def distribute_items(items: list[int], rank: int, size: int) -> list[int]:
    n = len(items)
    per = n // size
    rem = n % size
    if rank < rem:
        start = rank * (per + 1)
        end = start + (per + 1)
    else:
        start = rem * (per + 1) + (rank - rem) * per
        end = start + per
    return items[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="../trauma_data/diffusion_inputs", help="Diffusion inputs directory (contains delta.tsv and network file).")
    ap.add_argument("--delta", default="delta.tsv", help="Delta file name inside in_dir.")
    ap.add_argument("--network", default=None, help="Network file name inside in_dir (edgelist or adjacency). If omitted, will auto-detect.")
    ap.add_argument("--sep", default="\t", help="Separator for network file (default: tab).")
    ap.add_argument("--t_max", type=float, default=3.0, help="Max diffusion time.")
    ap.add_argument("--t_num", type=int, default=100, help="Number of diffusion time points.")
    ap.add_argument("--reduction", type=float, default=0.3, help="Knockdown factor for incident edges (e.g. 0.3).")
    ap.add_argument("--out_dir", default=None, help="Output directory. Default: knockouts/results")
    ap.add_argument("--genes_subset", default=None, help="Optional file with one gene per line to test (faster).")
    ap.add_argument("--topk_traces", type=int, default=30, help="Save diffused trajectories for top-K genes by impact.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    delta_path = in_dir / args.delta
    if not delta_path.exists():
        raise FileNotFoundError(delta_path)

    if args.out_dir is None:
        # Default: put results in trauma_results/perturb_diffusion
        script_dir = Path(__file__).resolve().parent
        out_dir = script_dir.parent / "knockouts" / "results" / "GSE37069" # change to GSE37069 or GSE182616
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # MPI setup
    use_mpi = _HAVE_MPI and ("SLURM_JOB_ID" in os.environ or "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ)
    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    if rank == 0:
        print(f"[MPI] use_mpi={use_mpi} size={size}")
        print(f"[LOAD] delta: {delta_path}")

    delta_s = read_delta(delta_path)
    genes = delta_s.index.tolist()
    n = len(genes)
    delta_vec = delta_s.to_numpy(dtype=np.float64)

    # Auto-detect network file if not provided
    if args.network is None:
        candidates = []
        for name in ["network_edgelist.tsv", "network_edgelist.txt", "edgelist.tsv", "network.tsv",
                     "adjacency.tsv", "adjacency.csv", "network_adjacency.tsv", "network_adjacency.csv"]:
            p = in_dir / name
            if p.exists():
                candidates.append(p)
        # also pick any *edgelist* file
        if not candidates:
            for p in in_dir.glob("*edgelist*"):
                if p.is_file():
                    candidates.append(p)
        if not candidates:
            for p in in_dir.glob("*adj*"):
                if p.is_file():
                    candidates.append(p)

        if not candidates:
            raise FileNotFoundError("Could not auto-detect a network file in in_dir. Pass --network <filename>.")
        net_path = candidates[0]
    else:
        net_path = in_dir / args.network
        if not net_path.exists():
            raise FileNotFoundError(net_path)

    if rank == 0:
        print(f"[LOAD] network: {net_path.name}")

    W = load_network_as_sparse(net_path, genes=genes, sep=args.sep)
    # Ensure symmetry-ish
    W = (W + W.T) * 0.5
    W.eliminate_zeros()

    isolates = int(np.sum(np.asarray(W.sum(axis=1)).ravel() == 0))
    if rank == 0:
        print(f"[NET] nodes={n} edges={W.nnz//2} isolates={isolates}")

    L = laplacian_from_adjacency(W)

    # Baseline diffusion
    if rank == 0:
        print(f"[BASELINE] computing diffusion t_max={args.t_max} t_num={args.t_num} ...")
    tvals, S_base = diffuse_signal(L, delta_vec, t_max=args.t_max, t_num=args.t_num)

    # Precompute COO + incident index lists for fast per-node scaling
    rows, cols, data, idx_lists = build_edge_index_lists(W)

    # Choose genes to test
    if args.genes_subset is not None:
        subset_genes = [g.strip() for g in Path(args.genes_subset).read_text().splitlines() if g.strip()]
        subset_idx = [i for i, g in enumerate(genes) if g in set(subset_genes)]
    else:
        subset_idx = list(range(n))

    # Distribute work
    my_idx = distribute_items(subset_idx, rank, size)

    if rank == 0:
        print(f"[PERTURB] reduction={args.reduction} | genes_to_test={len(subset_idx)}")

    start = time.time()
    local_rows = []

    pbar = tqdm(my_idx, desc=f"[Rank {rank}] Perturbing genes", disable=(rank != 0))
    for i in pbar:
        # Modify adjacency weights by scaling all edges incident to node i
        data_mod = data.copy()
        inc = idx_lists[i]
        if inc:
            data_mod[np.array(inc, dtype=np.int64)] *= float(args.reduction)

        W_mod = coo_matrix((data_mod, (rows, cols)), shape=(n, n)).tocsr()
        W_mod.sum_duplicates()
        W_mod.eliminate_zeros()

        L_mod = laplacian_from_adjacency(W_mod)

        _, S_mod = diffuse_signal(L_mod, delta_vec, t_max=args.t_max, t_num=args.t_num)

        # Impact score: max_t ||S_mod(t) - S_base(t)||_2
        diffs = S_mod - S_base
        l2 = np.sqrt(np.sum(diffs * diffs, axis=1))  # shape (t_num,)
        score = float(np.max(l2))

        local_rows.append((genes[i], score))
        pbar.set_postfix({"current_gene": genes[i], "impact": f"{score:.4g}"})
    
    pbar.close()

    # Gather results
    if use_mpi:
        gathered = comm.gather(local_rows, root=0)
    else:
        gathered = [local_rows]

    if rank == 0:
        all_rows = []
        for part in gathered:
            all_rows.extend(part)

        df = pd.DataFrame(all_rows, columns=["gene", "impact_max_l2"])
        df = df.sort_values("impact_max_l2", ascending=False).reset_index(drop=True)

        out_scores = out_dir / f"perturbative_gene_impacts_reduction_{args.reduction:.3f}.tsv"
        df.to_csv(out_scores, sep="\t", index=False)
        print(f"[SAVED] {out_scores}")

        # Save top-K diffusion trajectories (optional, but handy)
        topk = int(args.topk_traces)
        if topk > 0:
            top_genes = df["gene"].head(topk).tolist()
            top_idx = [genes.index(g) for g in top_genes]

            traces = []
            for g, i in zip(top_genes, top_idx):
                data_mod = data.copy()
                inc = idx_lists[i]
                if inc:
                    data_mod[np.array(inc, dtype=np.int64)] *= float(args.reduction)
                W_mod = coo_matrix((data_mod, (rows, cols)), shape=(n, n)).tocsr()
                W_mod.sum_duplicates()
                W_mod.eliminate_zeros()
                L_mod = laplacian_from_adjacency(W_mod)
                _, S_mod = diffuse_signal(L_mod, delta_vec, t_max=args.t_max, t_num=args.t_num)

                # Store per-gene norm trajectory
                l2 = np.sqrt(np.sum((S_mod - S_base) ** 2, axis=1))
                traces.append(pd.Series(l2, index=tvals, name=g))

            traces_df = pd.concat(traces, axis=1)
            out_tr = out_dir / f"top{topk}_impact_trajectories_reduction_{args.reduction:.3f}.tsv"
            traces_df.to_csv(out_tr, sep="\t", index=True)
            print(f"[SAVED] {out_tr}")

        elapsed = int(time.time() - start)
        print(f"[DONE] perturbative diffusion complete in ~{elapsed}s (rank0 wallclock).")


if __name__ == "__main__":
    main()