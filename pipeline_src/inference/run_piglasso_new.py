#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pickle

from glasso_installation import ensure_glasso
from piglasso_core import QJSweeper


# -----------------------------
# path helpers
# -----------------------------
def normalize_input_stem(s: str) -> str:
    s = s.strip()
    if s.endswith(".tsv"):
        s = s[:-4]
    return s


def resolve_input_path(user_arg: str, project_root: Path, in_dir: Path) -> Path:
    """
    Try:
      - <in_dir>/<stem>.tsv
      - user_arg as path
      - user_arg + .tsv as path
    """
    stem = normalize_input_stem(user_arg)
    candidates = [
        in_dir / f"{stem}.tsv",
        Path(user_arg),
        Path(user_arg + ".tsv"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()

    tried = "\n  - " + "\n  - ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Input file not found.\nTried:{tried}")


def list_benchmark_inputs(in_dir: Path) -> list[Path]:
    files = sorted(in_dir.glob("*.tsv"))
    return [f.resolve() for f in files if f.is_file()]


# -----------------------------
# data loader
# -----------------------------
def load_group_tsv(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """
    TSV format: genes x samples/experiments (index=genes, columns=samples)
    Return:
      - data_array: samples x genes (n x p)
      - gene_names: list[str] (length p)
      - sample_names: list[str] (length n)
    """
    X = pd.read_csv(path, sep="\t", index_col=0)
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    gene_names = list(X.index)
    sample_names = X.columns.tolist()
    data_array = X.T.values  # samples x genes

    return data_array, gene_names, sample_names


# -----------------------------
# core runner
# -----------------------------
def load_prior(prior_path: Path, gene_names: list[str]) -> np.ndarray:
    """
    Load a prior matrix and subset it to match gene_names.

    The prior was built from a reference gene list (genes.txt).  If the
    expression data contains a different or smaller set of genes, we load
    that reference list and index into the prior accordingly.  Genes in
    gene_names that are absent from the prior gene list receive a prior
    column/row of zeros (no prior belief — equivalent to no prior for those
    edges).
    """
    prior = np.load(prior_path).astype(np.float32)
    if prior.ndim != 2 or prior.shape[0] != prior.shape[1]:
        raise ValueError(f"Prior must be a square 2-D array, got shape {prior.shape}")

    p_genes = len(gene_names)

    # Exact match — no subsetting needed
    if prior.shape[0] == p_genes:
        return prior

    # Try to load the reference gene list that was used to build the prior
    prior_genes_txt = prior_path.parent / "genes.txt"
    if not prior_genes_txt.exists():
        raise FileNotFoundError(
            f"Prior shape {prior.shape} does not match number of genes {p_genes} "
            f"and reference gene list not found at {prior_genes_txt}. "
            "Rebuild the prior or supply genes.txt alongside the .npy file."
        )

    prior_genes = prior_genes_txt.read_text().strip().split("\n")
    prior_gene_idx = {g: i for i, g in enumerate(prior_genes)}

    # Build index array: for each expression gene, find its position in prior
    indices = [prior_gene_idx[g] for g in gene_names if g in prior_gene_idx]
    found = [g for g in gene_names if g in prior_gene_idx]
    missing = [g for g in gene_names if g not in prior_gene_idx]

    if missing:
        print(
            f"[WARN] {len(missing)}/{p_genes} expression genes not found in prior "
            f"gene list — those edges will have zero prior weight. "
            f"Example missing: {missing[:5]}"
        )

    if not found:
        print("[WARN] No expression genes found in prior gene list — prior has no effect.")
        return np.zeros((p_genes, p_genes), dtype=np.float32)

    # Subset prior to found genes, then expand to full p_genes x p_genes with zeros
    sub = prior[np.ix_(indices, indices)]
    full = np.zeros((p_genes, p_genes), dtype=np.float32)
    found_positions = [i for i, g in enumerate(gene_names) if g in prior_gene_idx]
    full[np.ix_(found_positions, found_positions)] = sub

    print(
        f"[INFO] Prior subsetted: {prior.shape[0]} → {p_genes} genes "
        f"({len(found)} matched, {len(missing)} missing set to 0)"
    )
    return full


def run_one(in_path: Path, out_dir: Path, args) -> Path:
    print(f"[INFO] Loading data: {in_path}")
    data_array, gene_names, sample_names = load_group_tsv(in_path)

    n_samples, p_genes = data_array.shape
    print(f"[INFO] Samples: {n_samples}, Genes: {p_genes}")

    b = int(args.b_perc * n_samples)
    if b <= 1 or b >= n_samples:
        raise ValueError(
            f"Invalid b={b} from b_perc={args.b_perc} with n={n_samples}. Must satisfy 1 < b < n."
        )

    lambda_range = np.linspace(args.llo, args.lhi, args.lamlen)

    # Load prior if requested
    prior_matrix = None
    if args.prior is not None:
        prior_path = Path(args.prior)
        if not prior_path.exists():
            raise FileNotFoundError(f"Prior file not found: {prior_path}")
        prior_matrix = load_prior(prior_path, gene_names)
        print(
            f"[INFO] Prior loaded: {prior_path}  shape={prior_matrix.shape}  "
            f"density={(prior_matrix > 0.05).sum() / 2 / (p_genes * (p_genes - 1) / 2):.4f}  "
            f"prior_weight={args.prior_weight}"
        )
    else:
        print("[INFO] No prior — running standard PIGLasso.")

    print("[DEBUG] building sweeper...")
    sweeper = QJSweeper(
        data=data_array,
        b=b,
        Q=args.Q,
        prior_matrix=prior_matrix,
        prior_weight=args.prior_weight,
        rank=0,
        size=1,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    print("[DEBUG] sweeper built. starting optimization...")

    edge_counts_all, success_counts = sweeper.run_subsample_optimization(lambda_range)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.stem
    prior_tag = f"__pw{args.prior_weight}" if prior_matrix is not None else ""
    out_base = (
        f"{stem}__Q{args.Q}__bperc{args.b_perc}"
        f"__lam{args.llo}-{args.lhi}x{args.lamlen}"
        f"__seed{args.seed}{prior_tag}"
    )
    out_pkl = out_dir / f"{out_base}__piglasso_results.pkl"

    payload = {
        "edge_counts_all": edge_counts_all,
        "success_counts": success_counts,
        "lambda_range": lambda_range,
        "genes": gene_names,
        "samples": sample_names,
        "input_path": str(in_path),
        "n_samples": int(n_samples),
        "p_genes": int(p_genes),
        "Q": int(args.Q),
        "b_perc": float(args.b_perc),
        "b": int(b),
        "llo": float(args.llo),
        "lhi": float(args.lhi),
        "lamlen": int(args.lamlen),
        "seed": int(args.seed),
        "mode": args.mode,
        "prior_path": str(args.prior) if args.prior is not None else None,
        "prior_weight": float(args.prior_weight),
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    (out_dir / f"{out_base}__genes.txt").write_text("\n".join(gene_names) + "\n")

    print(f"[SAVED] {out_pkl}")
    return out_pkl


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["burn", "bench"],
        default="burn",
        help="Select dataset mode: burn (default) or bench (benchmark).",
    )

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Input file stem/name or path. "
            "In burn mode: defaults to burn_data/preprocessed/filtered/<stem>.tsv. "
            "In bench mode: if omitted, runs ALL *.tsv in benchmark_data/preprocessed/."
        ),
    )

    parser.add_argument("--Q", type=int, default=200)
    parser.add_argument("--b_perc", type=float, default=0.65)
    parser.add_argument("--llo", type=float, default=0.05)
    parser.add_argument("--lhi", type=float, default=0.30)
    parser.add_argument("--lamlen", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (1=sequential, -1=all CPUs, N=specific count)",
    )
    parser.add_argument(
        "--prior",
        default=None,
        help=(
            "Path to a .npy prior matrix (p×p, values in [0,1], symmetric, zero diagonal). "
            "When supplied, per-edge regularisation is reduced for high-prior edges: "
            "rho_ij = lambda * (1 - prior_weight * prior_ij). "
            "Omit to run standard PIGLasso without a prior."
        ),
    )
    parser.add_argument(
        "--prior_weight",
        type=float,
        default=0.5,
        help="Strength of prior influence (0 = no effect, 1 = maximum). Default: 0.5.",
    )

    parser.add_argument("--allow_install_glasso", action="store_true")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]  # BurnInjuries/

    # Mode-specific dirs
    if args.mode == "burn":
        in_dir = project_root / "preprocessing" / "burn" / "filtered" / "GSE37069" / "phase" #change this to GSE37069 or GSE182616
        out_dir = project_root / "inference" / "results" / "piglasso" / "GSE37069" #change this to GSE37069 or GSE182616
    else:
        in_dir = project_root / "benchmarking" / "data" / "SGG" / "160" # change to dream, GRN or SGG. if dream or GRN > add: / "preprocessed"
        out_dir = project_root / "benchmarking" / "results" / "piglasso" / "SGG" / "160" # change to dream, GRN or SGG.

    # Ensure R glasso exists
    ensure_glasso(allow_install=args.allow_install_glasso)

    # Decide what to run
    if args.mode == "bench" and args.input is None:
        # Run ALL benchmark TSVs
        inputs = list_benchmark_inputs(in_dir)
        if not inputs:
            raise FileNotFoundError(f"No *.tsv files found in {in_dir}")
        print(f"[INFO] BENCH mode: running {len(inputs)} files from {in_dir}")
    else:
        if args.input is None:
            raise ValueError("--input is required in burn mode. (In bench mode it is optional.)")
        inputs = [resolve_input_path(args.input, project_root, in_dir)]

    # Run sequentially
    saved = []
    for i, fp in enumerate(inputs, start=1):
        print("============================================================")
        print(f"[PROGRESS] [{i}/{len(inputs)}] {fp.name}")
        print(f"[PROGRESS] mode   : {args.mode}")
        print(f"[PROGRESS] in_dir : {in_dir}")
        print(f"[PROGRESS] out_dir: {out_dir}")
        print("============================================================")
        out_pkl = run_one(fp, out_dir, args)
        saved.append(out_pkl)

    print(f"[DONE] Completed {len(saved)} run(s).")


if __name__ == "__main__":
    main()