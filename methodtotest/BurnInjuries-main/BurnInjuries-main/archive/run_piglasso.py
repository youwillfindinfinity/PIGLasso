from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pickle

from inference.glasso_installation import ensure_glasso
from inference.piglasso_core import QJSweeper


def normalize_input_stem(s: str) -> str:
    s = s.strip()

    # remove extension if present
    if s.endswith(".tsv"):
        s = s[:-4]

    return s


def resolve_input_path(user_arg: str, project_root: Path) -> Path:
    """
    Default search dir: burn_data/preprocessed/filtered/
    We try:
      - burn_data/preprocessed/filtered/<stem>.tsv
      - as a fallback: interpret user_arg as a path
    """
    in_dir = project_root / "burn_data" / "preprocessed" / "filtered"
    stem = normalize_input_stem(user_arg)

    candidates = [
        in_dir / f"{stem}.tsv",
        Path(user_arg),                 # raw arg (could be a path)
        Path(user_arg + ".tsv"),        # raw + .tsv
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()

    tried = "\n  - " + "\n  - ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        f"Input file not found.\nTried:{tried}\n\n"
        f"Tip: your filtered group files live in:\n"
        f"  burn_data/preprocessed/filtered/\n"
        f"So run e.g.:\n"
        f"  python3 monika/run_piglasso.py --input Massive__MidAdult__Acute__n42__filtered\n"
    )


def load_group_tsv(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """
    TSV format: genes x samples (index=genes, columns=GSMs)
    Return:
      - data_array: samples x genes (n x p)
      - gene_names: list[str] (length p)
      - sample_names: list[str] (length n)
    """
    X = pd.read_csv(path, sep="\t", index_col=0)

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    gene_names = list(X.index)
    sample_names = X.columns.astype(str).tolist()
    data_array = X.T.values  # samples x genes

    return data_array, gene_names, sample_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Group input file stem or name (defaults to burn_data/preprocessed/filtered/). "
             "Example: Early__Alive__n176__top4000",
    )
    parser.add_argument("--Q", type=int, default=200)
    parser.add_argument("--b_perc", type=float, default=0.65)
    parser.add_argument("--llo", type=float, default=0.05)
    parser.add_argument("--lhi", type=float, default=0.30)
    parser.add_argument("--lamlen", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=1, 
                        help="Number of parallel jobs (1=sequential, -1=all CPUs, N=specific count)")

    # If i want to allow R install attempts on laptop:
    parser.add_argument("--allow_install_glasso", action="store_true")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]  # BurnInjuries/

    # Ensure R glasso exists
    ensure_glasso(allow_install=args.allow_install_glasso)

    # Resolve input path (supports giving stem without .tsv)
    in_path = resolve_input_path(args.input, project_root)

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

    print("[DEBUG] building sweeper...")
    sweeper = QJSweeper(
        data=data_array,
        b=b,
        Q=args.Q,
        rank=0,
        size=1,
        seed=args.seed,
        n_jobs=args.n_jobs,  # pass parallelization setting
    )
    print("[DEBUG] sweeper built. starting optimization...")

    edge_counts_all, success_counts = sweeper.run_subsample_optimization(lambda_range)

    # Output directory
    out_dir = project_root / "burn_results" / "piglasso_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract just the filename stem, not the full path
    stem = in_path.stem  # gets filename without extension
    out_base = f"{stem}__Q{args.Q}__bperc{args.b_perc}__lam{args.llo}-{args.lhi}x{args.lamlen}"

    out_pkl = out_dir / f"{out_base}__piglasso_results.pkl"

    payload = {
        "edge_counts_all": edge_counts_all,     # shape: (p, p, lamlen)
        "success_counts": success_counts,       # shape: (lamlen,)
        "lambda_range": lambda_range,           # shape: (lamlen,)
        "genes": gene_names,                    # list[str], length p
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
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)

    # (Optional) keep a readable gene list too
    (out_dir / f"{out_base}__genes.txt").write_text("\n".join(gene_names) + "\n")

    print(f"[SAVED] {out_pkl}")
    print(f"[SAVED] {out_dir / (out_base + '__genes.txt')}")
    print("[DONE]")

if __name__ == "__main__":
    main()