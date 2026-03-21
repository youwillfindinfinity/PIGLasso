"""
NODIS data preparation utility.

Converts any tabular expression matrix (CSV, TSV, Excel, or numpy array)
into a ready-to-use (n_samples, n_genes) numpy array for NODIS inference.

Pipeline
--------
1. Load   — auto-detect format; samples as rows or columns (auto-detected)
2. Filter — remove genes with zero or near-zero variance
3. Log    — optional log1p transform (recommended for raw counts)
4. NPN    — nonparanormal shrinkage transform (recommended for non-Gaussian data)
5. Return — (n, p) float64 numpy array, gene names list

Usage (CLI)
-----------
    python scripts/prepare_data.py \
        --input  data/expression.csv \
        --output data/expression_prepared.npy \
        --log \
        --npn \
        --min-var 1e-6

Usage (Python)
--------------
    from scripts.prepare_data import prepare_expression_matrix
    X, gene_names = prepare_expression_matrix("data/expression.csv", log=True, npn=True)
"""
from __future__ import annotations

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd

from nodis.preprocess.npn import npn_shrinkage


def prepare_expression_matrix(
    path: str | pathlib.Path | np.ndarray | pd.DataFrame,
    *,
    log: bool = False,
    npn: bool = True,
    min_var: float = 1e-8,
    transpose: bool | None = None,
    sep: str | None = None,
    index_col: int | None = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Load and prepare any expression matrix for NODIS inference.

    Parameters
    ----------
    path       : File path (CSV/TSV/xlsx/npy) or a numpy array or DataFrame.
                 Rows can be samples OR genes — auto-detected if ``transpose``
                 is None (the longer dimension is taken as genes).
    log        : Apply log1p transform before NPN. Use for raw count data.
    npn        : Apply nonparanormal shrinkage transform (recommended).
    min_var    : Genes with variance < min_var are removed (zero-variance filter).
    transpose  : If True, transpose the matrix before processing.
                 If None (default), auto-detect orientation.
    sep        : Column separator for text files. Auto-detected if None.
    index_col  : Column to use as gene/sample index when reading files.

    Returns
    -------
    X          : (n_samples, n_genes) float64 array, ready for DesparifiedGGM.fit()
    gene_names : List of gene/feature names (empty list if not available).
    """
    gene_names: list[str] = []

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    if isinstance(path, np.ndarray):
        X_raw = path.astype(float)
    elif isinstance(path, pd.DataFrame):
        gene_names = list(path.columns)
        X_raw = path.values.astype(float)
    else:
        path = pathlib.Path(path)
        suffix = path.suffix.lower()

        if suffix == ".npy":
            X_raw = np.load(path).astype(float)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, index_col=index_col)
            gene_names = list(df.columns)
            X_raw = df.values.astype(float)
        else:
            # Auto-detect separator for text files
            if sep is None:
                sep = "\t" if suffix in (".tsv", ".txt") else ","
            df = pd.read_csv(path, sep=sep, index_col=index_col)
            gene_names = list(df.columns)
            X_raw = df.values.astype(float)

    # ------------------------------------------------------------------
    # 2. Orient: rows = samples, columns = genes
    # ------------------------------------------------------------------
    if transpose is True:
        X_raw = X_raw.T
        gene_names = []          # row names become gene names but we dropped them
    elif transpose is None:
        # Heuristic: if more rows than columns, assume rows = genes → transpose
        if X_raw.shape[0] > X_raw.shape[1]:
            warnings.warn(
                f"Matrix has shape {X_raw.shape} (more rows than columns). "
                "Assuming rows are genes — transposing so rows = samples. "
                "Pass transpose=False to suppress this.",
                UserWarning,
                stacklevel=2,
            )
            X_raw = X_raw.T

    n, p = X_raw.shape

    # ------------------------------------------------------------------
    # 3. Validate
    # ------------------------------------------------------------------
    if not np.isfinite(X_raw).all():
        n_bad = (~np.isfinite(X_raw)).sum()
        warnings.warn(
            f"{n_bad} non-finite values found. Replacing with column means.",
            UserWarning,
            stacklevel=2,
        )
        col_means = np.nanmean(X_raw, axis=0)
        inds = np.where(~np.isfinite(X_raw))
        X_raw[inds] = col_means[inds[1]]

    # ------------------------------------------------------------------
    # 4. Log transform (for raw count data)
    # ------------------------------------------------------------------
    if log:
        if (X_raw < 0).any():
            raise ValueError(
                "log=True requires non-negative values. "
                "Data contains negative entries."
            )
        X_raw = np.log1p(X_raw)

    # ------------------------------------------------------------------
    # 5. Zero-variance filter
    # ------------------------------------------------------------------
    variances = X_raw.var(axis=0)
    keep = variances >= min_var
    n_removed = (~keep).sum()
    if n_removed > 0:
        warnings.warn(
            f"Removed {n_removed} gene(s) with variance < {min_var}.",
            UserWarning,
            stacklevel=2,
        )
        X_raw = X_raw[:, keep]
        if gene_names:
            gene_names = [g for g, k in zip(gene_names, keep) if k]

    # ------------------------------------------------------------------
    # 6. NPN transform
    # ------------------------------------------------------------------
    if npn:
        X_out = npn_shrinkage(X_raw)
    else:
        X_out = X_raw.copy()

    print(
        f"Prepared matrix: {X_out.shape[0]} samples × {X_out.shape[1]} genes"
        + (f"  |  log1p={log}  NPN={npn}" )
    )
    return X_out.astype(np.float64), gene_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an expression matrix for NODIS inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",   required=True, help="Path to expression matrix (CSV/TSV/xlsx/npy)")
    parser.add_argument("--output",  required=True, help="Output path for prepared .npy array")
    parser.add_argument("--log",     action="store_true", help="Apply log1p transform (for raw counts)")
    parser.add_argument("--no-npn",  action="store_true", help="Skip NPN transform")
    parser.add_argument("--min-var", type=float, default=1e-8, help="Minimum gene variance threshold")
    parser.add_argument("--transpose", action="store_true", default=None,
                        help="Transpose input matrix (rows → columns)")
    parser.add_argument("--sep", default=None, help="Column separator (auto-detected if omitted)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, genes = prepare_expression_matrix(
        args.input,
        log=args.log,
        npn=not args.no_npn,
        min_var=args.min_var,
        transpose=args.transpose,
        sep=args.sep,
    )
    np.save(out_path, X)
    print(f"Saved to {out_path}")

    if genes:
        gene_path = out_path.with_suffix(".genes.txt")
        gene_path.write_text("\n".join(genes))
        print(f"Gene names saved to {gene_path}")
