#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


def load_network_bundle(inferred_pkl: Path) -> dict:
    with open(inferred_pkl, "rb") as f:
        bundle = pickle.load(f)

    required = ["genes", "adjacency", "expr_tsv"]
    for k in required:
        if k not in bundle:
            raise KeyError(f"Missing key '{k}' in bundle. Is this the __inferred.pkl from network_inference.py?")

    bundle["genes"] = list(map(str, bundle["genes"]))
    bundle["adjacency"] = np.array(bundle["adjacency"], dtype=int)
    return bundle


def load_burn_expr(expr_tsv: Path, genes_in_network: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    expr_tsv is genes x samples TSV with first col as gene index.
    Returns:
      X_aligned: genes_in_network x samples
      burn_state: mean across samples (Series indexed by genes_in_network)
    """
    X = pd.read_csv(expr_tsv, sep="\t", index_col=0)
    X.index = X.index.astype(str)

    missing = [g for g in genes_in_network if g not in X.index]
    if missing:
        raise RuntimeError(
            f"Burn expr TSV is missing {len(missing)} network genes. "
            f"Example missing: {missing[:10]}"
        )

    X_aligned = X.loc[genes_in_network]
    X_aligned = X_aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    burn_state = X_aligned.mean(axis=1)
    burn_state.name = "burn_state"
    return X_aligned, burn_state


def load_ctrl_pseudobulk(pseudobulk_paths: List[Path]) -> Dict[str, pd.DataFrame]:
    """
    Each file: genes x timepoints (Ctrl, 4h, 24h, 72h, ...)
    Returns dict: patient_id -> DataFrame(genes x timepoints)
    """
    out: Dict[str, pd.DataFrame] = {}
    for p in pseudobulk_paths:
        df = pd.read_csv(p, sep="\t", index_col=0)
        df.index = df.index.astype(str)
        df.columns = [str(c) for c in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        stem = p.stem
        patient = "UNKNOWN"
        parts = stem.split("_")
        for token in parts:
            if token.startswith("MM") and token[2:].isdigit():
                patient = token
                break
        out[patient] = df
    return out


def build_ctrl_reference(
    ctrl_pb: Dict[str, pd.DataFrame],
    genes_in_network: List[str],
    ctrl_col: str = "Ctrl",
    min_patients: int = 1,
    min_common_genes: int = 50,
) -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    """
    Builds cohort mean Ctrl vector, but ONLY on the common gene intersection
    across all included patients. Does NOT expand to full network size.

    Returns:
      ctrl_mean: Series indexed by common_genes
      ctrl_mat: DataFrame indexed by common_genes, cols=patients
      common_genes: list of genes (ordered as in genes_in_network)
    """
    ctrl_vectors: Dict[str, pd.Series] = {}

    for patient, df in ctrl_pb.items():
        if ctrl_col not in df.columns:
            continue

        present = [g for g in genes_in_network if g in df.index]
        if len(present) < 10:
            continue

        v = df.loc[present, ctrl_col].copy()
        v.name = patient
        ctrl_vectors[patient] = v

    if len(ctrl_vectors) < min_patients:
        raise RuntimeError(
            f"Found Ctrl in only {len(ctrl_vectors)} ctrl patient files "
            f"(min required={min_patients}). Check your pseudobulk columns."
        )

    # intersection across all included patients
    common = set(genes_in_network)
    for v in ctrl_vectors.values():
        common &= set(v.index)

    common_genes = [g for g in genes_in_network if g in common]

    if len(common_genes) < min_common_genes:
        raise RuntimeError(
            f"Common Ctrl gene intersection is too small: {len(common_genes)} genes "
            f"(min_common_genes={min_common_genes}). "
            f"Try using more consistent preprocessing or fewer patients for now."
        )

    ctrl_mat = pd.DataFrame({pat: ctrl_vectors[pat].loc[common_genes] for pat in ctrl_vectors})
    ctrl_mean = ctrl_mat.mean(axis=1)
    ctrl_mean.name = "ctrl_reference"

    return ctrl_mean, ctrl_mat, common_genes


def subset_adjacency(adj: np.ndarray, genes: List[str], keep_genes: List[str]) -> Tuple[np.ndarray, List[str]]:
    idx = {g: i for i, g in enumerate(genes)}
    keep_idx = [idx[g] for g in keep_genes]
    sub = adj[np.ix_(keep_idx, keep_idx)]
    return sub, keep_genes


def adjacency_to_edgelist(adj: np.ndarray, genes: List[str]) -> pd.DataFrame:
    rows = []
    p = adj.shape[0]
    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j] != 0:
                rows.append((genes[i], genes[j], int(adj[i, j])))
    return pd.DataFrame(rows, columns=["gene_i", "gene_j", "edge"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burn_inferred_pkl", required=True)
    ap.add_argument("--burn_expr_tsv", default=None)
    ap.add_argument("--ctrl_pseudobulk_dir", required=True)
    ap.add_argument("--ctrl_col", default="Ctrl")
    ap.add_argument("--min_patients", type=int, default=1)
    ap.add_argument("--min_common_genes", type=int, default=50)
    ap.add_argument("--out_dir", default="ctrl_data/diffusion_inputs")
    ap.add_argument("--project_root", default=None)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent

    inferred_pkl = Path(args.burn_inferred_pkl)
    if not inferred_pkl.is_absolute():
        inferred_pkl = project_root / inferred_pkl
    if not inferred_pkl.exists():
        raise FileNotFoundError(f"Missing: {inferred_pkl}")

    bundle = load_network_bundle(inferred_pkl)
    genes_full = bundle["genes"]
    adj_full = bundle["adjacency"]

    # Burn expression
    burn_expr_tsv = Path(bundle["expr_tsv"]) if args.burn_expr_tsv is None else Path(args.burn_expr_tsv)
    if not burn_expr_tsv.is_absolute():
        burn_expr_tsv = project_root / burn_expr_tsv
    if not burn_expr_tsv.exists():
        raise FileNotFoundError(f"Missing burn expr TSV: {burn_expr_tsv}")

    burn_X_full, burn_state_full = load_burn_expr(burn_expr_tsv, genes_full)

    # GSE37069 ctrl pseudobulk
    ctrl_dir = Path(args.ctrl_pseudobulk_dir)
    if not ctrl_dir.is_absolute():
        ctrl_dir = project_root / ctrl_dir
    if not ctrl_dir.exists():
        raise FileNotFoundError(f"Missing ctrl dir: {ctrl_dir}")

    ctrl_files = sorted(ctrl_dir.glob("*__pseudobulk_genes_x_timepoint.tsv"))
    if not ctrl_files:
        raise FileNotFoundError(f"No pseudobulk TSV files found in: {ctrl_dir}")

    ctrl_pb = load_ctrl_pseudobulk(ctrl_files)

    # build reference ONLY on common genes 
    ctrl_ref, ctrl_mat, common_genes = build_ctrl_reference(
        ctrl_pb=ctrl_pb,
        genes_in_network=genes_full,
        ctrl_col=args.ctrl_col,
        min_patients=args.min_patients,
        min_common_genes=args.min_common_genes,
    )

    # Restrict burn state to common genes
    burn_state = burn_state_full.loc[common_genes]
    burn_state.name = "burn_state"

    # Delta only on common genes
    delta = (burn_state - ctrl_ref).astype(float)
    delta.name = "delta_burn_minus_ctrl"

    # Restrict network to induced subgraph on common genes
    adj, genes = subset_adjacency(adj_full, genes_full, common_genes)

    # Output
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    burn_state.to_frame().to_csv(out_dir / "burn_state.tsv", sep="\t")
    ctrl_ref.to_frame().to_csv(out_dir / "ctrl_reference.tsv", sep="\t")
    delta.to_frame().to_csv(out_dir / "delta.tsv", sep="\t")

    ctrl_mat.to_csv(out_dir / "ctrl_matrix.tsv", sep="\t")
    Path(out_dir / "common_genes.txt").write_text("\n".join(genes) + "\n")

    adj_df = pd.DataFrame(adj, index=genes, columns=genes)
    adj_df.to_csv(out_dir / "burn_network_adjacency.csv")

    edgelist = adjacency_to_edgelist(adj, genes)
    edgelist.to_csv(out_dir / "burn_network_edgelist.tsv", sep="\t", index=False)

    meta = {
        "burn_inferred_pkl": str(inferred_pkl),
        "burn_expr_tsv": str(burn_expr_tsv),
        "ctrl_pseudobulk_dir": str(ctrl_dir),
        "n_network_genes_full": len(genes_full),
        "n_network_genes_common": len(genes),
        "n_burn_samples": int(burn_X_full.shape[1]),
        "n_ctrl_files_found": len(ctrl_files),
        "n_ctrl_patients_used": int(ctrl_mat.shape[1]),
        "ctrl_col": args.ctrl_col,
        "min_patients": int(args.min_patients),
        "min_common_genes": int(args.min_common_genes),
    }
    pd.Series(meta).to_csv(out_dir / "summary.tsv", sep="\t", header=False)

    print("[DONE] Built diffusion inputs (COMMON GENES ONLY)")
    print(f"  out_dir: {out_dir}")
    print(f"  network genes (full): {len(genes_full)}")
    print(f"  network genes (common): {len(genes)}")
    print(f"  burn samples: {burn_X_full.shape[1]}")
    print(f"  ctrl files found: {len(ctrl_files)}")
    print(f"  ctrl patients used: {ctrl_mat.shape[1]}")
    print("  saved: delta.tsv, burn_state.tsv, ctrl_reference.tsv, common_genes.txt, induced network")


if __name__ == "__main__":
    main()