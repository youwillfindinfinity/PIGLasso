#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


TBSA_LEVELS = ["Mild", "Moderate", "Severe", "Massive"]


def resolve_path(root: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    cand = (root / pp).resolve()
    if cand.exists():
        return cand
    return (Path.cwd() / pp).resolve()


def load_group_matrix(tsv_path: Path) -> pd.DataFrame:
    """genes x samples"""
    X = pd.read_csv(tsv_path, sep="\t", index_col=0)
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def parse_tbsa_from_name(name: str) -> str:
    """Extract TBSA group from filename stem."""
    for lvl in TBSA_LEVELS:
        if re.search(rf"(^|__){lvl}(__|$)", name):
            return lvl
        if name.startswith(lvl + "__"):
            return lvl
    return "Unknown"


def plot_phase(
    phase: str,
    files: list[Path],
    out_dir: Path,
    n_components: int = 10,
    pc_x: int = 1,
    pc_y: int = 2,
    standardize: bool = True,
    min_union_genes: int = 50,
) -> None:
    if not files:
        print(f"[WARN] No files for phase={phase}. Skipping.")
        return

    print(f"[INFO] Phase={phase}: found {len(files)} file(s).")

    # Load matrices
    mats: dict[str, pd.DataFrame] = {}
    gene_sets: list[set[str]] = []
    for fp in files:
        X = load_group_matrix(fp)
        mats[fp.stem] = X
        gene_sets.append(set(X.index.astype(str)))

    # ---- UNION (not intersection) ----
    union_genes = sorted(set.union(*gene_sets))
    if len(union_genes) < min_union_genes:
        raise RuntimeError(f"Too few union genes across {phase} files: {len(union_genes)}")
    print(f"[INFO] Phase={phase}: union genes = {len(union_genes)}")

    # Build combined sample x gene matrix (align each group to union gene space)
    sample_rows = []
    sample_meta = []

    for stem, X in mats.items():
        tbsa = parse_tbsa_from_name(stem)

        # genes x samples -> reindex to union -> fill missing -> transpose to samples x genes
        Xt = X.reindex(union_genes).fillna(0.0).T

        # Make sample IDs unique immediately to avoid collisions across groups
        Xt.index = [f"{s}__{stem}" for s in Xt.index.astype(str)]

        for s in Xt.index:
            sample_meta.append({"sample": s, "group": stem, "tbsa": tbsa, "phase": phase})

        sample_rows.append(Xt)

    X_all = pd.concat(sample_rows, axis=0)  # (total_samples x union_genes)
    meta_df = pd.DataFrame(sample_meta).set_index("sample")

    # Safety: ensure perfect alignment
    if not X_all.index.equals(meta_df.index):
        meta_df = meta_df.reindex(X_all.index)

    # Standardize features (recommended for PCA)
    X_mat = X_all.values.astype(float)
    if standardize:
        X_mat = StandardScaler(with_mean=True, with_std=True).fit_transform(X_mat)

    pca = PCA(n_components=n_components, random_state=0)
    Z = pca.fit_transform(X_mat)  # samples x PCs

    # Output dirs
    phase_dir = out_dir / phase.lower()
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Save explained variance
    evr = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(n_components)],
        name="explained_variance_ratio",
    )
    evr_out = phase_dir / f"pca_{phase.lower()}__explained_variance.tsv"
    evr.to_csv(evr_out, sep="\t")
    print(f"[SAVED] {evr_out}")

    # Plot PCx vs PCy
    pcx_i = pc_x - 1
    pcy_i = pc_y - 1
    if pcx_i >= n_components or pcy_i >= n_components:
        raise ValueError(f"pc_x/pc_y exceed n_components={n_components}")

    df_plot = meta_df.copy()
    df_plot[f"PC{pc_x}"] = Z[:, pcx_i]
    df_plot[f"PC{pc_y}"] = Z[:, pcy_i]

    plt.figure(figsize=(9, 6))
    levels = TBSA_LEVELS + (["Unknown"] if (df_plot["tbsa"] == "Unknown").any() else [])
    for lvl in levels:
        sub = df_plot[df_plot["tbsa"] == lvl]
        if sub.empty:
            continue
        plt.scatter(sub[f"PC{pc_x}"], sub[f"PC{pc_y}"], label=lvl, s=35, alpha=0.85)

    plt.xlabel(f"PC{pc_x} ({pca.explained_variance_ratio_[pcx_i]*100:.1f}%)")
    plt.ylabel(f"PC{pc_y} ({pca.explained_variance_ratio_[pcy_i]*100:.1f}%)")
    plt.title(f"PCA {phase} (union genes; colored by TBSA) | genes={len(union_genes)} | samples={X_all.shape[0]}")
    plt.legend(title="TBSA", frameon=True)
    plt.tight_layout()

    fig_out = phase_dir / f"pca_{phase.lower()}__PC{pc_x}_vs_PC{pc_y}__by_tbsa.png"
    plt.savefig(fig_out, dpi=200)
    plt.close()
    print(f"[SAVED] {fig_out}")

    # Save coordinates table
    coords_out = phase_dir / f"pca_{phase.lower()}__coords.tsv"
    df_plot.to_csv(coords_out, sep="\t")
    print(f"[SAVED] {coords_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="burn_data/preprocessed/gene_subsets",
                    help="Directory containing mvTop*.tsv outputs from gene_subset.py")
    ap.add_argument("--out_dir", default="burn_results/pca",
                    help="Output directory for PCA plots/tables")
    ap.add_argument("--pattern", default="*__mvTop1000.tsv",
                    help="Glob pattern to select files (default: *__mvTop1000.tsv)")
    ap.add_argument("--pc_x", type=int, default=1, help="PC on x-axis (1-indexed)")
    ap.add_argument("--pc_y", type=int, default=2, help="PC on y-axis (1-indexed)")
    ap.add_argument("--n_components", type=int, default=10)
    ap.add_argument("--no_standardize", action="store_true",
                    help="Disable standardization before PCA (only if data already z-scored)")
    ap.add_argument("--min_union_genes", type=int, default=50,
                    help="Fail if union gene set smaller than this (default: 50)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]  # BurnInjuries/
    in_dir = resolve_path(root, args.input_dir)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {in_dir} matching pattern {args.pattern}")

    acute_files = [fp for fp in files if "AcutePhase" in fp.stem]
    late_files = [fp for fp in files if "LatePhase" in fp.stem]

    plot_phase(
        phase="AcutePhase",
        files=acute_files,
        out_dir=out_dir,
        n_components=args.n_components,
        pc_x=args.pc_x,
        pc_y=args.pc_y,
        standardize=not args.no_standardize,
        min_union_genes=args.min_union_genes,
    )
    plot_phase(
        phase="LatePhase",
        files=late_files,
        out_dir=out_dir,
        n_components=args.n_components,
        pc_x=args.pc_x,
        pc_y=args.pc_y,
        standardize=not args.no_standardize,
        min_union_genes=args.min_union_genes,
    )

    print("[DONE] PCA by phase finished.")


if __name__ == "__main__":
    main()