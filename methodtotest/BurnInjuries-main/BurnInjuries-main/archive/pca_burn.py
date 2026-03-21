#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def resolve_path(root: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    cand = (root / pp).resolve()
    if cand.exists():
        return cand
    return (Path.cwd() / pp).resolve()


def load_metadata(meta_path: Path) -> pd.DataFrame:
    m = pd.read_csv(meta_path, sep="\t", index_col=0)
    m.index = m.index.astype(str)

    # make a consistent gender_code
    if "gender_code" not in m.columns and "gender" in m.columns:
        m["gender_code"] = m["gender"].astype(str)
    return m


def load_group_matrix(path: Path) -> pd.DataFrame:
    """Reads genes x samples TSV."""
    X = pd.read_csv(path, sep="\t", index_col=0)
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def parse_group_from_filename(stem: str) -> str:
    """
    If file is like:
      Massive__YngAdult__AcutePhase__n38__filtered__mvTop1000
    then group = everything before '__mvTop...'
    """
    m = re.split(r"__mvTop\d+", stem, maxsplit=1)
    return m[0] if m else stem


def run_pca_on_matrix(
    X_genes_x_samples: pd.DataFrame,
    n_components: int = 10,
    standardize: bool = True,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns:
      scores_df: samples x PCs
      loadings_df: genes x PCs
      var_ratio: explained variance ratio (Series)
    """
    # PCA expects samples x features
    X = X_genes_x_samples.T  # samples x genes

    if standardize:
        # standardize each gene across samples
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X.values)
    else:
        X_scaled = X.values.astype(float)

    pca = PCA(n_components=min(n_components, X_scaled.shape[1], X_scaled.shape[0]), random_state=random_state)
    scores = pca.fit_transform(X_scaled)  # samples x PCs

    pc_names = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, index=X.index, columns=pc_names)

    # loadings: genes x PCs
    loadings = pca.components_.T  # genes x PCs
    loadings_df = pd.DataFrame(loadings, index=X.columns, columns=pc_names)

    var_ratio = pd.Series(pca.explained_variance_ratio_, index=pc_names, name="explained_variance_ratio")
    return scores_df, loadings_df, var_ratio


def plot_pc_scatter(
    scores_df: pd.DataFrame,
    meta: pd.DataFrame,
    out_png: Path,
    pcx: str = "PC1",
    pcy: str = "PC2",
    color_by: str | None = None,
    title: str | None = None,
):
    df = scores_df.join(meta, how="left")

    plt.figure()
    if color_by is None or color_by not in df.columns:
        plt.scatter(df[pcx], df[pcy])
    else:
        # discrete-ish coloring without manually specifying colors:
        cats = df[color_by].astype(str).fillna("NA")
        for cat in sorted(cats.unique()):
            sub = df[cats == cat]
            plt.scatter(sub[pcx], sub[pcy], label=cat)
        plt.legend(fontsize=8)

    plt.xlabel(pcx)
    plt.ylabel(pcy)
    plt.title(title or f"{pcx} vs {pcy}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="burn_data/preprocessed/gene_subsets",
                    help="Directory containing gene_subset outputs (e.g., *__mvTop1000.tsv).")
    ap.add_argument("--input_tsv", default=None,
                    help="Optional single TSV to run PCA on. If omitted, runs over all *mvTop*.tsv in --in_dir.")
    ap.add_argument("--metadata_tsv", default="burn_data/preprocessed/stratify/burn_sample_metadata.tsv",
                    help="Metadata TSV (indexed by sample_id).")
    ap.add_argument("--out_root", default="burn_results/pca",
                    help="Root output directory. Outputs go into out_root/<severity>/<phase>/ by default.")
    ap.add_argument("--color_by", default="phase_bin",
                    help="Metadata column used for coloring (e.g., phase_bin, tbsa_bucket, age_bucket, mortality).")
    ap.add_argument("--n_components", type=int, default=10)
    ap.add_argument("--no_standardize", action="store_true",
                    help="Disable gene-wise standardization across samples before PCA.")
    ap.add_argument("--random_state", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]  # BurnInjuries/

    in_dir = resolve_path(root, args.in_dir)
    meta_path = resolve_path(root, args.metadata_tsv)
    out_root = resolve_path(root, args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(meta_path)

    # Choose inputs
    if args.input_tsv:
        inputs = [resolve_path(root, args.input_tsv)]
    else:
        inputs = sorted(in_dir.glob("*mvTop*.tsv"))
        if not inputs:
            raise FileNotFoundError(f"No *mvTop*.tsv found in: {in_dir}")

    standardize = not args.no_standardize

    for fp in inputs:
        stem = fp.stem  # no .tsv
        group_prefix = parse_group_from_filename(stem)

        # Try to route outputs into severity/phase folders from the group name
        # Example group: Massive__YngAdult__AcutePhase__n38__filtered
        parts = group_prefix.split("__")
        severity = parts[0] if len(parts) > 0 else "UNKNOWN"
        phase = "UNKNOWN"
        for tok in parts:
            if tok.lower().startswith("acute"):
                phase = "Acute"
            if tok.lower().startswith("late"):
                phase = "Late"

        out_dir = out_root / severity.lower() / phase.lower()
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[PCA] Loading: {fp}")
        X = load_group_matrix(fp)  # genes x samples

        # Align metadata to sample columns
        meta_aligned = meta.reindex(X.columns)

        scores_df, loadings_df, var_ratio = run_pca_on_matrix(
            X,
            n_components=args.n_components,
            standardize=standardize,
            random_state=args.random_state,
        )

        # Save tables
        scores_out = out_dir / f"{stem}__pca_scores.tsv"
        var_out = out_dir / f"{stem}__pca_variance.tsv"
        load_out = out_dir / f"{stem}__pca_loadings.tsv"

        scores_df.join(meta_aligned, how="left").to_csv(scores_out, sep="\t")
        var_ratio.to_frame().to_csv(var_out, sep="\t")
        loadings_df.to_csv(load_out, sep="\t")

        # Plots
        plot_pc_scatter(
            scores_df=scores_df,
            meta=meta_aligned,
            out_png=out_dir / f"{stem}__pca_PC1_PC2.png",
            pcx="PC1",
            pcy="PC2",
            color_by=args.color_by,
            title=f"{stem} (colored by {args.color_by})",
        )
        if "PC3" in scores_df.columns:
            plot_pc_scatter(
                scores_df=scores_df,
                meta=meta_aligned,
                out_png=out_dir / f"{stem}__pca_PC1_PC3.png",
                pcx="PC1",
                pcy="PC3",
                color_by=args.color_by,
                title=f"{stem} (colored by {args.color_by})",
            )

        print(f"[SAVED] {scores_out}")
        print(f"[SAVED] {var_out}")
        print(f"[SAVED] {load_out}")
        print(f"[SAVED] {out_dir / (stem + '__pca_PC1_PC2.png')}")
        print("-" * 60)

    print("[DONE] PCA complete.")


if __name__ == "__main__":
    main()