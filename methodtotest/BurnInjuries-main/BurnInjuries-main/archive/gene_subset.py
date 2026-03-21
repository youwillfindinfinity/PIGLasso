#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


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

    if "age_years" not in m.columns:
        raise KeyError(f"Missing 'age_years' in {meta_path}")

    # stratify sometimes stores 'gender' not 'gender_code'
    if "gender_code" not in m.columns and "gender" not in m.columns:
        raise KeyError(f"Missing 'gender_code' or 'gender' in {meta_path}")
    if "gender_code" not in m.columns:
        m["gender_code"] = m["gender"].astype(str)

    return m


def load_group_matrix(path: Path) -> pd.DataFrame:
    X = pd.read_csv(path, sep="\t", index_col=0)
    X.index = X.index.astype(str)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def pick_filtered_file(filtered_dir: Path, group: str) -> Path:
    """
    If group.tsv exists, use it.
    Else, pick the matching TSV with the MOST genes (to avoid accidentally selecting mvTopXXX).
    """
    exact = filtered_dir / f"{group}.tsv"
    if exact.exists():
        return exact

    matches = sorted(filtered_dir.glob(f"{group}*.tsv"))
    if not matches:
        raise FileNotFoundError(f"No filtered TSV matching: {filtered_dir}/{group}*.tsv")

    best_fp = None
    best_genes = -1
    for fp in matches:
        Xtmp = pd.read_csv(fp, sep="\t", index_col=0, usecols=[0])
        n = int(Xtmp.shape[0])
        if n > best_genes:
            best_genes = n
            best_fp = fp

    assert best_fp is not None
    print(f"[PICK] using filtered file with most genes: {best_fp.name} (genes={best_genes})")
    return best_fp


def regress_out_age_gender(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Regress each gene across samples on [intercept, age_years, gender one-hot] and return residuals.
    X: genes x samples
    """
    m = meta.reindex(X.columns).copy()

    if m.isna().all().any():
        bad_cols = [c for c in m.columns if m[c].isna().all()]
        raise RuntimeError(f"Metadata did not align to samples (all-NA columns): {bad_cols}")

    # age
    m["age_years"] = pd.to_numeric(m["age_years"], errors="coerce")
    m["age_years"] = m["age_years"].fillna(m["age_years"].median())

    # gender
    m["gender_code"] = (
        m["gender_code"]
        .astype(str)
        .replace({"nan": "Unknown", "None": "Unknown"})
        .fillna("Unknown")
    )
    g = pd.get_dummies(m["gender_code"], prefix="gender", drop_first=True)

    Z = pd.concat(
        [
            pd.Series(1.0, index=X.columns, name="intercept"),
            m["age_years"].astype(float).rename("age_years"),
            g,
        ],
        axis=1,
    ).astype(float)

    Y = X.values.astype(float)    # genes x n
    Zm = Z.values.astype(float)   # n x k

    ZTZ_inv = np.linalg.pinv(Zm.T @ Zm)
    B = (Y @ Zm) @ ZTZ_inv
    Y_hat = B @ Zm.T
    R = Y - Y_hat

    return pd.DataFrame(R, index=X.index, columns=X.columns)


def mv_residual_scores(mu: pd.Series, var_resid: pd.Series, eps: float = 1e-12) -> pd.Series:
    """
    Fit log10(var_resid) ~ a + b*log10(mu), return residuals (higher = more variable than expected).
    """
    mu = mu.astype(float).clip(lower=eps)
    var_resid = var_resid.astype(float).clip(lower=eps)

    x = np.log10(mu.values)
    y = np.log10(var_resid.values)

    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    resid = y - y_hat

    return pd.Series(resid, index=mu.index, name="mv_residual")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--group",
        required=True,
        help="Group stem (without .tsv). This should match the *filtered* group prefix.",
    )
    ap.add_argument(
        "--filtered_dir",
        default="burn_data/preprocessed/filtered",
        help="Folder containing your filtered group TSVs (~2800 genes).",
    )
    ap.add_argument(
        "--metadata_tsv",
        default="burn_data/preprocessed/stratify/burn_sample_metadata.tsv",
        help="Metadata TSV (needs age_years and gender/gender_code).",
    )
    ap.add_argument(
        "--out_dir",
        default="burn_data/preprocessed/gene_subsets",
        help="Where to write subset TSVs + ranking.",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="How many top MV-residual genes to keep (default: 1000).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]  # BurnInjuries/
    filtered_dir = resolve_path(root, args.filtered_dir)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = resolve_path(root, args.metadata_tsv)
    meta = load_metadata(meta_path)

    group = args.group

    # 1) Load filtered dataset (~2800 genes)
    filt_fp = pick_filtered_file(filtered_dir, group)
    print(f"[LOAD] filtered : {filt_fp}")
    X = load_group_matrix(filt_fp)

    # 2) Rank genes using covariate-regressed variance + MV residual
    R = regress_out_age_gender(X, meta)
    mu = X.mean(axis=1)
    var_resid = R.var(axis=1, ddof=1)
    scores = mv_residual_scores(mu, var_resid).sort_values(ascending=False)

    rank_df = pd.DataFrame(
        {
            "mean_expr": mu.loc[scores.index],
            "var_resid": var_resid.loc[scores.index],
            "mv_residual": scores,
        }
    )

    rank_out = out_dir / f"{group}__mv_ranking.tsv"
    rank_df.to_csv(rank_out, sep="\t")
    print(f"[SAVED] {rank_out} (ranked genes={rank_df.shape[0]})")

    # 3) Keep top_k
    top_k = int(args.top_k)
    if top_k > rank_df.shape[0]:
        raise RuntimeError(f"Requested top_k={top_k}, but only {rank_df.shape[0]} genes available.")

    keep = rank_df.index[:top_k]
    X_sub = X.loc[keep, X.columns]

    out_fp = out_dir / f"{group}__mvTop{top_k}.tsv"
    X_sub.to_csv(out_fp, sep="\t")
    (out_dir / f"{group}__mvTop{top_k}__genes.txt").write_text("\n".join(keep) + "\n")

    print(f"[WRITE] {out_fp.name}  genes={top_k} samples={X_sub.shape[1]}")
    print("[DONE]")


if __name__ == "__main__":
    main()