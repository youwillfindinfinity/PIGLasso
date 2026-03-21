#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# utilities
# -----------------------------
def resolve_path(root: Path, p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp.resolve()
    cand = (root / pp).resolve()
    if cand.exists():
        return cand
    return (Path.cwd() / pp).resolve()


def parse_group_name(stem: str) -> tuple[str, str, str] | None:
    """
    Expected: TBSA__Age__Acute__nXX
    Returns (tbsa_bucket, age_bucket, phase) or None if doesn't match.
    """
    parts = stem.split("__")
    if len(parts) < 4:
        return None
    tbsa, age, phase = parts[0], parts[1], parts[2]
    if phase not in {"Acute", "Proliferation", "Remodelling"}:
        return None
    return tbsa, age, phase


# -----------------------------
# loaders
# -----------------------------
def load_burn_groups_phases_only(stratify_dir: Path) -> dict[str, pd.DataFrame]:
    groups = {}
    for fp in sorted(stratify_dir.glob("*.tsv")):
        if fp.name == "burn_sample_metadata.tsv":
            continue
        if parse_group_name(fp.stem) is None:
            continue
        X = pd.read_csv(fp, sep="\t", index_col=0)
        X.index = X.index.astype(str)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        groups[fp.stem] = X
    if not groups:
        raise RuntimeError(f"No Acute/Proliferation/Remodelling burn groups found in {stratify_dir}")
    return groups


def load_burn_metadata(meta_path: Path) -> pd.DataFrame:
    m = pd.read_csv(meta_path, sep="\t", index_col=0)
    m.index = m.index.astype(str)
    for col in ["age_years", "gender"]:
        if col not in m.columns:
            raise KeyError(f"Missing '{col}' in metadata: {meta_path}")
    return m


def trauma_gene_set(trauma_dir: Path) -> set[str]:
    files = sorted(trauma_dir.glob("*__pseudobulk_genes_x_timepoint.tsv"))
    if not files:
        raise RuntimeError(f"No trauma pseudobulk files found in {trauma_dir}")
    gene_sets = []
    for fp in files:
        df = pd.read_csv(fp, sep="\t", index_col=0, usecols=[0])
        gene_sets.append(set(df.index.astype(str)))
    return set.intersection(*gene_sets) if gene_sets else set()


# -----------------------------
# filters
# -----------------------------
def mean_expression_keep_index(
    X: pd.DataFrame,
    keep_quantile: float = 0.90,
    min_mean: float | None = None,
) -> pd.Index:
    """
    X: genes x samples
    Keep genes with high mean expression within THIS group.

    - If min_mean is provided: keep mean >= min_mean
    - Else: keep mean >= quantile(keep_quantile)
      Example keep_quantile=0.90 keeps top 90% by mean, drops bottom 10%.
    """
    mu = X.mean(axis=1)

    if min_mean is not None:
        return mu.index[mu >= float(min_mean)]

    q = float(keep_quantile)
    if not (0.0 < q <= 1.0):
        raise ValueError("keep_quantile must be in (0, 1].")
    thr = np.quantile(mu.values, q)
    return mu.index[mu >= thr]


def regress_out_age_gender(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.reindex(X.columns).copy()

    m["age_years"] = pd.to_numeric(m["age_years"], errors="coerce")
    m["age_years"] = m["age_years"].fillna(m["age_years"].median())

    m["gender"] = (
        m["gender"].astype(str)
        .replace({"nan": "Unknown", "None": "Unknown"})
        .fillna("Unknown")
    )
    G = pd.get_dummies(m["gender"], drop_first=True)

    Z = pd.concat(
        [
            pd.Series(1.0, index=X.columns, name="intercept"),
            m["age_years"].rename("age_years"),
            G,
        ],
        axis=1,
    ).astype(float)

    Y = X.values.astype(float)   # genes x n
    Zm = Z.values.astype(float)  # n x k
    beta = (Y @ Zm) @ np.linalg.pinv(Zm.T @ Zm)  # genes x k
    Y_hat = beta @ Zm.T                          # genes x n
    R = Y - Y_hat
    return pd.DataFrame(R, index=X.index, columns=X.columns)


def mean_variance_adjust_keep(mu: pd.Series, var: pd.Series, q=0.75) -> pd.Index:
    x = np.log10(mu.astype(float).clip(1e-12).values)
    y = np.log10(var.astype(float).clip(1e-12).values)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    thr = np.quantile(resid, q)
    keep = resid >= thr
    return mu.index[keep]


# -----------------------------
# main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burn_stratify_dir", default="burn_data/preprocessed/stratified")
    ap.add_argument("--burn_metadata_tsv", default=None)
    ap.add_argument("--trauma_preprocessed_dir", default="trauma_data/preprocessed")
    ap.add_argument("--out_dir", default="burn_data/preprocessed/filtered")
    ap.add_argument("--mv_quantile_keep", type=float, default=0.75)

    # mean-expression filtering per group
    ap.add_argument(
        "--mean_keep_quantile",
        type=float,
        default=0.90,
        help="Keep genes with mean >= this quantile within each group (default: 0.90).",
    )
    ap.add_argument(
        "--min_mean",
        type=float,
        default=None,
        help="Optional absolute mean-expression cutoff (overrides --mean_keep_quantile).",
    )

    # Optional mean-variance residual filtering per group
    ap.add_argument(
        "--skip_mv", default="skip_mv",
        action="store_true",
        help="If set, skip MV-residual filtering and keep genes after intersection + mean-expression filter.",
    )

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]  # BurnInjuries/
    burn_dir = resolve_path(root, args.burn_stratify_dir)
    trauma_dir = resolve_path(root, args.trauma_preprocessed_dir)
    out_dir = resolve_path(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = resolve_path(
        root,
        args.burn_metadata_tsv if args.burn_metadata_tsv else (burn_dir / "burn_sample_metadata.tsv"),
    )

    meta = load_burn_metadata(meta_path)
    burn_groups = load_burn_groups_phases_only(burn_dir)
    trauma_genes = trauma_gene_set(trauma_dir)

    mean_mode = f"min_mean={args.min_mean}" if args.min_mean is not None else f"keep_quantile={args.mean_keep_quantile}"
    print("[INFO] burn groups loaded:", len(burn_groups))
    print("[INFO] trauma intersection genes:", len(trauma_genes))
    print(f"[INFO] mean-filter mode: {mean_mode}")
    print(f"[INFO] MV filtering: {'SKIPPED' if args.skip_mv else f'ON (q={args.mv_quantile_keep})'}")

    summary_rows = []

    for name, X0 in burn_groups.items():
        parsed = parse_group_name(name)
        assert parsed is not None
        tbsa, age, phase = parsed

        genes_before = int(X0.shape[0])
        n_samples = int(X0.shape[1])

        # 1) Burn ∩ Trauma
        X1 = X0.loc[X0.index.intersection(trauma_genes)]
        genes_after_intersection = int(X1.shape[0])

        # 2) Mean-expression filter within this group
        keep_mean = mean_expression_keep_index(
            X1,
            keep_quantile=args.mean_keep_quantile,
            min_mean=args.min_mean,
        )
        X1m = X1.loc[keep_mean]
        genes_after_mean = int(X1m.shape[0])

        # 3) Optional MV residual filtering
        if args.skip_mv:
            X2 = X1m
            genes_after_mv = int(X2.shape[0])
        else:
            R = regress_out_age_gender(X1m, meta)
            mu = X1m.mean(axis=1)
            var_resid = R.var(axis=1, ddof=1)
            keep_mv = mean_variance_adjust_keep(mu, var_resid, q=args.mv_quantile_keep)
            X2 = X1m.loc[keep_mv]
            genes_after_mv = int(X2.shape[0])

        out_fp = out_dir / f"{name}__filtered.tsv"
        X2.to_csv(out_fp, sep="\t")

        summary_rows.append(
            {
                "group": name,
                "tbsa_bucket": tbsa,
                "age_bucket": age,
                "phase": phase,
                "n_samples": n_samples,
                "genes_before": genes_before,
                "genes_after_intersection": genes_after_intersection,
                "genes_after_mean_filter": genes_after_mean,
                "did_mv": (not args.skip_mv),
                "genes_after_mv_adjust": genes_after_mv,
            }
        )

        print(f"[WRITE] {out_fp.name}: {genes_before} -> {genes_after_intersection} -> {genes_after_mean} -> {genes_after_mv}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["tbsa_bucket", "age_bucket", "phase", "group"])
    summary_out = out_dir / "filter_summary.tsv"
    summary_df.to_csv(summary_out, sep="\t", index=False)

    print(f"[SAVED] {summary_out}")
    print("[DONE]")

if __name__ == "__main__":
    main()