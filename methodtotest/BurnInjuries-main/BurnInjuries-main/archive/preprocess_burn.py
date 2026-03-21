#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from io import StringIO
import re
import numpy as np
import pandas as pd


# -----------------------------
# Readers
# -----------------------------
def read_agilent_annotation(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.strip('"')
        df.loc[df[c].isin(["nan", "NaN", "None"]), c] = ""
    return df


def read_geo_series_matrix_full(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - metadata_df with columns: key, values(list[str])
      - expr_df: ID_REF + GSM columns (strings)
    """
    path = Path(path)
    meta_rows = []
    table_lines = []
    in_table = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue

            if in_table:
                table_lines.append(line)
            elif line.startswith("!"):
                meta_rows.append(line)

    meta_parsed = []
    for row in meta_rows:
        parts = row.split("\t")
        key = parts[0].lstrip("!")
        values = parts[1:]
        meta_parsed.append((key, values))

    metadata_df = pd.DataFrame(
        {"key": [k for k, _ in meta_parsed], "values": [v for _, v in meta_parsed]}
    )

    if not table_lines:
        raise ValueError("No series matrix table found between begin/end markers.")

    expr_df = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t", dtype=str, low_memory=False)
    expr_df.columns = [c.strip().strip('"') for c in expr_df.columns]
    for c in expr_df.columns:
        expr_df[c] = expr_df[c].astype(str).str.strip().str.strip('"')
        expr_df.loc[expr_df[c].isin(["nan", "NaN", "None"]), c] = ""

    return metadata_df, expr_df


# -----------------------------
# Metadata parsing + stratify helpers
# -----------------------------
def parse_kv_cell(s: str) -> tuple[str, str] | None:
    if s is None:
        return None
    s = str(s).strip().strip('"')
    if ":" not in s:
        return None
    k, v = s.split(":", 1)
    return k.strip().lower(), v.strip()


def to_float(x):
    try:
        if x is None:
            return None
        xs = str(x).strip()
        if xs.lower() in {"na", "nan", ""}:
            return None
        xs = xs.replace("%", "")
        return float(xs)
    except Exception:
        return None


def parse_time_to_hours(raw: str) -> float | None:
    if raw is None:
        return None
    t = str(raw).strip().strip('"').strip()
    t = t.replace("_", "").replace(" ", "").lower()

    m = re.match(r"^hr(\d+)$", t)
    if m:
        return float(m.group(1))

    m = re.match(r"^d(\d+)$", t)
    if m:
        return 24.0 * float(m.group(1))

    return None


def assign_time_bin(time_hours: float | None) -> str | None:
    if time_hours is None:
        return None
    if time_hours == 0:
        return "T0"
    if 1 <= time_hours < 24:
        return "Early"
    if 24 <= time_hours < 96:
        return "Mid"
    if 96 <= time_hours < 168:
        return "Late"
    if time_hours >= 169:  # D14 + D21
        return "FollowUp"
    return "Other"


def assign_phase_bin(time_bin: str | None) -> str | None:
    if time_bin is None:
        return None
    if time_bin in {"T0", "Early", "Mid"}:
        return "Acute"
    if time_bin in {"Late"}:
        return "Proliferation"
    if time_bin in {"FollowUp"}:
        return "Remodelling"
    return None


def assign_age_bucket(age_years: float | None) -> str | None:
    if age_years is None:
        return None
    if 18 <= age_years <= 40:
        return "YngAdult"
    if 41 <= age_years <= 64:
        return "MidAdult"
    if 65 <= age_years <= 85:
        return "Elder"
    return "OtherAge"


def assign_tbsa_bucket(tbsa_pct: float | None) -> str | None:
    if tbsa_pct is None:
        return None
    if tbsa_pct < 10:
        return "Mild"
    if 10 <= tbsa_pct < 20:
        return "Moderate"
    if 20 <= tbsa_pct <= 40:
        return "Severe"
    return "Massive"


def safe_label(x: str) -> str:
    x = str(x)
    x = re.sub(r"\s+", "_", x.strip())
    x = re.sub(r"[^A-Za-z0-9_\-]+", "", x)
    return x


def extract_sample_metadata(meta_df: pd.DataFrame) -> pd.DataFrame:
    geo_acc_rows = meta_df[meta_df["key"].str.lower() == "sample_geo_accession"]
    if geo_acc_rows.empty:
        raise ValueError("Could not find '!Sample_geo_accession' in series matrix metadata.")
    geo_samples = [str(x).strip().strip('"') for x in geo_acc_rows.iloc[0]["values"]]
    per_sample = {gsm: {} for gsm in geo_samples}

    char_rows = meta_df[
        meta_df["key"].str.lower().isin(["sample_characteristics_ch1", "sample_characteristics_ch2"])
    ]
    if char_rows.empty:
        out = pd.DataFrame(index=geo_samples)
        out.index.name = "sample_id"
        return out

    for _, row in char_rows.iterrows():
        values = row["values"]
        n = min(len(values), len(geo_samples))
        for j in range(n):
            gsm = geo_samples[j]
            kv = parse_kv_cell(values[j])
            if kv is None:
                continue
            k, v = kv
            per_sample[gsm][k] = v

    meta = pd.DataFrame.from_dict(per_sample, orient="index")
    meta.index.name = "sample_id"

    # normalized fields
    if "age" in meta.columns:
        meta["age_years"] = meta["age"].apply(to_float)
    if "tbsa" in meta.columns:
        meta["tbsa_pct"] = meta["tbsa"].apply(to_float)

    # keep whichever exists
    if "gender" in meta.columns:
        meta["gender"] = meta["gender"].astype(str).str.strip()
    elif "sex" in meta.columns:
        meta["gender"] = meta["sex"].astype(str).str.strip()

    return meta


# -----------------------------
# Core preprocessing (same as preprocess_burn2, no mean filter)
# -----------------------------
def preprocess_burn_agilent(
    expr_df: pd.DataFrame,
    anno_df: pd.DataFrame,
    probe_col_expr: str = "ID_REF",
    probe_col_anno: str = "ID",
    gene_col_anno: str = "GENE_SYMBOL",
    control_col_anno: str = "CONTROL_TYPE",
    collapse: str = "var",   # "var" | "mean" | "median"
    winsor_q: float = 0.01,
    min_non_na_frac: float = 0.95,
    min_var: float = 1e-8,
    debug: bool = True,
) -> pd.DataFrame:
    """
    Returns gene x sample matrix (z-scored per gene).
    No mean-expression filtering here.
    """
    expr = expr_df.copy()
    anno = anno_df.copy()

    if probe_col_expr not in expr.columns:
        raise ValueError(f"Expected '{probe_col_expr}' in expression table; got {expr.columns[:20].tolist()}")
    if probe_col_anno not in anno.columns:
        raise ValueError(f"Expected '{probe_col_anno}' in annotation table; got {anno.columns[:20].tolist()}")
    if gene_col_anno not in anno.columns:
        raise ValueError(f"Expected '{gene_col_anno}' in annotation table; got {anno.columns[:20].tolist()}")

    expr = expr.rename(columns={probe_col_expr: "probe_id"})
    anno = anno.rename(columns={probe_col_anno: "probe_id", gene_col_anno: "gene_symbol"})
    if control_col_anno in anno.columns:
        anno = anno.rename(columns={control_col_anno: "control_type"})

    keep_anno = ["probe_id", "gene_symbol"] + (["control_type"] if "control_type" in anno.columns else [])
    anno = anno[keep_anno].copy()

    expr["probe_id"] = expr["probe_id"].astype(str).str.strip().str.strip('"')
    anno["probe_id"] = anno["probe_id"].astype(str).str.strip().str.strip('"')
    anno["gene_symbol"] = anno["gene_symbol"].fillna("").astype(str).str.strip()

    merged = expr.merge(anno, on="probe_id", how="left")

    meta_cols = {"probe_id", "gene_symbol", "control_type"}
    sample_cols = [c for c in merged.columns if c not in meta_cols]

    for c in sample_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    merged["gene_symbol"] = merged["gene_symbol"].fillna("").astype(str).str.strip()
    merged = merged[merged["gene_symbol"].str.len() > 0]

    if "control_type" in merged.columns:
        ct = merged["control_type"].fillna("").astype(str).str.lower().str.strip()
        keep_ct = {"", "0", "false", "f"}
        merged = merged[ct.isin(keep_ct)]

    non_na_frac = merged[sample_cols].notna().mean(axis=1)
    merged = merged[non_na_frac >= min_non_na_frac]
    if merged.empty:
        raise RuntimeError("All rows were filtered out after missingness/controls filtering.")

    X = merged[["gene_symbol"] + sample_cols].copy()

    if collapse == "mean":
        gene_mat = X.groupby("gene_symbol")[sample_cols].mean()
    elif collapse == "median":
        gene_mat = X.groupby("gene_symbol")[sample_cols].median()
    elif collapse == "var":
        X["_var"] = X[sample_cols].var(axis=1, ddof=1)
        idx = X.groupby("gene_symbol")["_var"].idxmax()
        gene_mat = X.loc[idx].set_index("gene_symbol")[sample_cols]
    else:
        raise ValueError("collapse must be one of: var, mean, median")

    v = gene_mat.var(axis=1, ddof=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gene_mat = gene_mat.loc[v > min_var]

    lo = gene_mat.quantile(winsor_q, axis=1)
    hi = gene_mat.quantile(1 - winsor_q, axis=1)
    gene_mat = gene_mat.apply(lambda row: row.clip(lo[row.name], hi[row.name]), axis=1)

    mu = gene_mat.mean(axis=1)
    sd = gene_mat.std(axis=1, ddof=1).replace(0, np.nan)
    gene_mat_z = gene_mat.sub(mu, axis=0).div(sd, axis=0).dropna(axis=0)

    if debug:
        print("\n[DEBUG] Final gene matrix")
        print("  gene_mat_z.shape:", gene_mat_z.shape)

    return gene_mat_z


# -----------------------------
# Combined pipeline
# -----------------------------
def run_pipeline(
    series_path: Path,
    anno_path: Path,
    out_dir: Path,
    min_n: int = 10,
    debug: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load series matrix + annotation
    if debug:
        print(f"[INFO] Loading annotation: {anno_path}")
    anno_df = read_agilent_annotation(anno_path)

    if debug:
        print(f"[INFO] Loading series matrix: {series_path}")
    meta_df, expr_df = read_geo_series_matrix_full(series_path)

    # 2) Build sample metadata (age/tbsa/gender + time bins)
    meta = extract_sample_metadata(meta_df)
    meta["time_point_raw"] = meta["time point"] if "time point" in meta.columns else None
    meta["time_hours"] = meta["time_point_raw"].apply(parse_time_to_hours)
    meta["time_bin"] = meta["time_hours"].apply(assign_time_bin)
    meta["phase_bin"] = meta["time_bin"].apply(assign_phase_bin)

    meta["tbsa_bucket"] = meta.get("tbsa_pct", pd.Series(index=meta.index, dtype=float)).apply(assign_tbsa_bucket)
    meta["age_bucket"] = meta.get("age_years", pd.Series(index=meta.index, dtype=float)).apply(assign_age_bucket)

    # 3) Preprocess expression into gene x sample z-scored matrix
    gene_mat_z = preprocess_burn_agilent(
        expr_df=expr_df,
        anno_df=anno_df,
        probe_col_expr="ID_REF",
        probe_col_anno="ID",
        min_non_na_frac=0.95,
        debug=debug,
    )

    # 4) Align metadata to expression samples
    sample_cols = list(gene_mat_z.columns.astype(str))
    meta_aligned = meta.loc[meta.index.intersection(sample_cols)].copy()
    meta_aligned = meta_aligned.reindex(sample_cols)

    if debug:
        print(f"\n[INFO] Metadata aligned shape: {meta_aligned.shape}")
        for col in ["tbsa_bucket", "age_bucket", "time_bin", "phase_bin"]:
            if col in meta_aligned.columns:
                print(f"\n[INFO] {col} counts:")
                print(meta_aligned[col].value_counts(dropna=False))

    # 5) Save metadata once
    out_meta = out_dir / "burn_sample_metadata.tsv"
    meta_aligned.to_csv(out_meta, sep="\t")
    print(f"\n[SAVED] {out_meta}")

    # 6) Export group matrices
    keep_time_bins = {"T0", "Early", "Mid", "Late", "FollowUp"}
    keep_phases = {"Acute", "Proliferation", "Remodelling"}
    keep_age = {"YngAdult", "MidAdult", "Elder"}
    keep_tbsa = {"Mild", "Moderate", "Severe", "Massive"}

    exported = 0

    # TBSA x Age x Time
    for tbsa_b in sorted(keep_tbsa):
        for age_b in sorted(keep_age):
            for time_b in sorted(keep_time_bins):
                mask = (
                    (meta_aligned["tbsa_bucket"] == tbsa_b)
                    & (meta_aligned["age_bucket"] == age_b)
                    & (meta_aligned["time_bin"] == time_b)
                )
                cols = meta_aligned.index[mask].tolist()
                if len(cols) < min_n:
                    continue
                sub = gene_mat_z[cols]
                fname = f"{safe_label(tbsa_b)}__{safe_label(age_b)}__{safe_label(time_b)}__n{len(cols)}.tsv"
                out_path = out_dir / fname
                sub.to_csv(out_path, sep="\t")
                print(f"[WRITE] {out_path.name}  (genes x {len(cols)} samples)")
                exported += 1

    # TBSA x Age x Phase
    for tbsa_b in sorted(keep_tbsa):
        for age_b in sorted(keep_age):
            for phase_b in sorted(keep_phases):
                mask = (
                    (meta_aligned["tbsa_bucket"] == tbsa_b)
                    & (meta_aligned["age_bucket"] == age_b)
                    & (meta_aligned["phase_bin"] == phase_b)
                )
                cols = meta_aligned.index[mask].tolist()
                if len(cols) < min_n:
                    continue
                sub = gene_mat_z[cols]
                fname = f"{safe_label(tbsa_b)}__{safe_label(age_b)}__{safe_label(phase_b)}__n{len(cols)}.tsv"
                out_path = out_dir / fname
                sub.to_csv(out_path, sep="\t")
                print(f"[WRITE] {out_path.name}  (genes x {len(cols)} samples)")
                exported += 1

    print(f"\n[DONE] Exported {exported} group matrices to: {out_dir}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]  # BurnInjuries/
    DATA = ROOT / "burn_data"

    series_path = DATA / "Burn_GSE182616_series_matrix.txt"
    anno_path = DATA / "BurnAgilentNotation_GPL17077-17467.txt"

    out_dir = DATA / "preprocessed" / "stratified"

    run_pipeline(
        series_path=series_path,
        anno_path=anno_path,
        out_dir=out_dir,
        min_n=10,
        debug=True,
    )