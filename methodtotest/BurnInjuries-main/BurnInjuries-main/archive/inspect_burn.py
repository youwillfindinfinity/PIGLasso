#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
ROOT = ROOT = Path(__file__).resolve().parents[1]  # BurnInjuries/
META_PATH = ROOT / "burn_data" / "burn_sample_metadata.tsv"

if not META_PATH.exists():
    raise FileNotFoundError(f"Missing metadata file: {META_PATH}")

# -----------------------------
# Load metadata
# -----------------------------
meta = pd.read_csv(META_PATH, sep="\t", index_col=0)

print("\n=== BASIC METADATA SHAPE ===")
print(meta.shape)

print("\n=== COLUMNS ===")
print(list(meta.columns))

# -----------------------------
# Age diagnostics
# -----------------------------
print("\n=== AGE DISTRIBUTION ===")
print(meta["age_years"].describe())

print("\n=== AGE BUCKET COUNTS ===")
print(meta["age_bucket"].value_counts(dropna=False))

print("\n=== YOUNGEST PATIENTS ===")
print(meta.loc[meta["age_years"].notna(), "age_years"].sort_values().head(20))

# -----------------------------
# TBSA diagnostics
# -----------------------------
print("\n=== TBSA DISTRIBUTION ===")
print(meta["tbsa_pct"].describe())

print("\n=== TBSA BUCKET COUNTS ===")
print(meta["tbsa_bucket"].value_counts(dropna=False))

print("\n=== SAMPLES WITH MISSING TBSA ===")
print(meta["tbsa_pct"].isna().sum())

# -----------------------------
# Time bin diagnostics
# -----------------------------
print("\n=== TIME BIN COUNTS ===")
print(meta["time_bin"].value_counts(dropna=False))

print("\n=== PHASE BIN COUNTS ===")
print(meta["phase_bin"].value_counts(dropna=False))

# -----------------------------
# FollowUp analysis
# -----------------------------
print("\n=== FOLLOWUP SAMPLES BY TBSA BUCKET ===")
print(
    meta.loc[meta["time_bin"] == "FollowUp", "tbsa_bucket"]
    .value_counts(dropna=False)
)

print("\n=== FOLLOWUP SAMPLES BY AGE BUCKET ===")
print(
    meta.loc[meta["time_bin"] == "FollowUp", "age_bucket"]
    .value_counts(dropna=False)
)

# -----------------------------
# Cross-tabs (most important)
# -----------------------------
print("\n=== TIME BIN × TBSA BUCKET ===")
print(pd.crosstab(meta["time_bin"], meta["tbsa_bucket"], dropna=False))

print("\n=== TIME BIN × AGE BUCKET ===")
print(pd.crosstab(meta["time_bin"], meta["age_bucket"], dropna=False))

print("\n=== PHASE BIN × TBSA BUCKET ===")
print(pd.crosstab(meta["phase_bin"], meta["tbsa_bucket"], dropna=False))

# -----------------------------
# Minimum-N check (for stratification)
# -----------------------------
MIN_N = 10

print(f"\n=== BUCKETS FAILING min_n = {MIN_N} ===")
bucket_cols = ["age_bucket", "tbsa_bucket", "time_bin"]

counts = (
    meta
    .groupby(bucket_cols)
    .size()
    .reset_index(name="n")
)

print(counts.loc[counts["n"] < MIN_N].sort_values("n"))

# -----------------------------
# Mortality sanity check (not used for stratification)
# -----------------------------
if "mortality" in meta.columns:
    print("\n=== MORTALITY COUNTS (OUTCOME ONLY) ===")
    print(meta["mortality"].value_counts(dropna=False))

print("\n=== INSPECTION COMPLETE ===")