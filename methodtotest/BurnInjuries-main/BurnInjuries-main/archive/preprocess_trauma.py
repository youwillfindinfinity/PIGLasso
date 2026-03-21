#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def read_aggr_matrix(path: Path) -> pd.DataFrame:
    """
    Reads an aggr.txt file that is space-separated with quoted column names.
    Returns: genes x cells (DataFrame)
    """
    print(f"Reading {path.name}...")
    df = pd.read_csv(path, sep=" ", index_col=0, quotechar='"', low_memory=False)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    df.index = df.index.astype(str)
    print(f"  Loaded: {df.shape[0]} genes × {df.shape[1]} cells")
    return df


def parse_cell_header(cell_id: str) -> dict:
    """
    Example cell column:
      AAACCCAGTCACAGAG.1_MM3001_4h
      AAAC..._MM3001_Ctrl

    Returns dict with patient + timepoint.
    """
    cell_id = cell_id.strip().strip('"')
    parts = cell_id.split("_")
    # last token is timepoint, token before is patient (MM####)
    timepoint = parts[-1] if len(parts) >= 2 else "NA"
    patient = parts[-2] if len(parts) >= 2 else "NA"
    return {"patient": patient, "timepoint": timepoint}


def pseudobulk_by_timepoint(df: pd.DataFrame) -> pd.DataFrame:
    """
    df is genes x cells. Returns genes x (Ctrl,4h,24h,72h,...) pseudo-bulk
    using mean across cells (you can switch to sum if you prefer).
    """
    meta = [parse_cell_header(c) for c in df.columns]
    timepoints = [m["timepoint"] for m in meta]

    groups = defaultdict(list)
    for col, tp in zip(df.columns, timepoints):
        groups[tp].append(col)

    out = {}
    for tp, cols in groups.items():
        # mean expression per gene across cells
        out[tp] = df[cols].mean(axis=1)

    pb = pd.DataFrame(out)
    # put common order first if present
    preferred = ["Ctrl", "4h", "24h", "72h"]
    ordered = [c for c in preferred if c in pb.columns] + [c for c in pb.columns if c not in preferred]
    pb = pb[ordered]
    return pb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True, help="Path to GSE162806_MM####_aggr.txt")
    ap.add_argument("--log1p", action="store_true", help="Apply log1p transform to the matrix (after reading)")
    args = ap.parse_args()

    in_path = Path(args.in_file)
    ROOT = Path(__file__).resolve().parents[1]  # BurnInjuries/
    out_dir = ROOT / "trauma_data"/"preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_aggr_matrix(in_path)  # genes x cells
    # ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if args.log1p:
        df = np.log1p(df)

    # summarize timepoints
    tps = [parse_cell_header(c)["timepoint"] for c in df.columns]
    tp_counts = pd.Series(tps).value_counts()

    # pseudo-bulk
    pb = pseudobulk_by_timepoint(df)

    # save
    raw_out = out_dir / f"{in_path.stem}__genes_x_cells.tsv"
    pb_out = out_dir / f"{in_path.stem}__pseudobulk_genes_x_timepoint.tsv"
    summary_out = out_dir / f"{in_path.stem}__timepoint_counts.tsv"

    df.to_csv(raw_out, sep="\t")
    pb.to_csv(pb_out, sep="\t")
    tp_counts.to_csv(summary_out, sep="\t", header=False)

    print(f"[SAVED] {raw_out}")
    print(f"[SAVED] {pb_out}")
    print(f"[SAVED] {summary_out}")
    print("\nTimepoint counts:")
    print(tp_counts.to_string())


if __name__ == "__main__":
    main()