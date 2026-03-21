"""
GSE236713 preprocessing script.

Downloads GSE236713 from NCBI GEO and extracts two subgroups:
  - Healthy controls  (n=30)  → used as diffusion baseline reference
  - SIRS/OOHCA        (n=42)  → systemic-inflammation non-burn control group

Platform: Agilent-039494 SurePrint G3 Human GE v2 8x60K (GPL17077)
This is the same manufacturer family as GSE182616 (also Agilent), eliminating
cross-platform batch effects between the burn dataset and the healthy baseline.

Pipeline
--------
1. Download series matrix file via NCBI FTP (if not already cached)
2. Parse sample characteristics to identify healthy vs SIRS/OOHCA
3. Extract log2-intensity expression matrices per group
4. Probe-to-gene mapping: highest-variance probe per gene symbol (matching GSE182616)
5. Intersect gene set with the 164-gene NODIS panel (if panel file provided)
6. Save: healthy_controls.npy / sirs_oohca.npy + corresponding .genes.txt

Usage
-----
    python scripts/prepare_gse236713.py --out-dir data/gse236713/

    # With gene panel intersection:
    python scripts/prepare_gse236713.py \
        --out-dir data/gse236713/        \
        --gene-panel data/gene_panel_164.txt

    # Skip download if series matrix already cached:
    python scripts/prepare_gse236713.py \
        --series-matrix data/gse236713/GSE236713_series_matrix.txt.gz \
        --out-dir data/gse236713/
"""
from __future__ import annotations

import argparse
import gzip
import io
import pathlib
import urllib.request
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEO_ACCESSION = "GSE236713"
GPL_ACCESSION = "GPL17077"

# NCBI GEO FTP path for series matrix
_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
_SERIES_MATRIX_URL = (
    f"{_FTP_BASE}/GSE236nnn/{GEO_ACCESSION}/matrix/"
    f"{GEO_ACCESSION}_series_matrix.txt.gz"
)

# Sample group labels (from GSE236713 GEO metadata)
_HEALTHY_KEYWORDS = ("healthy", "control", "normal")
_SIRS_KEYWORDS    = ("sirs", "oohca", "cardiac arrest", "sepsis")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_series_matrix(out_dir: pathlib.Path) -> pathlib.Path:
    """Download the GSE236713 series matrix file if not already cached."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{GEO_ACCESSION}_series_matrix.txt.gz"
    if dest.exists():
        print(f"Cached: {dest}")
        return dest
    print(f"Downloading {GEO_ACCESSION} series matrix ...")
    urllib.request.urlretrieve(_SERIES_MATRIX_URL, dest)
    print(f"Saved to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Parse series matrix
# ---------------------------------------------------------------------------

def _open_series_matrix(path: pathlib.Path) -> io.StringIO:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return io.StringIO(f.read())
    return io.StringIO(path.read_text(encoding="utf-8", errors="replace"))


def parse_series_matrix(path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a GEO series matrix file.

    Returns
    -------
    expr_df   : DataFrame (genes × samples), raw log2 intensities
    meta_df   : DataFrame (samples × metadata fields)
    """
    fh = _open_series_matrix(path)
    lines = fh.readlines()

    meta_rows: dict[str, list[str]] = {}
    sample_ids: list[str] = []
    expr_rows: list[list[str]] = []
    in_table = False

    for line in lines:
        line = line.rstrip("\n")

        if line.startswith("!series_matrix_table_begin"):
            in_table = True
            continue
        if line.startswith("!series_matrix_table_end"):
            in_table = False
            continue

        if in_table:
            parts = line.split("\t")
            expr_rows.append(parts)
            continue

        # Header metadata
        if line.startswith("!Sample_geo_accession"):
            sample_ids = [v.strip('"') for v in line.split("\t")[1:]]
        elif line.startswith("!Sample_"):
            key = line.split("\t")[0].lstrip("!")
            vals = [v.strip('"') for v in line.split("\t")[1:]]
            meta_rows.setdefault(key, []).extend(vals)

    if not sample_ids:
        raise ValueError("No !Sample_geo_accession found in series matrix.")
    if not expr_rows:
        raise ValueError("Expression table (!series_matrix_table) not found.")

    # Build expression DataFrame (first column = probe ID, rest = samples)
    header = expr_rows[0]
    data   = expr_rows[1:]
    probe_ids = [row[0].strip('"') for row in data]
    values    = [[v.strip('"') for v in row[1:]] for row in data]
    expr_df = pd.DataFrame(values, index=probe_ids, columns=sample_ids, dtype=float)

    # Build metadata DataFrame
    # meta_rows[key] may be concatenated from multiple lines → take first n=len(sample_ids)
    meta_dict = {}
    for key, vals in meta_rows.items():
        meta_dict[key] = vals[:len(sample_ids)]
    meta_df = pd.DataFrame(meta_dict, index=sample_ids)

    print(f"Parsed: {expr_df.shape[0]} probes × {expr_df.shape[1]} samples")
    return expr_df, meta_df


# ---------------------------------------------------------------------------
# Group classification
# ---------------------------------------------------------------------------

def classify_samples(meta_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Classify samples into 'healthy' and 'sirs_oohca' groups based on metadata.

    Inspects Sample_characteristics_ch1 (and ch2 if present) for keywords.
    """
    char_cols = [c for c in meta_df.columns if "characteristics" in c.lower()]
    if not char_cols:
        # Fall back to Sample_source_name or Sample_title
        char_cols = [c for c in meta_df.columns if any(
            k in c.lower() for k in ("source_name", "title", "description")
        )]

    groups: dict[str, list[str]] = {"healthy": [], "sirs_oohca": []}

    for sample_id, row in meta_df.iterrows():
        combined = " ".join(str(row[c]) for c in char_cols if c in row.index).lower()
        if any(k in combined for k in _HEALTHY_KEYWORDS):
            groups["healthy"].append(sample_id)
        elif any(k in combined for k in _SIRS_KEYWORDS):
            groups["sirs_oohca"].append(sample_id)
        else:
            warnings.warn(
                f"Sample {sample_id} not classified (metadata: '{combined[:80]}')",
                UserWarning,
                stacklevel=2,
            )

    print(f"Classified: {len(groups['healthy'])} healthy, "
          f"{len(groups['sirs_oohca'])} SIRS/OOHCA")
    return groups


# ---------------------------------------------------------------------------
# Probe → gene mapping
# ---------------------------------------------------------------------------

def probe_to_gene(
    expr_df: pd.DataFrame,
    gene_panel: list[str] | None = None,
) -> pd.DataFrame:
    """
    Collapse Agilent probes to gene-level expression.

    Strategy: for probes that map to the same HGNC symbol, keep the probe
    with highest variance across all samples (matches GSE182616 preprocessing).

    The Agilent GPL17077 series matrix encodes gene symbols in the probe ID
    column when soft-file annotation is unavailable. If probe IDs are numeric
    (Agilent feature numbers), we return the raw probe-level matrix and warn.
    """
    probe_ids = expr_df.index.tolist()
    # Heuristic: if probe IDs look like gene symbols (contain letters), use them
    has_gene_ids = any(not pid.replace("_", "").isdigit() for pid in probe_ids[:20])

    if not has_gene_ids:
        warnings.warn(
            "Probe IDs appear to be numeric Agilent feature numbers. "
            "Gene-symbol mapping requires the GPL17077 annotation file. "
            "Returning probe-level matrix — map manually if needed.",
            UserWarning,
            stacklevel=2,
        )
        gene_expr = expr_df.copy()
    else:
        # Deduplicate: highest-variance probe per gene symbol
        variances = expr_df.var(axis=1)
        gene_expr = (
            expr_df.assign(_var=variances)
            .groupby(expr_df.index)
            .apply(lambda g: g.loc[g["_var"].idxmax()])
            .drop(columns="_var")
        )
        gene_expr.index.name = "gene"

    # Intersect with panel if provided
    if gene_panel is not None:
        overlap = [g for g in gene_panel if g in gene_expr.index]
        missing = [g for g in gene_panel if g not in gene_expr.index]
        if missing:
            warnings.warn(
                f"{len(missing)} panel genes not found in GSE236713: "
                + ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""),
                UserWarning,
                stacklevel=2,
            )
        gene_expr = gene_expr.loc[overlap]
        print(f"Gene panel intersection: {len(overlap)}/{len(gene_panel)} genes retained")

    print(f"Gene-level matrix: {gene_expr.shape[0]} genes × {gene_expr.shape[1]} samples")
    return gene_expr


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_group(
    expr_df: pd.DataFrame,
    sample_ids: list[str],
    name: str,
    out_dir: pathlib.Path,
) -> None:
    """Save one sample group as (n_samples, n_genes) .npy + .genes.txt."""
    sub = expr_df[sample_ids].T  # samples × genes
    arr = sub.values.astype(np.float64)
    genes = list(sub.columns)

    npy_path   = out_dir / f"{name}.npy"
    genes_path = out_dir / f"{name}.genes.txt"

    np.save(npy_path, arr)
    genes_path.write_text("\n".join(genes))

    print(f"Saved {name}: {arr.shape[0]} samples × {arr.shape[1]} genes → {npy_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_gse236713(
    out_dir: pathlib.Path,
    series_matrix: pathlib.Path | None = None,
    gene_panel_path: pathlib.Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Obtain series matrix
    if series_matrix is None:
        series_matrix = download_series_matrix(out_dir)

    # 2. Parse
    expr_df, meta_df = parse_series_matrix(series_matrix)

    # 3. Classify samples
    groups = classify_samples(meta_df)
    if len(groups["healthy"]) == 0:
        raise RuntimeError(
            "No healthy-control samples identified. "
            "Check metadata keywords or inspect meta_df manually."
        )

    # 4. Load gene panel if provided
    gene_panel: list[str] | None = None
    if gene_panel_path is not None:
        gene_panel = [
            line.strip()
            for line in gene_panel_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"Gene panel loaded: {len(gene_panel)} genes from {gene_panel_path}")

    # 5. Probe → gene mapping
    gene_expr = probe_to_gene(expr_df, gene_panel=gene_panel)

    # 6. Save groups
    save_group(gene_expr, groups["healthy"],   "healthy_controls", out_dir)
    if groups["sirs_oohca"]:
        save_group(gene_expr, groups["sirs_oohca"], "sirs_oohca",      out_dir)

    # 7. Save full metadata for audit
    meta_path = out_dir / "sample_metadata.csv"
    meta_df.to_csv(meta_path)
    print(f"Sample metadata saved → {meta_path}")

    print(
        f"\nDone. Outputs in {out_dir}:\n"
        f"  healthy_controls.npy   — {len(groups['healthy'])} samples (diffusion baseline)\n"
        f"  sirs_oohca.npy         — {len(groups['sirs_oohca'])} samples (SIRS reference)\n"
        f"  sample_metadata.csv    — full GEO sample metadata\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and preprocess GSE236713 (Agilent GPL17077) for NODIS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", required=True, type=pathlib.Path,
        help="Output directory for processed files.",
    )
    parser.add_argument(
        "--series-matrix", type=pathlib.Path, default=None,
        help="Path to already-downloaded series matrix (.txt or .txt.gz). "
             "If omitted, the file is downloaded from NCBI GEO FTP.",
    )
    parser.add_argument(
        "--gene-panel", type=pathlib.Path, default=None,
        help="Optional text file (one gene symbol per line) to intersect with. "
             "Use data/gene_panel_164.txt for the 164-gene NODIS panel.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare_gse236713(
        out_dir=args.out_dir,
        series_matrix=args.series_matrix,
        gene_panel_path=args.gene_panel,
    )
