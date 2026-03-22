"""
GSE182616 preprocessing script.

Downloads GSE182616 from NCBI GEO and produces an Acute-phase burn expression
matrix ready for PIGLasso inference.

Dataset
-------
GSE182616 — Burn injury blood transcriptomics, Agilent GPL17077-17467
Platform: Agilent-039494 SurePrint G3 Human GE v2 8x60K (same platform as
          GSE236713, eliminating inter-platform batch effects)
Samples: burn patients measured at multiple post-injury timepoints.
Acute phase = T0 (0 hr) + Early (1–23 hr) + Mid (24–95 hr).

Pipeline
--------
1. Download series matrix from NCBI GEO FTP (if not already cached)
2. Download GPL17077 annotation file from NCBI GEO FTP (if not cached)
3. Parse sample metadata: time-point → phase assignment
4. Preprocess expression:
     a. Merge probe IDs with annotation → HGNC gene symbols
     b. Remove control probes and high-missingness probes (> 5% missing)
     c. Collapse duplicate genes: keep highest-variance probe per gene
     d. Winsorise at 1st/99th percentile per gene
     e. Gene-wise z-score normalisation
5. Filter to Acute-phase samples only
6. Intersect with prior gene list (optional, recommended)
7. Output: genes × samples TSV + genes.txt + sample metadata CSV

Usage
-----
    # Basic (downloads all data automatically):
    python scripts/prepare_gse182616.py --out-dir data/burn/

    # Skip re-download if files already cached:
    python scripts/prepare_gse182616.py \\
        --out-dir data/burn/ \\
        --series-matrix data/burn/GSE182616_series_matrix.txt.gz \\
        --annotation data/burn/GPL17077.annot.gz

    # Intersect with prior gene list:
    python scripts/prepare_gse182616.py \\
        --out-dir data/burn/ \\
        --prior-genes pipeline_src/prior/genes.txt

    # All phases (not just Acute):
    python scripts/prepare_gse182616.py \\
        --out-dir data/burn/ \\
        --phase all
"""
from __future__ import annotations

import argparse
import gzip
import io
import pathlib
import re
import urllib.request
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEO_ACCESSION = "GSE182616"
GPL_ACCESSION = "GPL17077"

_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo"

_SERIES_MATRIX_URL = (
    f"{_FTP_BASE}/series/GSE182nnn/{GEO_ACCESSION}/matrix/"
    f"{GEO_ACCESSION}_series_matrix.txt.gz"
)

# GEO eutils — GPL17077 platform annotation only (no sample data, ~28 MB)
_ANNOTATION_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    f"?acc={GPL_ACCESSION}&targ=gpl&form=text&view=full"
)

# Time-point label → hours mapping (Agilent GSE182616 metadata labels)
_TIME_LABEL_MAP: dict[str, float] = {
    "hr0": 0.0, "hr4": 4.0, "hr24": 24.0, "hr108": 108.0,
    "d14": 14 * 24.0, "d21": 21 * 24.0,
}

# Phase assignment (matching notebook cells 4–6)
_PHASE_OF: dict[str, str] = {
    "T0": "Acute",       # hr0
    "Early": "Acute",    # hr4
    "Mid": "Acute",      # hr24 + hr108 (108 hr < 5 days)
    "Late": "Proliferation",   # 5–7 days
    "FollowUp": "Remodelling", # 14+ days
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: pathlib.Path, label: str) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Cached: {dest}")
        return dest
    print(f"Downloading {label} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved: {dest}")
    return dest


def download_series_matrix(out_dir: pathlib.Path) -> pathlib.Path:
    dest = out_dir / f"{GEO_ACCESSION}_series_matrix.txt.gz"
    return _download(_SERIES_MATRIX_URL, dest, f"{GEO_ACCESSION} series matrix")


def download_annotation(out_dir: pathlib.Path) -> pathlib.Path:
    dest = out_dir / f"{GPL_ACCESSION}.annot.txt"
    return _download(_ANNOTATION_URL, dest, f"{GPL_ACCESSION} annotation")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _open_text(path: pathlib.Path) -> io.StringIO:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return io.StringIO(f.read())
    # Try gzip even for non-.gz suffix (some GEO files are gzip despite name)
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return io.StringIO(f.read())
    except (gzip.BadGzipFile, OSError):
        return io.StringIO(path.read_text(encoding="utf-8", errors="replace"))


def parse_series_matrix(path: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a GEO series matrix file.

    Returns
    -------
    expr_df : DataFrame — probe IDs as index, GSM accessions as columns (log2 intensities)
    meta_df : DataFrame — key/values pairs from series header rows
    """
    fh = _open_text(path)
    lines = fh.readlines()

    meta_rows: list[tuple[str, list[str]]] = []
    expr_lines: list[str] = []
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
            expr_lines.append(line)
        elif line.startswith("!"):
            parts = line.split("\t")
            key = parts[0].lstrip("!")
            vals = [v.strip('"') for v in parts[1:]]
            meta_rows.append((key, vals))

    if not expr_lines:
        raise ValueError("Expression table not found in series matrix.")

    meta_df = pd.DataFrame({"key": [k for k, _ in meta_rows],
                             "values": [v for _, v in meta_rows]})

    expr_df = pd.read_csv(
        io.StringIO("\n".join(expr_lines)),
        sep="\t", dtype=str, low_memory=False,
    )
    expr_df.columns = [c.strip().strip('"') for c in expr_df.columns]
    for c in expr_df.columns:
        expr_df[c] = expr_df[c].astype(str).str.strip().str.strip('"')

    probe_col = expr_df.columns[0]
    expr_df = expr_df.set_index(probe_col)
    expr_df.index.name = "probe_id"
    expr_df.index = expr_df.index.astype(str).str.strip()

    # Convert to numeric
    for c in expr_df.columns:
        expr_df[c] = pd.to_numeric(expr_df[c], errors="coerce")

    n_samples = expr_df.shape[1]
    print(f"Series matrix: {expr_df.shape[0]} probes × {n_samples} samples")
    return expr_df, meta_df


def parse_annotation(path: pathlib.Path) -> pd.DataFrame:
    """
    Parse GPL17077 annotation file.

    Handles two formats:
    - GEO soft format (downloaded via eutils): has ``!platform_table_begin`` /
      ``!platform_table_end`` markers with tab-separated data between them.
    - Custom flat files (e.g. BurnAgilentNotation_GPL17077-17467.txt): plain
      tab-separated with optional leading ``#`` comment rows.

    Returns
    -------
    DataFrame with columns: 'probe_id', 'gene_symbol', and optionally 'control_type'
    """
    fh = _open_text(path)
    raw = fh.read()

    # --- Try GEO soft format first ---
    if "!platform_table_begin" in raw:
        lines = raw.split("\n")
        table_lines: list[str] = []
        in_table = False
        for line in lines:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                table_lines.append(line)
        if not table_lines:
            raise ValueError("!platform_table_begin found but table is empty.")
        anno_df = pd.read_csv(
            io.StringIO("\n".join(table_lines)),
            sep="\t", dtype=str, low_memory=False,
        )
    else:
        # Plain flat file — skip '#' comment rows
        non_comment = [
            ln for ln in raw.split("\n")
            if not ln.startswith("#") and not ln.startswith("^") and ln.strip()
        ]
        if not non_comment:
            raise ValueError(f"Annotation file is empty after stripping comments: {path}")
        anno_df = pd.read_csv(
            io.StringIO("\n".join(non_comment)),
            sep="\t", dtype=str, low_memory=False,
        )
    anno_df.columns = [c.strip() for c in anno_df.columns]
    for c in anno_df.columns:
        anno_df[c] = anno_df[c].astype(str).str.strip().str.strip('"')
        anno_df.loc[anno_df[c].isin(["nan", "NaN", "None", "NA"]), c] = ""

    # --- Detect probe ID column ---
    probe_col_candidates = ["ID", "SPOT_ID", "ProbeID", "Probe_Id"]
    probe_col = next((c for c in probe_col_candidates if c in anno_df.columns), None)
    if probe_col is None:
        raise ValueError(
            f"Cannot find probe ID column in annotation. "
            f"Expected one of {probe_col_candidates}. Got: {anno_df.columns.tolist()[:10]}"
        )

    # --- Detect gene symbol column ---
    gene_col_candidates = [
        "GENE_SYMBOL", "Gene Symbol", "gene_symbol",
        "GeneSymbol", "Symbol", "ORF",
    ]
    gene_col = next((c for c in gene_col_candidates if c in anno_df.columns), None)
    if gene_col is None:
        raise ValueError(
            f"Cannot find gene symbol column in annotation. "
            f"Expected one of {gene_col_candidates}. Got: {anno_df.columns.tolist()[:10]}"
        )

    # --- Detect control type column (optional) ---
    ctrl_col_candidates = ["CONTROL_TYPE", "Control_Type", "control_type", "ControlType"]
    ctrl_col = next((c for c in ctrl_col_candidates if c in anno_df.columns), None)

    out = pd.DataFrame({
        "probe_id": anno_df[probe_col].astype(str).str.strip(),
        "gene_symbol": anno_df[gene_col].fillna("").astype(str).str.strip(),
    })
    if ctrl_col:
        out["control_type"] = anno_df[ctrl_col].fillna("").astype(str).str.lower().str.strip()

    print(f"Annotation: {len(out)} probes, gene symbol column = '{gene_col}'")
    return out


# ---------------------------------------------------------------------------
# Sample metadata
# ---------------------------------------------------------------------------

def _parse_kv(s: str) -> tuple[str, str] | None:
    s = s.strip().strip('"')
    if ":" not in s:
        return None
    k, v = s.split(":", 1)
    return k.strip().lower(), v.strip()


def _parse_time_hours(raw: str) -> float | None:
    t = str(raw).strip().lower().replace("_", "").replace(" ", "")
    m = re.match(r"^hr(\d+)$", t)
    if m:
        return float(m.group(1))
    m = re.match(r"^d(\d+)$", t)
    if m:
        return 24.0 * float(m.group(1))
    # Also try just a number (hours)
    try:
        return float(t)
    except ValueError:
        return None


def _time_bin(hours: float | None) -> str | None:
    if hours is None:
        return None
    if hours == 0:
        return "T0"
    if 1 <= hours < 24:
        return "Early"
    if 24 <= hours < 120:
        return "Mid"
    if 120 <= hours < 168:
        return "Late"
    if hours >= 168:
        return "FollowUp"
    return None


def extract_sample_metadata(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-sample metadata from the series matrix header DataFrame.

    Returns
    -------
    DataFrame indexed by GSM accession with columns for available metadata
    fields (age, gender, time point, phase, etc.)
    """
    acc_row = meta_df[meta_df["key"].str.lower() == "sample_geo_accession"]
    if acc_row.empty:
        raise ValueError("!Sample_geo_accession not found in series matrix.")
    gsm_ids = [str(x).strip().strip('"') for x in acc_row.iloc[0]["values"]]

    per_sample: dict[str, dict[str, str]] = {g: {} for g in gsm_ids}

    char_rows = meta_df[
        meta_df["key"].str.lower().str.startswith("sample_characteristics")
    ]
    for _, row in char_rows.iterrows():
        vals = row["values"]
        for j, gsm in enumerate(gsm_ids):
            if j >= len(vals):
                break
            kv = _parse_kv(str(vals[j]))
            if kv is None:
                continue
            k, v = kv
            per_sample[gsm][k] = v

    meta = pd.DataFrame.from_dict(per_sample, orient="index")
    meta.index.name = "sample_id"

    # --- time-point parsing ---
    time_col = next(
        (c for c in ["time point", "time_point", "timepoint", "time", "sampling time"]
         if c in meta.columns),
        None,
    )
    if time_col:
        meta["time_hours"] = meta[time_col].apply(_parse_time_hours)
        meta["time_bin"] = meta["time_hours"].apply(_time_bin)
        meta["phase"] = meta["time_bin"].map(_PHASE_OF)
    else:
        warnings.warn(
            "No time-point column found in GSE182616 metadata. "
            "All samples will be treated as phase='unknown'.",
            UserWarning,
            stacklevel=2,
        )
        meta["time_hours"] = None
        meta["time_bin"] = None
        meta["phase"] = "unknown"

    n_acute = (meta["phase"] == "Acute").sum()
    print(f"Sample metadata: {len(meta)} samples; "
          f"{n_acute} Acute, "
          f"{(meta['phase'] == 'Proliferation').sum()} Proliferation, "
          f"{(meta['phase'] == 'Remodelling').sum()} Remodelling, "
          f"{meta['phase'].isna().sum()} unclassified")
    return meta


# ---------------------------------------------------------------------------
# Expression preprocessing
# ---------------------------------------------------------------------------

def preprocess_expression(
    expr_df: pd.DataFrame,
    anno_df: pd.DataFrame,
    *,
    winsor_q: float = 0.01,
    min_non_na_frac: float = 0.95,
    min_var: float = 1e-8,
) -> pd.DataFrame:
    """
    Merge series-matrix expression with annotation, collapse to gene level,
    and apply winsorisation + gene-wise z-score normalisation.

    Parameters
    ----------
    expr_df       : probes × samples (probe_id as index, numeric values)
    anno_df       : probe annotation with 'probe_id', 'gene_symbol',
                    and optionally 'control_type'
    winsor_q      : winsorisation percentile (both tails)
    min_non_na_frac : minimum fraction of non-NaN values per probe
    min_var       : minimum within-sample variance after collapse (removes flat genes)

    Returns
    -------
    genes × samples DataFrame of z-scored log2 intensities
    """
    # Merge annotation
    merged = expr_df.reset_index().merge(
        anno_df, on="probe_id", how="left",
    )
    sample_cols = [c for c in merged.columns
                   if c not in {"probe_id", "gene_symbol", "control_type"}]

    # Remove control probes
    if "control_type" in merged.columns:
        is_real = merged["control_type"].fillna("").isin({"", "0", "false", "f"})
        merged = merged[is_real]

    # Remove probes with no gene annotation
    merged["gene_symbol"] = merged["gene_symbol"].fillna("").str.strip()
    merged = merged[merged["gene_symbol"].str.len() > 0]

    # Remove probes with too many missing values
    non_na = merged[sample_cols].notna().mean(axis=1)
    merged = merged[non_na >= min_non_na_frac]
    if merged.empty:
        raise RuntimeError(
            "All probes removed after missingness filter "
            f"(min_non_na_frac={min_non_na_frac}). "
            "Check that the annotation file matches this series matrix."
        )

    # Numeric coercion
    for c in sample_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Collapse to gene level: keep highest-variance probe per gene
    merged["_var"] = merged[sample_cols].var(axis=1, ddof=1)
    idx = merged.groupby("gene_symbol")["_var"].idxmax()
    gene_mat = merged.loc[idx].set_index("gene_symbol")[sample_cols]

    # Remove zero/near-zero variance genes
    v = gene_mat.var(axis=1, ddof=1).fillna(0.0)
    gene_mat = gene_mat.loc[v > min_var]

    # Winsorise per gene
    lo = gene_mat.quantile(winsor_q, axis=1)
    hi = gene_mat.quantile(1 - winsor_q, axis=1)
    gene_mat = gene_mat.apply(lambda row: row.clip(lo[row.name], hi[row.name]), axis=1)

    # Gene-wise z-score
    mu = gene_mat.mean(axis=1)
    sd = gene_mat.std(axis=1, ddof=1).replace(0.0, np.nan)
    gene_mat_z = gene_mat.sub(mu, axis=0).div(sd, axis=0).dropna(axis=0)

    print(f"Preprocessed: {gene_mat_z.shape[0]} genes × {gene_mat_z.shape[1]} samples")
    return gene_mat_z


# ---------------------------------------------------------------------------
# Phase selection
# ---------------------------------------------------------------------------

def select_phase(
    gene_mat: pd.DataFrame,
    meta: pd.DataFrame,
    phase: str,
) -> pd.DataFrame:
    """
    Select samples belonging to the given phase.

    Parameters
    ----------
    phase : 'Acute' | 'Proliferation' | 'Remodelling' | 'all'
    """
    if phase == "all":
        selected = meta.index.tolist()
    else:
        selected = meta.index[meta["phase"] == phase].tolist()

    # Intersect with available expression columns
    available = [s for s in selected if s in gene_mat.columns]
    missing = len(selected) - len(available)
    if missing > 0:
        warnings.warn(
            f"{missing} {phase}-phase samples not found in expression matrix "
            "(may have been removed during quality filtering).",
            UserWarning,
            stacklevel=2,
        )
    if not available:
        raise RuntimeError(
            f"No {phase}-phase samples found. "
            f"Metadata phases: {meta['phase'].value_counts().to_dict()}"
        )

    sub = gene_mat[available]
    print(f"Phase '{phase}': {len(available)} samples selected")
    return sub


# ---------------------------------------------------------------------------
# Prior gene intersection
# ---------------------------------------------------------------------------

def intersect_prior_genes(
    gene_mat: pd.DataFrame,
    prior_genes: list[str],
) -> pd.DataFrame:
    overlap = [g for g in prior_genes if g in gene_mat.index]
    n_missing = len(prior_genes) - len(overlap)
    if n_missing > 0:
        warnings.warn(
            f"{n_missing}/{len(prior_genes)} prior genes not found in GSE182616 "
            f"(expected if prior was built on a broader gene space).",
            UserWarning,
            stacklevel=2,
        )
    sub = gene_mat.loc[overlap]
    print(f"Prior gene intersection: {len(overlap)}/{len(prior_genes)} genes retained")
    return sub


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_outputs(
    gene_mat: pd.DataFrame,
    meta: pd.DataFrame,
    out_dir: pathlib.Path,
    phase: str,
) -> None:
    """
    Save:
      expression.tsv      — genes × samples TSV (for piglasso run --data)
      genes.txt           — one HGNC symbol per line
      sample_metadata.csv — full sample metadata for audit
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_path  = out_dir / "expression.tsv"
    gene_path = out_dir / "genes.txt"
    meta_path = out_dir / "sample_metadata.csv"

    gene_mat.to_csv(tsv_path, sep="\t")
    gene_path.write_text("\n".join(gene_mat.index.tolist()))
    meta.to_csv(meta_path)

    print(
        f"\nOutputs written to {out_dir}:\n"
        f"  expression.tsv      — {gene_mat.shape[0]} genes × {gene_mat.shape[1]} samples "
        f"(phase: {phase})\n"
        f"  genes.txt           — {len(gene_mat.index)} gene symbols\n"
        f"  sample_metadata.csv — {len(meta)} samples total\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_gse182616(
    out_dir: pathlib.Path,
    series_matrix: pathlib.Path | None = None,
    annotation: pathlib.Path | None = None,
    phase: str = "Acute",
    prior_genes_path: pathlib.Path | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Obtain files
    if series_matrix is None:
        series_matrix = download_series_matrix(out_dir)
    if annotation is None:
        annotation = download_annotation(out_dir)

    # 2. Parse
    expr_df, meta_df = parse_series_matrix(series_matrix)
    anno_df = parse_annotation(annotation)

    # 3. Sample metadata
    meta = extract_sample_metadata(meta_df)

    # 4. Preprocess expression
    gene_mat = preprocess_expression(expr_df, anno_df)

    # 5. Select phase
    gene_mat_phase = select_phase(gene_mat, meta, phase)

    # 6. Intersect with prior genes (optional)
    if prior_genes_path is not None:
        prior_genes = [
            line.strip()
            for line in prior_genes_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"Prior gene list: {len(prior_genes)} genes from {prior_genes_path}")
        gene_mat_phase = intersect_prior_genes(gene_mat_phase, prior_genes)

    # 7. Save
    save_outputs(gene_mat_phase, meta, out_dir, phase)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Download and preprocess {GEO_ACCESSION} (Agilent {GPL_ACCESSION}) "
            "burn expression data for PIGLasso inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir", required=True, type=pathlib.Path,
        help="Output directory. expression.tsv and genes.txt are written here.",
    )
    parser.add_argument(
        "--series-matrix", type=pathlib.Path, default=None,
        help="Path to already-downloaded series matrix (.txt or .txt.gz). "
             "If omitted, downloads from NCBI GEO FTP.",
    )
    parser.add_argument(
        "--annotation", type=pathlib.Path, default=None,
        help=f"Path to {GPL_ACCESSION} annotation file (.annot.gz or .txt). "
             "If omitted, downloads from NCBI GEO FTP.",
    )
    parser.add_argument(
        "--phase",
        choices=["Acute", "Proliferation", "Remodelling", "all"],
        default="Acute",
        help="Which wound-healing phase to include. "
             "'Acute' = T0 + Early + Mid (hr0 through hr108). "
             "'all' = all timepoints.",
    )
    parser.add_argument(
        "--prior-genes", type=pathlib.Path, default=None,
        help="Path to prior gene list (one HGNC symbol per line). "
             "Expression is subset to genes present in this list. "
             "Use pipeline_src/prior/genes.txt to match the PIGLasso prior.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare_gse182616(
        out_dir=args.out_dir,
        series_matrix=args.series_matrix,
        annotation=args.annotation,
        phase=args.phase,
        prior_genes_path=args.prior_genes,
    )
