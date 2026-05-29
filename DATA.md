# Data Availability

This document describes all datasets used by PIGLasso, how to obtain them, and what is and is not included in the repository.

---

## Datasets

### GSE182616 — Burn injury (primary analysis)

| Field | Value |
|-------|-------|
| GEO accession | [GSE182616](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182616) |
| Platform | Agilent GPL17077 (Agilent-039494 SurePrint G3 Human GE v2 8×60K) |
| Organism | Homo sapiens |
| Tissue | Whole blood |
| Design | Longitudinal burn injury cohort; multiple post-burn timepoints (T0, Early, Mid, Late, Follow-up) |
| Role in PIGLasso | Primary expression matrix for GGM inference and network diffusion |
| Included in repo | **No** — raw/processed data are not committed (gitignored) |

**Download:**
```bash
# Using GEOquery (R)
Rscript -e "GEOquery::getGEO('GSE182616', destdir='data/burn/')"
```

Then preprocess:
```bash
python3 pipeline_src/preprocessing/preprocess_burn.py   # or open burn notebook
```

---

### GSE236713 — Healthy controls + SIRS (diffusion baseline)

| Field | Value |
|-------|-------|
| GEO accession | [GSE236713](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236713) |
| Platform | Agilent GPL17077 (same array as GSE182616 — no cross-platform normalisation needed) |
| Organism | Homo sapiens |
| Tissue | Whole blood |
| Design | Healthy volunteers (n ≈ 30) + SIRS patients (n ≈ 417) |
| Role in PIGLasso | Reference baseline for constructing the delta expression vector (δ = burn − healthy) used in network diffusion; also used for correlation-based prior construction |
| Included in repo | **No** — gitignored |

**Download:**
```bash
Rscript -e "GEOquery::getGEO('GSE236713', destdir='data/gse236713/')"
```

Then preprocess:
```bash
python3 scripts/prepare_gse236713.py
```

---

## Prior matrix

| File | Size | Description |
|------|------|-------------|
| `pipeline_src/prior/prior_piglasso.npy` | ~1.5 GB | Combined STRING + KEGG per-edge prior; symmetric (p × p), values in [0, 1] |

This file is gitignored (`.gitignore` excludes `*.npy`). To reproduce it:

```bash
piglasso prior --step 1
piglasso prior --step 2a
piglasso prior --step 2b
piglasso prior --step 2c
piglasso prior --step 3
piglasso prior --step 4
```

Or on HPC:
```bash
bash jobs/submit_prior_chain.sh
```

The prior requires an internet connection for the STRING API (step 2a) and KEGG REST API (step 2b).

---

## What is included in the repository

| Path | Included | Notes |
|------|----------|-------|
| `pipeline_src/prior/genes.txt` | Yes | Gene list (top-5000 by variance); derived from GSE182616 |
| `pipeline_src/prior/prior_piglasso.npy` | No | Too large (~1.5 GB); regenerate via prior pipeline |
| `data/` | No | All expression matrices gitignored |
| `results/` | No | All inference results gitignored |
| `preprocessing/` | No | Preprocessed data outputs gitignored |

---

## Licensing and access

Both GEO datasets (GSE182616, GSE236713) are publicly available under NCBI's standard data access terms. No registration is required to download them.

External databases used during prior construction:
- **STRING** — [string-db.org](https://string-db.org), CC BY 4.0
- **KEGG** — [kegg.jp](https://www.kegg.jp), academic use; see [KEGG licence](https://www.kegg.jp/kegg/legal.html)
