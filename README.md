# PIGLasso — Prior-Informed Graphical Lasso

**Stability-selection GGM inference with biological prior integration for burn injury transcriptomics**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-0.1.0-green)](pyproject.toml)

---

## Overview

PIGLasso is a burn-injury transcriptomics pipeline built on top of [NODIS](https://github.com/youwillfindinfinity/nodis). It implements stability-selection graphical Lasso (Meinshausen & Bühlmann 2010) with per-edge penalty modulation driven by biological prior knowledge (STRING protein–protein interactions, KEGG pathway co-membership), enabling recovery of biologically plausible network structure in high-dimensional, low-sample-size settings.

**PIGLasso does not perform statistical inference.** Edge-level p-values, FDR-controlled adjacency matrices, and confidence intervals are provided by NODIS, which sits above PIGLasso in the stack. `PIGLassoEstimator` itself lives in NODIS (`nodis.estimators.piglasso`) — this repository is the domain-specific pipeline that applies it to burn injury data (GSE182616 / GSE236713) and extends it with biological prior construction, network diffusion, node knockouts, pathway annotation, and GSEA.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  NODIS                                          │  ← statistical inference layer
│  nodis.estimators.piglasso.PIGLassoEstimator    │     p-values, FDR, CIs, benchmarks
│  nodis.estimators.glasso.SklearnGLasso          │
└────────────────┬────────────────────────────────┘
                 │  depends on
┌────────────────▼────────────────────────────────┐
│  PIGLasso (this repo)                           │  ← domain pipeline
│  Prior construction (STRING + KEGG)             │     burn injury application
│  Network diffusion  ·  Node knockouts           │
│  Pathway annotation  ·  GSEA                    │
│  Burn-specific preprocessing (GSE182616/236713) │
└─────────────────────────────────────────────────┘
```

---

## Features

- **Stability-selection GGM** — subsampling-based edge selection with per-edge stability scores (Meinshausen & Bühlmann 2010) via `PIGLassoEstimator` from NODIS using GGLasso `ADMM_SGL` as the base solver
- **Biological prior integration** — STRING PPI and KEGG pathway priors reduce regularisation on biologically credible edges: `λ_ij = λ · (1 − α · P_ij)`
- **Prior construction pipeline** — automated build chain for STRING (API) + KEGG (REST) priors with configurable mixture weights (steps 1 → 2a/2b/2c → 3 → 4)
- **Network diffusion** — heat-kernel diffusion over the inferred GGM to identify transcriptional hubs and propagate perturbation signals
- **Node knockouts** — in silico node removal to assess network robustness and hub essentiality
- **Knockout–pathway cross-reference** — links knocked-out hub genes to enriched pathways for biological interpretation
- **Pathway annotation & GSEA** — gene-level annotation (biotype, GO) and gene-set enrichment on stability scores and acute-phase delta vectors
- **Burn-specific preprocessing** — GSE182616 (burn) and GSE236713 (healthy + SIRS) on matched Agilent GPL17077 arrays, eliminating cross-platform batch effects
- **CLI** — `piglasso run / prior / diffuse / knockout` via Click
- **HPC-ready** — SLURM array jobs for Snellius (SURF) with a full prior-build submission chain

---

## Installation

PIGLasso requires NODIS. The `PIGLassoEstimator` class and all GGM inference utilities live in NODIS — PIGLasso is the application layer on top.

```bash
pip install nodis        # required: provides PIGLassoEstimator, SklearnGLasso, inference
pip install piglasso
```

```bash
pip install "piglasso[dev]"   # includes pytest
```

**From source:**
```bash
# Install NODIS first — PIGLasso depends on it for the core estimator
git clone https://github.com/youwillfindinfinity/nodis
pip install -e ./nodis

git clone https://github.com/youwillfindinfinity/piglasso
cd piglasso
pip install -e ".[dev]"
```

**From `requirements.txt`:**
```bash
pip install -e ../NODIS      # NODIS must be installed before PIGLasso
pip install -r requirements.txt
pip install -e .
```

**R dependencies** (pathways and GSEA modules):
```bash
pip install "piglasso[r]"    # installs rpy2
# In R: install.packages(c("clusterProfiler", "org.Hs.eg.db", "fgsea", "ggplot2"))
```

### Virtual environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ../NODIS            # NODIS first
pip install -e ".[dev]"
```

### HPC (Snellius) setup

```bash
cp .env.template .env
# Edit .env with your Snellius username and password
source .env

# Sync code to Snellius
rsync -avz --exclude='.venv/' --exclude='data/' \
    . ${SNELLIUS_USER}@${SNELLIUS_HOST}:${SNELLIUS_REMOTE_DIR}/

# Sync prior matrix and preprocessed data (prior_piglasso.npy is ~1.5 GB)
rsync -avz pipeline_src/prior/ \
    ${SNELLIUS_USER}@${SNELLIUS_HOST}:${SNELLIUS_REMOTE_DIR}/pipeline_src/prior/
rsync -avz data/ \
    ${SNELLIUS_USER}@${SNELLIUS_HOST}:${SNELLIUS_REMOTE_DIR}/data/

# Install on Snellius (activate NODIS venv which already has NODIS installed)
ssh ${SNELLIUS_USER}@${SNELLIUS_HOST} \
    "source ~/NODIS/.venv/bin/activate && pip install -e ${SNELLIUS_REMOTE_DIR}"
```

---

## Quick Start

### Fit with prior

`PIGLassoEstimator` is part of NODIS. Import it from there:

```python
import numpy as np
from nodis.estimators.piglasso import PIGLassoEstimator
from nodis.estimators.prior_utils import build_corr_prior

# Expression matrix: (n_samples, n_genes), already normalised
X = ...   # e.g. log2-intensity, top-5000 genes by variance

# Data-derived correlation prior (no ground-truth leak)
prior = build_corr_prior(X, gamma=2.0)

# Fit PIGLasso with prior
est = PIGLassoEstimator(
    Q=50,
    b_perc=0.65,
    n_lambda=20,
    lambda_lo=0.05,
    lambda_hi=0.30,
    pi_thr=0.6,
    prior_weight=0.5,   # α: 0 = no prior effect, 1 = full prior
    n_jobs=4,
)
est.fit(X, prior=prior)

adj    = est.get_adjacency()   # binary (p, p) adjacency
scores = est.precision_        # stability scores for edge ranking
```

To add statistical inference (p-values, FDR, confidence intervals) on top of the estimated network, use NODIS directly — that is outside the scope of this pipeline.

### Build biological prior from STRING + KEGG

```bash
# Step 1: extract gene list from expression data
piglasso prior --step 1

# Step 2a: STRING PPI prior (queries string-db.org API)
piglasso prior --step 2a

# Step 2b: KEGG pathway co-membership prior
piglasso prior --step 2b

# Step 2c: supplementary prior refinement
piglasso prior --step 2c

# Step 3: combine (60% STRING + 40% KEGG)
piglasso prior --step 3

# Step 4: validate
piglasso prior --step 4
```

On HPC, submit the full chain as a SLURM dependency chain:
```bash
bash jobs/submit_prior_chain.sh
```

Expected output: `pipeline_src/prior/prior_piglasso.npy` — a symmetric (p × p) matrix with values in [0, 1], zero diagonal, and density ≈ 0.01–0.05.

### CLI

```bash
# Run inference on burn expression data with prior
piglasso run \
    --data    data/burn/expression.tsv \
    --prior   pipeline_src/prior/prior_piglasso.npy \
    --out     results/network/

# Network diffusion (identify transcriptional hubs)
piglasso diffuse \
    --in_dir  data/burn/diffusion_inputs \
    --out_dir results/diffusion/

# Node knockout analysis
piglasso knockout \
    --in_dir    data/burn/diffusion_inputs \
    --reduction 0.3 \
    --out_dir   results/knockouts/
```

---

## Prior Weighting

The per-edge penalty mask is:

```
λ_ij = λ · (1 − α · P_ij)
```

where P_ij ∈ [0, 1] is the prior belief in edge (i, j) and α = `prior_weight`.

| P_ij | α = 0.5 | Effect |
|------|---------|--------|
| 0.0 | λ · 1.0 | Standard regularisation — no prior effect |
| 0.5 | λ · 0.75 | Moderate reduction — edge moderately favoured |
| 1.0 | λ · 0.5 | Halved regularisation — edge strongly favoured |

The floor is clipped at 0.1·λ to prevent the penalty collapsing to zero on any edge.

---

## Pipeline Structure

```
PIGLasso/
├── piglasso/
│   └── cli.py                              ← Click CLI (run/prior/diffuse/knockout)
├── pipeline_src/
│   ├── build_prior.py                      ← prior construction driver (steps 1–4)
│   ├── filter_top_genes.py                 ← subset to top-5000 genes by variance
│   ├── hub_analysis_burns.py               ← hub gene analysis for burn data
│   ├── inference/
│   │   ├── network_inference.py            ← main GGM inference wrapper
│   │   ├── piglasso_core.py                ← core stability-selection loop
│   │   ├── plot_piglasso_results.py        ← results plots
│   │   └── run_piglasso_new.py             ← inference runner script
│   ├── diffusion/
│   │   ├── network_diffusion.py            ← heat-kernel diffusion
│   │   └── diffusion_signal.py             ← perturbation signal construction
│   ├── knockouts/
│   │   ├── node_knockout.py                ← in silico node removal
│   │   ├── plot_knockouts.py               ← knockout result plots
│   │   └── crossref/
│   │       ├── crossref_knockouts_pathways.R  ← knockout–pathway cross-reference
│   │       └── plot_crossref.R             ← cross-reference visualisation
│   ├── pathways/
│   │   ├── annotate_genes.R                ← gene annotation (biotype, GO terms)
│   │   ├── run_gsea.R                      ← GSEA on stable-network gene scores
│   │   └── run_gsea_delta.R                ← GSEA on acute-phase delta vector (T0+Early+Mid)
│   ├── preprocessing/
│   │   ├── burn/                           ← GSE182616 preprocessing notebooks
│   │   └── burn_control/                   ← GSE236713 preprocessing notebooks
│   └── prior/
│       ├── genes.txt                       ← gene list (top-5000 by variance)
│       └── prior_piglasso.npy              ← combined STRING+KEGG prior (~1.5 GB)
├── scripts/
│   ├── prepare_data.py                     ← burn dataset preprocessing
│   ├── prepare_gse182616.py                ← GSE182616-specific normalisation
│   └── prepare_gse236713.py                ← GSE236713 healthy/SIRS preprocessing
├── jobs/
│   ├── submit_prior_chain.sh               ← submit full prior build as SLURM chain
│   ├── prior_step1.job                     ← SLURM: gene list extraction
│   ├── prior_step2a.job                    ← SLURM: STRING PPI prior
│   ├── prior_step2b.job                    ← SLURM: KEGG prior
│   ├── prior_step2c.job                    ← SLURM: prior refinement
│   ├── prior_step3.job                     ← SLURM: combine priors
│   ├── prior_step4.job                     ← SLURM: validate prior
│   ├── piglasso_burns.job                  ← SLURM: burn inference
│   ├── piglasso_diffusion.job              ← SLURM: diffusion pipeline
│   ├── piglasso_dream5.job                 ← SLURM: DREAM5 benchmark
│   ├── piglasso_synth_n513p164.job         ← SLURM: synthetic benchmark (n=513, p=164)
│   └── piglasso_synthetic.job              ← SLURM: general synthetic benchmark
└── tests/
```

---

## Datasets

| Dataset | Accession | Platform | n | Role |
|---------|-----------|----------|---|------|
| Burn injury | GSE182616 | Agilent GPL17077 | — | Primary analysis |
| Healthy + SIRS | GSE236713 | Agilent GPL17077 | 30 + 417 | Diffusion baseline, δ reference |

Both datasets use the same Agilent GPL17077 array, eliminating cross-platform normalisation issues. Gene expression matrices are subset to the top-5000 genes by variance before inference (`filter_top_genes.py`).

---

## Pathway Analysis

The `pipeline_src/pathways/` module provides post-inference biological interpretation:

- **`annotate_genes.R`** — maps network hub genes to biotype and GO terms
- **`run_gsea.R`** — gene-set enrichment on stability scores from the full inferred network
- **`run_gsea_delta.R`** — GSEA on the acute-phase delta expression vector (T0 + Early + Mid timepoints, 0–84 h), identifying pathway activation in the acute burn response

Knockout–pathway cross-reference (`knockouts/crossref/`) links hub genes removed in the knockout analysis to their enriched pathways, providing a combined structural + functional interpretation of network hubs.

---

## Testing

```bash
pytest tests/ -v
```

The suite contains 23 tests. 6 pass without NODIS (import and CLI smoke tests). All 23 pass once NODIS is installed. Tests that require NODIS are automatically skipped — not failed — when NODIS is absent, so `pytest` is always safe to run in a partial environment.

---

## Relationship to NODIS

NODIS and PIGLasso are distinct tools with a strict dependency direction: **NODIS is above PIGLasso in the stack.**

| Layer | Tool | Responsibility |
|-------|------|----------------|
| Inference | **NODIS** | `PIGLassoEstimator`, `SklearnGLasso`, de-sparsified nodewise Lasso, edge p-values, FDR control, confidence intervals, benchmarking framework |
| Pipeline | **PIGLasso** (this repo) | Biological prior construction, burn-specific preprocessing, stability-selection workflow, network diffusion, node knockouts, pathway annotation, GSEA |

`PIGLassoEstimator` is developed and maintained inside NODIS. PIGLasso borrows it via the `nodis` dependency and builds the full burn-injury analysis pipeline around it. PIGLasso does not reimplement, wrap, or shadow any NODIS inference functionality.

The recommended workflow:

```
PIGLasso pipeline  →  adjacency / stability scores  →  NODIS inference
(network structure)                                    (p-values, FDR, CIs)
```

---

## Citation

If you use PIGLasso in your research, please cite:

> **Bumbuc RV, Blei ZA, Baryllii M** (2026). *PIGLasso: Prior-Informed Graphical Lasso for transcriptomics network inference.* *(manuscript in preparation)*

---

## Authors

| Name | Role | Affiliation |
|------|------|-------------|
| **Roland V. Bumbuc** | Co-developer, First and corresponding author | Amsterdam UMC, University of Amsterdam |
| **Marcello Baryllii** | Co-developer | Amsterdam UMC, University of Amsterdam |
| **Zoe Azra Blei** | Developer, Co-First author & Burns case study application | Amsterdam UMC, University of Amsterdam |

**Corresponding author:** Roland V. Bumbuc — rbumbuc@gmail.com

---

## Licence

MIT © 2026 Roland V. Bumbuc, Marcello Baryllii, Zoe Azra Blei
