# PIGLasso — Prior-Informed Graphical Lasso

**Stability-selection GGM inference with biological prior integration for transcriptomics**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![Version](https://img.shields.io/badge/version-0.1.0-green)](pyproject.toml)

---

## Overview

PIGLasso is a Python package for prior-informed gene co-expression network inference from transcriptomics data. It implements stability-selection graphical Lasso (Meinshausen & Bühlmann 2010) with per-edge penalty modulation driven by biological prior knowledge (STRING protein–protein interactions, KEGG pathway co-membership), enabling the recovery of biologically plausible network structure even in high-dimensional, low-sample-size settings.

PIGLasso was developed for the analysis of burn injury transcriptomics (GSE182616) at Amsterdam UMC and is designed to complement [NODIS](https://github.com/youwillfindinfinity/nodis), which provides the statistical inference layer for validating edge significance.

---

## Features

- **Stability-selection GGM** — subsampling-based edge selection with per-edge stability scores (Meinshausen & Bühlmann 2010) using GGLasso `ADMM_SGL` as the base solver
- **Biological prior integration** — STRING PPI and KEGG pathway priors reduce regularisation on biologically credible edges via per-edge penalty masks: `λ_ij = λ · (1 − α · P_ij)`
- **Prior construction pipeline** — automated build chain for STRING (API) + KEGG (REST) priors with configurable mixture weights
- **Network diffusion** — heat-kernel diffusion over the inferred GGM to identify transcriptional hubs and propagate perturbation signals
- **Node knockouts** — in silico node removal to assess network robustness and hub essentiality
- **Multi-dataset support** — processes burn (GSE182616) and control (GSE236713) datasets on matched Agilent GPL17077 arrays, eliminating cross-platform batch effects
- **CLI** — `piglasso run / prior / diffuse / knockout` via Click
- **HPC-ready** — SLURM array jobs for Snellius (SURF) with rsync sync utilities

---

## Installation

```bash
pip install piglasso
```

NODIS is a required dependency (provides `PIGLassoEstimator`, `SklearnGLasso`, and benchmark utilities):

```bash
pip install "piglasso[dev]"   # includes pytest
```

**From source:**
```bash
git clone https://github.com/youwillfindinfinity/piglasso
cd piglasso
pip install -e ".[dev]"
```

**From `requirements.txt`:**
```bash
# Install NODIS first (required dependency)
pip install -e ../NODIS
pip install -r requirements.txt
pip install -e .
```

### Virtual environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
# Install NODIS (required dependency) then PIGLasso
pip install -e ../NODIS
pip install -e ".[dev]"
```

### HPC (Snellius) setup

```bash
cp .env.template .env
# Edit .env with your Snellius username and password
source .env

# Sync code to Snellius (prior_piglasso.npy is ~1.5 GB — synced separately)
rsync -avz --exclude='.venv/' --exclude='data/' \
    . ${SNELLIUS_USER}@${SNELLIUS_HOST}:${SNELLIUS_REMOTE_DIR}/

# Sync prior matrix and preprocessed data
rsync -avz pipeline_src/prior/ \
    ${SNELLIUS_USER}@${SNELLIUS_HOST}:${SNELLIUS_REMOTE_DIR}/pipeline_src/prior/
rsync -avz data/ \
    ${SNELLIUS_USER}@${SNELLIUS_HOST}:${SNELLIUS_REMOTE_DIR}/data/

# Install on Snellius (reuse NODIS venv or create dedicated one)
ssh ${SNELLIUS_USER}@${SNELLIUS_HOST} \
    "source ~/NODIS/.venv/bin/activate && pip install -e ${SNELLIUS_REMOTE_DIR}"
```

---

## Quick Start

### Fit with prior

```python
import numpy as np
from nodis.estimators.piglasso import PIGLassoEstimator
from nodis.estimators.prior_utils import build_corr_prior

# Expression matrix: (n_samples, n_genes), already normalised
X = ...   # e.g. log2-intensity, top-p genes by variance

# Data-derived correlation prior (no ground-truth leak)
prior = build_corr_prior(X, gamma=2.0)

# Fit PIGLasso with prior
est = PIGLassoEstimator(
    Q=50,
    b_perc=0.65,
    n_lambda=20,
    lambda_lo=0.05,
    lambda_hi=0.30,
    pi_thr=0.5,
    prior_weight=0.5,   # α: 0 = no prior effect, 1 = full prior
    n_jobs=4,
)
est.fit(X, prior=prior)

adj    = est.get_adjacency()   # binary (p, p) adjacency
scores = est.precision_        # stability scores for edge ranking
```

### Build biological prior from STRING + KEGG

```bash
cd pipeline_src/

# Step 1: extract gene list from expression data
python ../methodtotest/build_prior.py --step 1

# Step 2a: STRING PPI prior (queries string-db.org API)
python ../methodtotest/build_prior.py --step 2a

# Step 2b: KEGG pathway co-membership prior
python ../methodtotest/build_prior.py --step 2b

# Step 3: combine (60% STRING + 40% KEGG)
python ../methodtotest/build_prior.py --step 3

# Step 4: validate
python ../methodtotest/build_prior.py --step 4
```

Expected output: `prior/prior_piglasso.npy` — a symmetric (p × p) matrix with values in [0, 1], zero diagonal, and density ≈ 0.01–0.05.

### CLI

```bash
# Run inference on burn expression data with prior
piglasso run \
    --expr    data/burn/expression.npy \
    --genes   data/burn/genes.txt \
    --prior   pipeline_src/prior/prior_piglasso.npy \
    --out     results/network/

# Network diffusion (identify transcriptional hubs)
piglasso diffuse \
    --network results/network/adjacency.npy \
    --signal  data/burn/delta_expression.npy \
    --out     results/diffusion/

# Node knockout analysis
piglasso knockout \
    --network results/network/adjacency.npy \
    --nodes   results/diffusion/top_hubs.txt \
    --out     results/knockouts/
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
│   └── cli.py                     ← Click CLI (run/prior/diffuse/knockout)
├── pipeline_src/
│   ├── inference/
│   │   ├── network_inference.py   ← main GGM inference wrapper
│   │   └── piglasso_core.py       ← core stability-selection loop
│   ├── diffusion/
│   │   ├── network_diffusion.py   ← heat-kernel diffusion
│   │   └── diffusion_signal.py    ← perturbation signal construction
│   ├── knockouts/
│   │   └── node_knockout.py       ← in silico node removal
│   ├── preprocessing/
│   │   ├── burn/                  ← GSE182616 preprocessing
│   │   └── burn_control/          ← GSE236713 preprocessing
│   └── prior/
│       ├── genes.txt              ← gene list (19,923 human genes)
│       └── prior_piglasso.npy     ← combined STRING+KEGG prior (1.5 GB)
├── scripts/
│   ├── prepare_data.py            ← burn dataset preprocessing
│   └── prepare_gse236713.py       ← GSE236713 healthy/SIRS preprocessing
├── jobs/
│   ├── prior_step{1-4}.job        ← SLURM prior build chain
│   ├── piglasso_synthetic.job     ← SLURM synthetic benchmark
│   └── piglasso_diffusion.job     ← SLURM diffusion pipeline
└── tests/
```

---

## Datasets

| Dataset | Accession | Platform | n | Role |
|---------|-----------|----------|---|------|
| Burn injury | GSE182616 | Agilent GPL17077 | — | Primary analysis |
| Healthy + SIRS | GSE236713 | Agilent GPL17077 | 30 + 417 | Diffusion baseline, δ reference |

Both datasets use the same Agilent GPL17077 array, eliminating cross-platform normalisation issues.

---

## Testing

```bash
pytest tests/ -v
```

All 23 tests pass on the current release.

---

## Relationship to NODIS

PIGLasso and NODIS are companion tools developed in parallel:

- **NODIS** provides the statistical inference layer: edge-level p-values, FDR control, and confidence intervals for any GGM (including those estimated by PIGLasso).
- **PIGLasso** provides the prior-informed estimation layer: biologically guided network structure recovery under high dimensionality.

The recommended workflow is to run PIGLasso for network estimation and pass the resulting adjacency or stability scores to NODIS for formal statistical testing.

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
| **Zoe Azra Blei** |  Developer, Co-First author & Burns case study application | Amsterdam UMC, University of Amsterdam |

**Corresponding author:** Roland V. Bumbuc — rbumbuc@gmail.com

---

## Licence

MIT © 2026 Roland V. Bumbuc, Marcello Baryllii,  Zoe Azra Blei,
