#!/usr/bin/env Rscript
# annotate_genes.R
# ----------------
# Annotates all genes in the ranked knockout list using biomaRt.
# For each gene: retrieves description, GO biological process terms,
# and molecular function terms. Enriches the ranked_genes_with_modules.csv
# with these annotations and flags previously uncharacterised genes.
#
# Input:  results/knockouts/ranked_genes_with_modules.csv
# Output: pipeline_src/pathways/results/GSE182616/PIGLasso/gene_annotations.csv

suppressPackageStartupMessages({
  library(biomaRt)
  library(dplyr)
  library(tidyr)
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  f    <- sub("--file=", "", args[grepl("--file=", args)])
  dirname(normalizePath(f))
}, error = function(e) getwd())
ROOT     <- normalizePath(file.path(HERE, "..", ".."), mustWork = FALSE)
IN_FILE  <- file.path(ROOT, "results", "knockouts", "ranked_genes_with_modules.csv")
OUT_FILE <- file.path(HERE, "results", "GSE182616", "PIGLasso", "gene_annotations.csv")

# ---------------------------------------------------------------------------
# Load ranked gene list
# ---------------------------------------------------------------------------
ranked <- read.csv(IN_FILE, stringsAsFactors = FALSE)
genes  <- ranked$gene
message("[INFO] Genes to annotate: ", length(genes))

# ---------------------------------------------------------------------------
# Connect to Ensembl via biomaRt
# ---------------------------------------------------------------------------
message("[INFO] Connecting to Ensembl BioMart ...")
mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")

# ---------------------------------------------------------------------------
# Query 1: gene descriptions
# ---------------------------------------------------------------------------
message("[INFO] Fetching gene descriptions ...")
desc <- getBM(
  attributes = c("hgnc_symbol", "description", "gene_biotype",
                 "chromosome_name", "start_position", "end_position"),
  filters    = "hgnc_symbol",
  values     = genes,
  mart       = mart
)
desc <- desc[!duplicated(desc$hgnc_symbol), ]
desc$description <- sub("\\s*\\[.*\\]$", "", desc$description)  # remove source tag

# ---------------------------------------------------------------------------
# Query 2: GO Biological Process
# ---------------------------------------------------------------------------
message("[INFO] Fetching GO Biological Process terms ...")
go_bp <- getBM(
  attributes = c("hgnc_symbol", "go_id", "name_1006", "namespace_1003"),
  filters    = "hgnc_symbol",
  values     = genes,
  mart       = mart
)
go_bp <- go_bp[go_bp$namespace_1003 == "biological_process", ]
go_bp_collapsed <- go_bp %>%
  group_by(hgnc_symbol) %>%
  summarise(
    go_bp_ids   = paste(unique(go_id),    collapse = "; "),
    go_bp_terms = paste(unique(name_1006), collapse = "; "),
    n_go_bp     = n_distinct(go_id),
    .groups = "drop"
  )

# ---------------------------------------------------------------------------
# Query 3: GO Molecular Function
# ---------------------------------------------------------------------------
message("[INFO] Fetching GO Molecular Function terms ...")
go_mf <- getBM(
  attributes = c("hgnc_symbol", "go_id", "name_1006", "namespace_1003"),
  filters    = "hgnc_symbol",
  values     = genes,
  mart       = mart
)
go_mf <- go_mf[go_mf$namespace_1003 == "molecular_function", ]
go_mf_collapsed <- go_mf %>%
  group_by(hgnc_symbol) %>%
  summarise(
    go_mf_ids   = paste(unique(go_id),    collapse = "; "),
    go_mf_terms = paste(unique(name_1006), collapse = "; "),
    n_go_mf     = n_distinct(go_id),
    .groups = "drop"
  )

# ---------------------------------------------------------------------------
# Merge everything onto ranked list
# ---------------------------------------------------------------------------
message("[INFO] Merging annotations ...")
annotated <- ranked %>%
  left_join(desc,             by = c("gene" = "hgnc_symbol")) %>%
  left_join(go_bp_collapsed,  by = c("gene" = "hgnc_symbol")) %>%
  left_join(go_mf_collapsed,  by = c("gene" = "hgnc_symbol")) %>%
  arrange(rank)

# Flag genes that had no annotation before (module == "uncharacterised")
# but now have GO terms
annotated <- annotated %>%
  mutate(
    was_uncharacterised = module == "uncharacterised",
    now_annotated       = was_uncharacterised & (!is.na(go_bp_terms) | !is.na(go_mf_terms))
  )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
dir.create(dirname(OUT_FILE), recursive = TRUE, showWarnings = FALSE)
write.csv(annotated, OUT_FILE, row.names = FALSE)
message("[SAVED] ", OUT_FILE)

# Summary
message("\n[SUMMARY]")
message("  Total genes:           ", nrow(annotated))
message("  Previously uncharact.: ", sum(annotated$was_uncharacterised, na.rm = TRUE))
message("  Now have GO BP terms:  ", sum(!is.na(annotated$go_bp_terms)))
message("  Now have GO MF terms:  ", sum(!is.na(annotated$go_mf_terms)))
message("  Still unannotated:     ",
        sum(annotated$was_uncharacterised &
            is.na(annotated$go_bp_terms) &
            is.na(annotated$go_mf_terms), na.rm = TRUE))
