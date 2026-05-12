#!/usr/bin/env Rscript
# run_gsea.R
# ----------
# Two complementary pathway analyses on the ranked knockout gene list:
#
#   1. ORA  (enrichGO)  — over-representation of GO terms in top-ranked genes
#   2. GSEA (gseGO)     — enrichment across the full ranked list by impact score
#
# Both use GO Biological Process terms.
# Results saved as CSV + dot plots.
#
# Input:  results/knockouts/ranked_genes_with_modules.csv
# Output: pipeline_src/pathways/results/GSE182616/PIGLasso/
#           ora_results.csv, gsea_results.csv
#           figures/ora_dotplot.pdf/png, gsea_dotplot.pdf/png, gsea_ridgeplot.pdf/png

suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(ggplot2)
  library(dplyr)
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
EXPR_FILE <- normalizePath(file.path(ROOT, "..", "preprocessing", "data",
                                     "plotting", "preprocessed",
                                     "burn_gene_matrix.tsv"), mustWork = FALSE)
OUT_DIR  <- file.path(HERE, "results", "GSE182616", "PIGLasso")
FIG_DIR  <- file.path(OUT_DIR, "figures")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Load + build ranked vector
# ---------------------------------------------------------------------------
ranked <- read.csv(IN_FILE, stringsAsFactors = FALSE)

# Named numeric vector: gene symbol -> knockout impact score, sorted descending
gene_vec <- setNames(ranked$knockout_impact_score, ranked$gene)
gene_vec <- sort(gene_vec, decreasing = TRUE)
message("[INFO] Ranked genes: ", length(gene_vec))

# Top genes for ORA: top tercile by impact score
tercile_cut <- quantile(gene_vec, 0.67)
top_genes   <- names(gene_vec[gene_vec >= tercile_cut])
message("[INFO] Top genes for ORA (top tercile, n=", length(top_genes), ")")

# Background = genes with non-trivial variance in the acute phase samples only
message("[INFO] Loading expression matrix and metadata ...")
meta_file <- normalizePath(file.path(ROOT, "..", "preprocessing", "data",
                                     "plotting", "preprocessed",
                                     "burn_sample_metadata.tsv"), mustWork = FALSE)
meta     <- read.table(meta_file, sep = "\t", header = TRUE,
                       row.names = 1, stringsAsFactors = FALSE)
acute_samples <- rownames(meta)[meta$time_bin %in% c("T0", "Early", "Mid")]
message("[INFO] Acute phase samples: ", length(acute_samples))

expr_mat <- read.table(EXPR_FILE, sep = "\t", header = TRUE,
                       row.names = 1, check.names = FALSE)
acute_cols <- intersect(acute_samples, colnames(expr_mat))
expr_acute <- expr_mat[, acute_cols, drop = FALSE]

# Keep genes with variance > 0 in acute phase (removes flat/unexpressed probes)
gene_vars  <- apply(expr_acute, 1, var, na.rm = TRUE)
background <- rownames(expr_acute)[gene_vars > 0]
message("[INFO] Background genes (variable in acute phase): ", length(background))

# ---------------------------------------------------------------------------
# 1. ORA — over-representation analysis
# ---------------------------------------------------------------------------
message("\n[INFO] Running ORA (enrichGO) ...")
ora <- enrichGO(
  gene          = top_genes,
  universe      = background,
  OrgDb         = org.Hs.eg.db,
  keyType       = "SYMBOL",
  ont           = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.10,
  readable      = TRUE
)

if (!is.null(ora) && nrow(ora@result) > 0) {
  ora_df <- as.data.frame(ora)
  write.csv(ora_df, file.path(OUT_DIR, "ora_results.csv"), row.names = FALSE)
  message("[SAVED] ora_results.csv  (", nrow(ora_df), " terms)")

  p_ora <- dotplot(ora, showCategory = 20, font.size = 11) +
    ggtitle("ORA - GO Biological Process\n(top-ranked knockout genes)") +
    theme(plot.title = element_text(size = 13, face = "bold"))

  ggsave(file.path(FIG_DIR, "ora_dotplot.pdf"), p_ora, width = 9, height = 8)
  ggsave(file.path(FIG_DIR, "ora_dotplot.png"), p_ora, width = 9, height = 8, dpi = 200)
  message("[SAVED] ora_dotplot")
} else {
  message("[WARN] ORA: no significant terms found — try relaxing pvalueCutoff.")
}

# ---------------------------------------------------------------------------
# 2. GSEA — gene set enrichment analysis
# ---------------------------------------------------------------------------
# Check how many genes are recognised by org.Hs.eg.db
all_db_genes <- keys(org.Hs.eg.db, keytype = "SYMBOL")
n_matched <- sum(names(gene_vec) %in% all_db_genes)
message("[INFO] Genes matched in org.Hs.eg.db: ", n_matched, " / ", length(gene_vec))

message("\n[INFO] Running GSEA (gseGO) ...")
set.seed(42)
gsea <- gseGO(
  geneList      = gene_vec,
  OrgDb         = org.Hs.eg.db,
  keyType       = "SYMBOL",
  ont           = "BP",
  minGSSize     = 2,
  maxGSSize     = 500,
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.50,
  scoreType     = "pos",
  verbose       = FALSE,
  nPermSimple   = 10000
)

if (!is.null(gsea) && nrow(gsea@result) > 0) {
  gsea_df <- as.data.frame(gsea)
  write.csv(gsea_df, file.path(OUT_DIR, "gsea_results.csv"), row.names = FALSE)
  message("[SAVED] gsea_results.csv  (", nrow(gsea_df), " terms)")

  p_dot <- dotplot(gsea, showCategory = 20, split = ".sign", font.size = 11) +
    facet_grid(. ~ .sign) +
    ggtitle("GSEA - GO Biological Process") +
    theme(plot.title = element_text(size = 13, face = "bold"))

  ggsave(file.path(FIG_DIR, "gsea_dotplot.pdf"), p_dot, width = 12, height = 8)
  ggsave(file.path(FIG_DIR, "gsea_dotplot.png"), p_dot, width = 12, height = 8, dpi = 200)
  message("[SAVED] gsea_dotplot")

  if (requireNamespace("ggridges", quietly = TRUE)) {
    p_ridge <- ridgeplot(gsea, showCategory = 20, fill = "p.adjust") +
      ggtitle("GSEA - GO Biological Process (ridge plot)") +
      theme(plot.title = element_text(size = 13, face = "bold"),
            axis.text.y = element_text(size = 9))
    ggsave(file.path(FIG_DIR, "gsea_ridgeplot.pdf"), p_ridge, width = 10, height = 10)
    ggsave(file.path(FIG_DIR, "gsea_ridgeplot.png"), p_ridge, width = 10, height = 10, dpi = 200)
    message("[SAVED] gsea_ridgeplot")
  } else {
    message("[SKIP] gsea_ridgeplot — install ggridges to enable")
  }
} else {
  message("[WARN] GSEA: no significant terms found — try increasing pvalueCutoff or check gene list size.")
}

message("\n[DONE]")
