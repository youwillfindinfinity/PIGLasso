#!/usr/bin/env Rscript
# run_gsea_delta.R
# ----------------
# GSEA on the acute-phase differential signal from GSE182616.
#
# Ranking statistic: Welch t-statistic (T0+Early vs Late+FollowUp) per gene.
# Positive t = upregulated in acute phase; negative t = downregulated.
#
# Input:  preprocessing/data/plotting/preprocessed/burn_gene_matrix.tsv
#         preprocessing/data/plotting/preprocessed/burn_sample_metadata.tsv
# Output: pipeline_src/pathways/results/GSE182616/delta/
#           delta_ranked_genes.csv
#           gsea_delta_results.csv
#           figures/gsea_delta_dotplot.pdf/png

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
ROOT      <- normalizePath(file.path(HERE, "..", ".."), mustWork = FALSE)
EXPR_FILE <- normalizePath(file.path(ROOT, "..", "preprocessing", "data",
                                     "plotting", "preprocessed",
                                     "burn_gene_matrix.tsv"), mustWork = FALSE)
META_FILE <- normalizePath(file.path(ROOT, "..", "preprocessing", "data",
                                     "plotting", "preprocessed",
                                     "burn_sample_metadata.tsv"), mustWork = FALSE)
OUT_DIR   <- file.path(HERE, "results", "GSE182616", "delta")
FIG_DIR   <- file.path(OUT_DIR, "figures")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
message("[INFO] Loading expression matrix ...")
expr <- read.table(EXPR_FILE, sep = "\t", header = TRUE,
                   row.names = 1, check.names = FALSE)
message("[INFO] Genes: ", nrow(expr), "  Samples: ", ncol(expr))

meta <- read.table(META_FILE, sep = "\t", header = TRUE,
                   row.names = 1, stringsAsFactors = FALSE)

# ---------------------------------------------------------------------------
# Define groups
# ---------------------------------------------------------------------------
acute_samples    <- rownames(meta)[meta$time_bin %in% c("T0", "Early", "Mid")]
recovery_samples <- rownames(meta)[meta$time_bin %in% c("Late", "FollowUp")]

acute_cols    <- intersect(acute_samples,    colnames(expr))
recovery_cols <- intersect(recovery_samples, colnames(expr))

message("[INFO] Acute (T0+Early):        ", length(acute_cols), " samples")
message("[INFO] Recovery (Late+FollowUp): ", length(recovery_cols), " samples")

# ---------------------------------------------------------------------------
# Compute Welch t-statistic per gene (vectorised)
# ---------------------------------------------------------------------------
message("[INFO] Computing t-statistics ...")
expr_mat <- as.matrix(expr)

acute_mat    <- expr_mat[, acute_cols,    drop = FALSE]
recovery_mat <- expr_mat[, recovery_cols, drop = FALSE]

n1 <- ncol(acute_mat)
n2 <- ncol(recovery_mat)
m1 <- rowMeans(acute_mat,    na.rm = TRUE)
m2 <- rowMeans(recovery_mat, na.rm = TRUE)
v1 <- apply(acute_mat,    1, var, na.rm = TRUE)
v2 <- apply(recovery_mat, 1, var, na.rm = TRUE)

t_stats <- (m1 - m2) / sqrt(v1 / n1 + v2 / n2)
t_stats <- t_stats[is.finite(t_stats)]
t_stats <- sort(t_stats, decreasing = TRUE)

message("[INFO] Genes with valid t-statistics: ", length(t_stats))
message("[INFO] Range: [", round(min(t_stats), 2), ", ", round(max(t_stats), 2), "]")

# ---------------------------------------------------------------------------
# Save ranked gene list
# ---------------------------------------------------------------------------
ranked_df <- data.frame(gene        = names(t_stats),
                        t_statistic = as.numeric(t_stats),
                        row.names   = NULL)
write.csv(ranked_df, file.path(OUT_DIR, "delta_ranked_genes.csv"), row.names = FALSE)
message("[SAVED] delta_ranked_genes.csv")

# ---------------------------------------------------------------------------
# GSEA
# ---------------------------------------------------------------------------
n_db <- sum(names(t_stats) %in% keys(org.Hs.eg.db, keytype = "SYMBOL"))
message("[INFO] Genes matched in org.Hs.eg.db: ", n_db, " / ", length(t_stats))

message("[INFO] Running GSEA (gseGO) ...")
set.seed(42)
gsea <- gseGO(
  geneList      = t_stats,
  OrgDb         = org.Hs.eg.db,
  keyType       = "SYMBOL",
  ont           = "BP",
  minGSSize     = 10,
  maxGSSize     = 500,
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  scoreType     = "std",
  verbose       = FALSE,
  nPermSimple   = 10000
)

if (!is.null(gsea) && nrow(gsea@result) > 0) {
  gsea_df <- as.data.frame(gsea)
  write.csv(gsea_df, file.path(OUT_DIR, "gsea_delta_results.csv"), row.names = FALSE)
  message("[SAVED] gsea_delta_results.csv  (", nrow(gsea_df), " terms)")

  p_dot <- dotplot(gsea, showCategory = 20, split = ".sign", font.size = 11) +
    facet_grid(. ~ .sign) +
    theme(plot.title = element_text(size = 13, face = "bold"))

  ggsave(file.path(FIG_DIR, "gsea_delta_dotplot.pdf"), p_dot, width = 12, height = 8)
  ggsave(file.path(FIG_DIR, "gsea_delta_dotplot.png"), p_dot, width = 12, height = 8, dpi = 200)
  message("[SAVED] gsea_delta_dotplot")

  message("\n[TOP 10 activated pathways (acute phase)]")
  top_act <- gsea_df[gsea_df$enrichmentScore > 0, ][
    1:min(10, sum(gsea_df$enrichmentScore > 0)),
    c("Description", "NES", "p.adjust")]
  print(top_act, row.names = FALSE)

  message("\n[TOP 10 suppressed pathways (acute phase)]")
  top_sup <- gsea_df[gsea_df$enrichmentScore < 0, ][
    1:min(10, sum(gsea_df$enrichmentScore < 0)),
    c("Description", "NES", "p.adjust")]
  print(top_sup, row.names = FALSE)
} else {
  message("[WARN] GSEA: no significant terms at p.adjust < 0.05 — try relaxing pvalueCutoff.")
}

message("\n[DONE]")
