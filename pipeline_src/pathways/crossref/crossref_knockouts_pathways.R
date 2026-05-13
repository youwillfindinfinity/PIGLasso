#!/usr/bin/env Rscript
# crossref_knockouts_pathways.R
# -----------------------------
# Cross-references the ranked knockout genes against the GSEA delta
# pathway results (T0+Early vs Late+FollowUp).
#
# For each enriched pathway, identifies which knockout genes appear in
# its core enrichment set, and annotates with rank and impact score.
#
# Input:  results/knockouts/ranked_genes_with_modules.csv
#         pipeline_src/pathways/results/GSE182616/delta/gsea_delta_results.csv
# Output: pipeline_src/pathways/crossref/results/
#           knockout_pathway_crossref.csv   (pathway-level: which KO genes hit each pathway)
#           knockout_pathway_summary.csv    (gene-level: which pathways each KO gene appears in)

suppressPackageStartupMessages({
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
ROOT      <- normalizePath(file.path(HERE, "..", "..", ".."), mustWork = FALSE)
KO_FILE   <- file.path(ROOT, "results", "knockouts", "ranked_genes_with_modules.csv")
GSEA_FILE <- file.path(HERE, "..", "results", "GSE182616", "delta", "gsea_delta_results.csv")
OUT_DIR   <- file.path(HERE, "results")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ko   <- read.csv(KO_FILE,   stringsAsFactors = FALSE)
gsea <- read.csv(GSEA_FILE, stringsAsFactors = FALSE)

message("[INFO] Knockout genes: ", nrow(ko))
message("[INFO] GSEA pathways:  ", nrow(gsea))

ko_genes <- ko$gene

# ---------------------------------------------------------------------------
# Pathway-level cross-reference
# ---------------------------------------------------------------------------
rows <- list()

for (i in seq_len(nrow(gsea))) {
  core_genes <- strsplit(gsea$core_enrichment[i], "/")[[1]]
  hits       <- intersect(core_genes, ko_genes)
  if (length(hits) == 0) next

  hit_info <- ko[ko$gene %in% hits, c("rank", "gene", "knockout_impact_score", "module")]
  hit_info <- hit_info[order(hit_info$rank), ]

  rows[[length(rows) + 1]] <- data.frame(
    pathway_id          = gsea$ID[i],
    pathway_description = gsea$Description[i],
    NES                 = gsea$NES[i],
    p.adjust            = gsea$p.adjust[i],
    direction           = ifelse(gsea$enrichmentScore[i] > 0, "activated", "suppressed"),
    n_core_genes        = length(core_genes),
    n_ko_hits           = length(hits),
    ko_genes_hit        = paste(hit_info$gene,                          collapse = "; "),
    ko_ranks_hit        = paste(hit_info$rank,                          collapse = "; "),
    ko_scores_hit       = paste(round(hit_info$knockout_impact_score, 3), collapse = "; "),
    ko_modules_hit      = paste(hit_info$module,                        collapse = "; "),
    stringsAsFactors    = FALSE
  )
}

crossref <- do.call(rbind, rows)
crossref  <- crossref[order(crossref$direction, -abs(crossref$NES)), ]

write.csv(crossref, file.path(OUT_DIR, "knockout_pathway_crossref.csv"), row.names = FALSE)
message("[SAVED] knockout_pathway_crossref.csv  (", nrow(crossref), " pathways with KO hits)")

# ---------------------------------------------------------------------------
# Gene-level summary: for each KO gene, which pathways does it appear in?
# ---------------------------------------------------------------------------
gene_rows <- list()

for (g in ko_genes) {
  in_pathways <- crossref[grepl(paste0("\\b", g, "\\b"), crossref$ko_genes_hit), ]
  if (nrow(in_pathways) == 0) next

  ko_info <- ko[ko$gene == g, ]
  gene_rows[[length(gene_rows) + 1]] <- data.frame(
    rank                  = ko_info$rank,
    gene                  = g,
    knockout_impact_score = ko_info$knockout_impact_score,
    module                = ko_info$module,
    n_pathways_hit        = nrow(in_pathways),
    n_activated           = sum(in_pathways$direction == "activated"),
    n_suppressed          = sum(in_pathways$direction == "suppressed"),
    top_activated_pathway = ifelse(
      any(in_pathways$direction == "activated"),
      in_pathways$pathway_description[in_pathways$direction == "activated"][
        which.max(abs(in_pathways$NES[in_pathways$direction == "activated"]))],
      NA),
    top_suppressed_pathway = ifelse(
      any(in_pathways$direction == "suppressed"),
      in_pathways$pathway_description[in_pathways$direction == "suppressed"][
        which.max(abs(in_pathways$NES[in_pathways$direction == "suppressed"]))],
      NA),
    stringsAsFactors = FALSE
  )
}

gene_summary <- do.call(rbind, gene_rows)
gene_summary <- gene_summary[order(gene_summary$rank), ]

write.csv(gene_summary, file.path(OUT_DIR, "knockout_pathway_summary.csv"), row.names = FALSE)
message("[SAVED] knockout_pathway_summary.csv  (", nrow(gene_summary), " KO genes with pathway hits)")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
message("\n[SUMMARY]")
message("  Pathways with >= 1 KO gene in core enrichment: ", nrow(crossref))
message("    Activated (acute):  ", sum(crossref$direction == "activated"))
message("    Suppressed (acute): ", sum(crossref$direction == "suppressed"))
message("  KO genes hitting >= 1 pathway: ", nrow(gene_summary), " / ", nrow(ko))

message("\n[TOP 10 ACTIVATED pathways by NES]")
top_act <- crossref[crossref$direction == "activated", ]
top_act <- top_act[order(-top_act$NES), ]
print(head(top_act[, c("pathway_description", "NES", "p.adjust", "n_ko_hits", "ko_genes_hit")], 10),
      row.names = FALSE)

message("\n[TOP 10 SUPPRESSED pathways by |NES|]")
top_sup <- crossref[crossref$direction == "suppressed", ]
top_sup <- top_sup[order(top_sup$NES), ]
print(head(top_sup[, c("pathway_description", "NES", "p.adjust", "n_ko_hits", "ko_genes_hit")], 10),
      row.names = FALSE)

message("\n[TOP 20 KO GENES by pathway coverage]")
print(head(gene_summary[order(-gene_summary$n_pathways_hit), ], 20), row.names = FALSE)
