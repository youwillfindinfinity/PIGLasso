#!/usr/bin/env Rscript
# plot_crossref.R
# ---------------
# Three complementary visualisations of knockout genes × pathway enrichment.
#
#   A  Lollipop — top 20 by knockout impact score; pathway-hit genes highlighted
#   B  Bubble   — top 20 by impact + top 20 by pathway coverage (union);
#                 x=rank, y=n_pathways_hit, size=impact, colour=module
#   C  Two-panel — left: top 20 by impact score; right: top 20 by pathway coverage
#
# Input:  crossref/results/knockout_pathway_summary.csv
#         results/knockouts/ranked_genes_with_modules.csv
# Output: crossref/results/figures/

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(ggrepel)
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE <- tryCatch({
  args <- commandArgs(trailingOnly = FALSE)
  f    <- sub("--file=", "", args[grepl("--file=", args)])
  dirname(normalizePath(f))
}, error = function(e) getwd())
ROOT     <- normalizePath(file.path(HERE, "..", "..", ".."), mustWork = FALSE)
KO_FILE  <- file.path(ROOT, "pipeline_src", "knockouts", "results", "ranked_genes_with_modules.csv")
SUM_FILE <- file.path(HERE, "results", "knockout_pathway_summary.csv")
FIG_DIR  <- file.path(HERE, "results", "figures")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Colour palette (project-wide)
# ---------------------------------------------------------------------------
MODULE_COLOURS <- c(
  immune          = "#B4436C",
  signalling      = "#4C72B0",
  apoptosis       = "#4D9078",
  metabolism      = "#F78154",
  epigenetic      = "#F2C14E",
  uncharacterised = "#999999"
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
ko  <- read.csv(KO_FILE,  stringsAsFactors = FALSE)
sum <- read.csv(SUM_FILE, stringsAsFactors = FALSE)

# Merge: all 163 genes with pathway info (NA for non-hits)
all_genes <- ko %>%
  left_join(sum %>% select(gene, n_pathways_hit, n_activated,
                            top_activated_pathway),
            by = "gene") %>%
  mutate(
    n_pathways_hit   = ifelse(is.na(n_pathways_hit), 0L, n_pathways_hit),
    has_pathway_hit  = n_pathways_hit > 0
  )

top20_impact   <- all_genes %>% arrange(rank)          %>% head(20)
top20_pathways <- all_genes %>% arrange(-n_pathways_hit) %>% head(20)
bubble_set     <- bind_rows(top20_impact, top20_pathways) %>%
  distinct(gene, .keep_all = TRUE)

# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------
base_theme <- theme_classic(base_size = 12) +
  theme(
    axis.text  = element_text(size = 11),
    axis.title = element_text(size = 12),
    legend.text  = element_text(size = 11),
    legend.title = element_text(size = 11),
    plot.title   = element_text(size = 13, face = "bold")
  )

save_fig <- function(p, name, w, h) {
  ggsave(file.path(FIG_DIR, paste0(name, ".pdf")), p, width = w, height = h)
  ggsave(file.path(FIG_DIR, paste0(name, ".png")), p, width = w, height = h, dpi = 200)
  message("[SAVED] ", name)
}

# ---------------------------------------------------------------------------
# Option A — Lollipop: top 20 by knockout impact score
# ---------------------------------------------------------------------------
lollipop_df <- top20_impact %>%
  mutate(gene = factor(gene, levels = rev(gene)))

pA <- ggplot(lollipop_df, aes(x = knockout_impact_score, y = gene)) +
  geom_segment(aes(x = 0, xend = knockout_impact_score,
                   y = gene, yend = gene,
                   colour = module),
               linewidth = 0.9) +
  geom_point(aes(colour = module,
                 shape  = has_pathway_hit,
                 size   = has_pathway_hit)) +
  scale_colour_manual(values = MODULE_COLOURS, name = "Module") +
  scale_shape_manual(values = c("FALSE" = 1, "TRUE" = 19),
                     labels = c("FALSE" = "No hit", "TRUE" = "Pathway hit"),
                     name   = "") +
  scale_size_manual(values  = c("FALSE" = 2.5, "TRUE" = 3.5),
                    guide   = "none") +
  geom_text(data = filter(lollipop_df, has_pathway_hit),
            aes(label = n_pathways_hit),
            hjust = -0.5, size = 3.2, colour = "#444444") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = "Knockout impact score", y = NULL) +
  base_theme +
  theme(legend.position = "right")

save_fig(pA, "optionA_lollipop", w = 8, h = 6)

# ---------------------------------------------------------------------------
# Option B — Bubble: knockout rank vs pathway coverage
# ---------------------------------------------------------------------------
pB <- ggplot(bubble_set,
             aes(x = rank, y = n_pathways_hit,
                 size = knockout_impact_score,
                 colour = module)) +
  geom_point(alpha = 0.85) +
  geom_text_repel(aes(label = gene), size = 3.2,
                  max.overlaps = 20, colour = "#333333",
                  box.padding = 0.4, point.padding = 0.3) +
  scale_colour_manual(values = MODULE_COLOURS, name = "Module") +
  scale_size_continuous(range = c(2, 9), name = "Impact score") +
  scale_x_continuous(breaks = seq(0, 160, 20)) +
  labs(x = "Knockout rank", y = "Pathways hit (n)") +
  base_theme +
  theme(legend.position = "right")

save_fig(pB, "optionB_bubble", w = 9, h = 6)

# ---------------------------------------------------------------------------
# Option C — Two-panel horizontal bars
# ---------------------------------------------------------------------------

# Left panel: top 20 by impact score
left_df <- top20_impact %>%
  mutate(gene = factor(gene, levels = rev(gene)),
         label = ifelse(n_pathways_hit > 0,
                        paste0(n_pathways_hit, " pathways"), ""))

pC_left <- ggplot(left_df, aes(x = knockout_impact_score, y = gene, fill = module)) +
  geom_col(width = 0.7, colour = "#444444", linewidth = 0.4) +
  geom_text(aes(label = label),
            hjust = -0.1, size = 3.2, colour = "#444444") +
  scale_fill_manual(values = MODULE_COLOURS, name = "Module") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(x = "Knockout impact score", y = NULL,
       title = "Top 20 by impact score") +
  base_theme +
  theme(legend.position = "none")

# Right panel: top 20 by pathway coverage
right_df <- top20_pathways %>%
  mutate(gene = factor(gene, levels = rev(gene)),
         score_label = round(knockout_impact_score, 2))

pC_right <- ggplot(right_df, aes(x = n_pathways_hit, y = gene, fill = module)) +
  geom_col(width = 0.7, colour = "#444444", linewidth = 0.4) +
  geom_text(aes(label = score_label),
            hjust = -0.1, size = 3.2, colour = "#444444") +
  scale_fill_manual(values = MODULE_COLOURS, name = "Module") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.14))) +
  labs(x = "Pathways hit (n)", y = NULL,
       title = "Top 20 by pathway coverage") +
  base_theme +
  theme(legend.position = "right")

# Combine with patchwork if available, else save separately
if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  pC <- pC_left + pC_right +
    plot_layout(guides = "collect") &
    theme(legend.position = "right")
  save_fig(pC, "optionC_twopanel", w = 14, h = 6)
} else {
  save_fig(pC_left,  "optionC_left",  w = 7, h = 6)
  save_fig(pC_right, "optionC_right", w = 7, h = 6)
  message("[NOTE] Install patchwork for a combined two-panel figure")
}

message("\n[DONE]")
