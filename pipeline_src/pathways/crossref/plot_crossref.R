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
  library(ragg)
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
base_theme <- theme_classic(base_size = 12, base_family = "Arial") +
  theme(
    axis.text  = element_text(size = 11),
    axis.title = element_text(size = 12),
    legend.text  = element_text(size = 11),
    legend.title = element_text(size = 11),
    plot.title   = element_text(size = 13, face = "bold")
  )

save_fig <- function(p, name, w, h) {
  ggsave(file.path(FIG_DIR, paste0(name, ".pdf")), p, width = w, height = h,
         device = cairo_pdf)
  ggsave(file.path(FIG_DIR, paste0(name, ".png")), p, width = w, height = h,
         dpi = 200, device = agg_png)
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
            hjust = 0, nudge_x = 0.08, size = 3.7, colour = "#444444") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = "Knockout impact score", y = NULL) +
  base_theme +
  theme(legend.position = "right")

save_fig(pA, "optionA_lollipop", w = 8, h = 6)

# ---------------------------------------------------------------------------
# Option B — Scatter with zoom inset
# ---------------------------------------------------------------------------

# Define zoom region (crowded area around impact score ~1.5-2.6, pathways 0-13)
ZOOM_X <- c(1.45, 2.65)
ZOOM_Y <- c(-1,    7.5)

zoom_data <- bubble_set %>%
  filter(knockout_impact_score >= ZOOM_X[1] & knockout_impact_score <= ZOOM_X[2] &
         n_pathways_hit        >= ZOOM_Y[1] & n_pathways_hit        <= ZOOM_Y[2])

# Main plot — label only genes outside the zoom region or with many pathway hits
pB_main <- ggplot(bubble_set,
                  aes(x = knockout_impact_score, y = n_pathways_hit,
                      colour = module)) +
  geom_point(size = 3, alpha = 0.85) +
  annotate("rect",
           xmin = ZOOM_X[1], xmax = ZOOM_X[2],
           ymin = ZOOM_Y[1], ymax = ZOOM_Y[2],
           fill = NA, colour = "#444444", linewidth = 0.6, linetype = "dashed") +
  geom_text_repel(
    data = bubble_set %>% filter(!(knockout_impact_score >= ZOOM_X[1] &
                                   knockout_impact_score <= ZOOM_X[2] &
                                   n_pathways_hit        >= ZOOM_Y[1] &
                                   n_pathways_hit        <= ZOOM_Y[2]) &
                                 gene != "TREM1"),
    aes(label = gene), size = 3.7, colour = "#333333",
    box.padding = 0.6, point.padding = 0.4,
    segment.size = 0.3, segment.colour = "#aaaaaa",
    min.segment.length = 0, force = 3, seed = 42) +
  geom_text_repel(
    data = bubble_set %>% filter(gene == "TREM1"),
    aes(label = gene), size = 3.7, colour = "#333333",
    nudge_y = 4, direction = "y",
    segment.size = 0.3, segment.colour = "#aaaaaa",
    min.segment.length = 0) +
  scale_colour_manual(values = MODULE_COLOURS, name = "Module") +
  labs(x = "Knockout impact score", y = "Pathways hit (n)") +
  base_theme +
  theme(legend.position      = c(0.01, 0.99),
        legend.justification = c(0, 1),
        legend.background    = element_rect(fill = "white", colour = NA))

# Inset zoom plot
pB_inset <- ggplot(zoom_data,
                   aes(x = knockout_impact_score, y = n_pathways_hit,
                       colour = module)) +
  geom_point(size = 2.5, alpha = 0.85) +
  geom_text_repel(aes(label = gene), size = 3.1, colour = "#333333",
                  box.padding = 0.6, point.padding = 0.4,
                  segment.size = 0.3, segment.colour = "#aaaaaa",
                  min.segment.length = 0, force = 12, force_pull = 0.5,
                  seed = 7, max.overlaps = Inf) +
  scale_colour_manual(values = MODULE_COLOURS) +
  coord_cartesian(xlim = ZOOM_X, ylim = ZOOM_Y) +
  theme_classic(base_size = 9, base_family = "Arial") +
  theme(legend.position  = "none",
        panel.border     = element_rect(fill = NA, colour = "#444444", linewidth = 0.6),
        axis.title       = element_blank(),
        axis.text        = element_text(size = 8),
        plot.background  = element_rect(fill = "white", colour = NA))

# Combine with cowplot inset
if (requireNamespace("cowplot", quietly = TRUE)) {
  library(cowplot)
  pB <- ggdraw(pB_main) +
    draw_plot(pB_inset, x = 0.52, y = 0.50, width = 0.45, height = 0.47)
} else {
  message("[NOTE] Install cowplot for inset zoom; saving main plot only")
  pB <- pB_main
}

save_fig(pB, "optionB_bubble", w = 10, h = 6.5)

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
            hjust = -0.1, size = 3.7, colour = "#444444") +
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
            hjust = -0.1, size = 3.7, colour = "#444444") +
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
