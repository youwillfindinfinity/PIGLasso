#!/usr/bin/env Rscript
# plot_module_summary.R
# ---------------------
# Two-panel module-level summary figure (top 20 genes by KO impact score).
#
#   Left  — Stacked horizontal bar: total pathways hit per module,
#            segmented by KO impact score tier (Okabe-Ito palette).
#   Right — Boxplot: individual KO impact scores per module,
#            ordered by median (descending); boxes coloured by module.
#
# Input:  results/knockout_pathway_summary.csv
# Output: results/figures/module_summary.pdf/.png
#
# Usage:
#   cd PIGLasso/pipeline_src/pathways/crossref/
#   Rscript plot_module_summary.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
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

SUM_FILE <- file.path(HERE, "results", "knockout_pathway_summary.csv")
FIG_DIR  <- file.path(HERE, "results", "figures")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
MODULE_COLOURS <- c(
  Immune          = "#B4436C",
  Signalling      = "#4C72B0",
  Apoptosis       = "#4D9078",
  Metabolism      = "#F78154",
  Epigenetic      = "#F2C14E",
  Uncharacterised = "#999999"
)

# NODIS palette sampled to 5 tiers (light → saturated, matching diffusion Panel A)
NODIS_5 <- colorRampPalette(
  c("#4C72B0", "#4D9078", "#F2C14E", "#F78154", "#B4436C")
)(5)
TIER_LEVELS  <- c("<0.5", "0.5-1", "1-1.5", "1.5-2", ">2")
TIER_COLOURS <- setNames(NODIS_5, TIER_LEVELS)

# ---------------------------------------------------------------------------
# Load & prepare — top 20 by knockout impact score
# ---------------------------------------------------------------------------
sum_df <- read.csv(SUM_FILE, stringsAsFactors = FALSE) %>%
  arrange(desc(knockout_impact_score)) %>%
  mutate(
    module     = tools::toTitleCase(tolower(trimws(module))),
    score_tier = cut(knockout_impact_score,
                     breaks = c(0, 0.5, 1.0, 1.5, 2.0, Inf),
                     labels = TIER_LEVELS,
                     right  = FALSE,
                     include.lowest = TRUE)
  )

message(sprintf("Using top %d genes across %d modules", nrow(sum_df),
                n_distinct(sum_df$module)))

# ---------------------------------------------------------------------------
# Panel A — Stacked horizontal bar
# ---------------------------------------------------------------------------
bar_module_order <- sum_df %>%
  group_by(module) %>%
  summarise(total = sum(n_pathways_hit), .groups = "drop") %>%
  arrange(desc(total)) %>%
  pull(module)

stacked <- sum_df %>%
  group_by(module, score_tier) %>%
  summarise(total_pathways = sum(n_pathways_hit), .groups = "drop") %>%
  complete(module, score_tier = factor(TIER_LEVELS, levels = TIER_LEVELS),
           fill = list(total_pathways = 0L)) %>%
  mutate(module = factor(module, levels = rev(bar_module_order)))

pA <- ggplot(stacked, aes(x = total_pathways, y = module, fill = score_tier)) +
  geom_col(width = 0.65, colour = "#333333", linewidth = 0.25) +
  scale_fill_manual(values = TIER_COLOURS, name = "KO score tier",
                    limits = TIER_LEVELS) +
  scale_x_reverse(expand = expansion(mult = c(0, 0.05))) +
  scale_y_discrete(position = "right") +
  labs(x = "Total pathways hit", y = NULL) +
  theme_classic(base_size = 11, base_family = "Arial") +
  theme(
    axis.text.y      = element_text(size = 10, colour = "black", face = "plain",
                                    family = "Arial"),
    axis.text.x      = element_text(size = 9, family = "Arial"),
    axis.title.x     = element_text(size = 10, family = "Arial"),
    legend.position  = "bottom",
    legend.title     = element_text(size = 9),
    legend.text      = element_text(size = 8),
    legend.key.size  = unit(0.4, "cm"),
    legend.direction = "horizontal",
    plot.margin      = margin(5.5, 0, 5.5, 5.5)
  ) +
  guides(fill = guide_legend(nrow = 1))

# ---------------------------------------------------------------------------
# Panel B — Boxplot of KO impact scores per module (same order as panel A)
# ---------------------------------------------------------------------------
box_df <- sum_df %>%
  mutate(module = factor(module, levels = rev(bar_module_order)))

pB <- ggplot(box_df, aes(x = knockout_impact_score, y = module, fill = module)) +
  geom_boxplot(
    width         = 0.65,
    outlier.size  = 1.8,
    outlier.alpha = 0.6,
    linewidth     = 0.25,
    colour        = "#333333"
  ) +
  stat_summary(fun = median, geom = "point", shape = 21,
               size = 2.2, fill = "white", colour = "#333333", stroke = 0.8) +
  scale_fill_manual(values = MODULE_COLOURS, guide = "none") +
  scale_x_continuous(labels = scales::number_format(accuracy = 0.01)) +
  labs(x = "Knockout impact score", y = NULL) +
  theme_classic(base_size = 11, base_family = "Arial") +
  theme(
    axis.text.y   = element_blank(),
    axis.text.x   = element_text(size = 9,  family = "Arial"),
    axis.title.x  = element_text(size = 10, family = "Arial"),
    plot.margin   = margin(5.5, 5.5, 5.5, 0)
  )

# ---------------------------------------------------------------------------
# Combine & save
# ---------------------------------------------------------------------------
p_combined <- pA + pB +
  plot_layout(widths = c(1.3, 1))

ggsave(file.path(FIG_DIR, "module_summary.pdf"),
       p_combined, width = 9, height = 4.5, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "module_summary.png"),
       p_combined, width = 9, height = 4.5, dpi = 200, device = agg_png)
message("[SAVED] module_summary")
message("[DONE]")
