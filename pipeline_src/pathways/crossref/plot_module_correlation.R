#!/usr/bin/env Rscript
# plot_module_correlation.R
# -------------------------
# Module-level correlation block matrix (6×6).
# Builds a gene × pathway binary matrix from knockout-pathway crossref,
# computes pairwise Spearman rho between genes across pathway space,
# then averages within and between functional modules.
#
# Colormap for the legend matches diffusion analysis Panel A:
#   #F2F2F2 → #4C72B0 → #4D9078 → #F2C14E → #F78154 → #B4436C
#
# Input:  results/knockout_pathway_crossref.csv
#         results/knockout_pathway_summary.csv
# Output: results/figures/module_correlation.pdf/.png
#
# Usage:
#   cd PIGLasso/pipeline_src/pathways/crossref/
#   Rscript plot_module_correlation.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
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

CROSSREF_FILE <- file.path(HERE, "results", "knockout_pathway_crossref.csv")
SUMMARY_FILE  <- file.path(HERE, "results", "knockout_pathway_summary.csv")
FIG_DIR       <- file.path(HERE, "results", "figures")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Module colours (project-wide)
# ---------------------------------------------------------------------------
MODULE_COLOURS <- c(
  immune          = "#B4436C",
  signalling      = "#4C72B0",
  apoptosis       = "#4D9078",
  metabolism      = "#F78154",
  epigenetic      = "#F2C14E",
  uncharacterised = "#999999"
)
MODULES <- names(MODULE_COLOURS)

# ---------------------------------------------------------------------------
# NODIS colormap — matches diffusion analysis Panel A
# ---------------------------------------------------------------------------
NODIS_PALETTE <- colorRampPalette(
  c("#F2F2F2", "#4C72B0", "#4D9078", "#F2C14E", "#F78154", "#B4436C")
)(256)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
crossref   <- read.csv(CROSSREF_FILE, stringsAsFactors = FALSE)
summary_df <- read.csv(SUMMARY_FILE,  stringsAsFactors = FALSE)

message(sprintf("Loaded %d pathways from crossref", nrow(crossref)))

# ---------------------------------------------------------------------------
# Build gene × pathway binary matrix
# ko_genes_hit is comma-separated per pathway row
# ---------------------------------------------------------------------------
gene_pathway_long <- crossref %>%
  mutate(gene = strsplit(as.character(ko_genes_hit), ";\\s*")) %>%
  unnest(gene) %>%
  filter(nchar(trimws(gene)) > 0) %>%
  mutate(gene = trimws(gene)) %>%
  select(gene, pathway_id) %>%
  distinct() %>%
  mutate(hit = 1L)

gene_path_wide <- gene_pathway_long %>%
  pivot_wider(names_from = pathway_id, values_from = hit, values_fill = 0L)

genes_with_hits <- gene_path_wide$gene
mat             <- as.matrix(gene_path_wide[, -1])
rownames(mat)   <- genes_with_hits

message(sprintf("Gene × pathway matrix: %d genes × %d pathways",
                nrow(mat), ncol(mat)))

# ---------------------------------------------------------------------------
# Attach module assignments; restrict to genes present in both files
# ---------------------------------------------------------------------------
gene_mod <- summary_df %>%
  filter(gene %in% genes_with_hits) %>%
  select(gene, module) %>%
  mutate(module = tolower(trimws(module)))

genes_keep <- intersect(genes_with_hits, gene_mod$gene)
mat        <- mat[genes_keep, , drop = FALSE]
mod_map    <- setNames(gene_mod$module[match(genes_keep, gene_mod$gene)], genes_keep)

message(sprintf("Genes with module info: %d", length(genes_keep)))

# ---------------------------------------------------------------------------
# Pairwise Spearman rho between genes across pathway space
# ---------------------------------------------------------------------------
rho_mat <- cor(t(mat), method = "spearman")
rownames(rho_mat) <- colnames(rho_mat) <- genes_keep

# ---------------------------------------------------------------------------
# Module × Module mean Spearman rho (off-diagonal pairs only on diagonal)
# ---------------------------------------------------------------------------
n_mod   <- length(MODULES)
mod_rho <- matrix(NA_real_, n_mod, n_mod, dimnames = list(MODULES, MODULES))
mod_n   <- matrix(0L,       n_mod, n_mod, dimnames = list(MODULES, MODULES))

for (mi in seq_along(MODULES)) {
  for (mj in seq_along(MODULES)) {
    m1 <- MODULES[mi]; m2 <- MODULES[mj]
    g1 <- intersect(names(mod_map)[mod_map == m1], rownames(rho_mat))
    g2 <- intersect(names(mod_map)[mod_map == m2], rownames(rho_mat))

    if (length(g1) == 0 || length(g2) == 0) {
      mod_rho[mi, mj] <- 0; next
    }

    if (mi == mj) {
      if (length(g1) > 1) {
        sub  <- rho_mat[g1, g1]
        vals <- sub[upper.tri(sub)]
        mod_rho[mi, mj] <- mean(vals, na.rm = TRUE)
        mod_n[mi, mj]   <- length(vals)
      } else {
        mod_rho[mi, mj] <- 1.0
        mod_n[mi, mj]   <- 0L
      }
    } else {
      vals            <- as.vector(rho_mat[g1, g2])
      mod_rho[mi, mj] <- mean(vals, na.rm = TRUE)
      mod_n[mi, mj]   <- length(vals)
    }
  }
}

message("Module × module mean Spearman rho:")
print(round(mod_rho, 3))

# ---------------------------------------------------------------------------
# Build tidy data frame for ggplot
# ---------------------------------------------------------------------------
df_plot <- expand.grid(mod_row = MODULES, mod_col = MODULES,
                       stringsAsFactors = FALSE) %>%
  mutate(
    rho       = as.vector(mod_rho),
    n         = as.integer(as.vector(mod_n)),
    cell_lbl  = sprintf("%.2f\n(n=%d)", rho, n),
    txt_col   = ifelse(rho > 0.30 | rho < -0.05, "white", "black"),
    is_diag   = mod_row == mod_col
  ) %>%
  mutate(
    mod_row = factor(mod_row, levels = rev(MODULES)),
    mod_col = factor(mod_col, levels = MODULES)
  )

# Coloured axis label data frames (placed via geom_text with clip = "off")
x_labels <- data.frame(
  mod_col   = factor(MODULES, levels = MODULES),
  mod_row   = factor(rev(MODULES)[1], levels = rev(MODULES)),
  label     = tools::toTitleCase(MODULES),
  lbl_color = unname(MODULE_COLOURS[MODULES]),
  stringsAsFactors = FALSE
)
y_labels <- data.frame(
  mod_col   = factor(MODULES[1], levels = MODULES),
  mod_row   = factor(rev(MODULES), levels = rev(MODULES)),
  label     = tools::toTitleCase(rev(MODULES)),
  lbl_color = unname(MODULE_COLOURS[rev(MODULES)]),
  stringsAsFactors = FALSE
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
p <- ggplot(df_plot, aes(x = mod_col, y = mod_row)) +

  # Heatmap tiles
  geom_tile(aes(fill = rho), colour = "white", linewidth = 0.7) +

  # Bold diagonal border
  geom_tile(data = filter(df_plot, is_diag),
            aes(fill = rho), colour = "#333333", linewidth = 1.1) +

  # Cell text: rho + gene-pair count
  geom_text(aes(label = cell_lbl, colour = txt_col),
            size = 3.0, lineheight = 0.9, fontface = "plain") +

  # Coloured x-axis labels (below plot, clip off)
  geom_text(data = x_labels,
            mapping = aes(x = mod_col, label = label, colour = lbl_color),
            y = 0.38, vjust = 1, angle = 35, hjust = 1,
            size = 3.7, fontface = "bold", inherit.aes = FALSE) +

  # Coloured y-axis labels (left of plot, clip off)
  geom_text(data = y_labels,
            mapping = aes(y = mod_row, label = label, colour = lbl_color),
            x = 0.38, hjust = 1,
            size = 3.7, fontface = "bold", inherit.aes = FALSE) +

  scale_fill_gradientn(
    colours = NODIS_PALETTE,
    limits  = c(-0.5, 0.5),
    name    = expression("Mean Spearman" ~ rho),
    breaks  = c(-0.5, -0.25, 0, 0.25, 0.5),
    labels  = c("-0.50", "-0.25", "0.00", "0.25", "0.50"),
    guide   = guide_colourbar(barheight = unit(4, "cm"), ticks.linewidth = 0.5)
  ) +
  scale_colour_identity() +

  labs(x = NULL, y = NULL,
       title    = "Module-level co-regulation in pathway space",
       subtitle = "Mean pairwise Spearman ρ between functional modules") +

  coord_cartesian(clip = "off") +

  theme_classic(base_size = 12, base_family = "Arial") +
  theme(
    axis.text        = element_blank(),
    axis.ticks       = element_blank(),
    axis.line        = element_blank(),
    panel.border     = element_rect(fill = NA, colour = "#888888", linewidth = 0.5),
    plot.title       = element_text(size = 12, face = "bold"),
    plot.subtitle    = element_text(size = 10, colour = "#555555"),
    legend.title     = element_text(size = 10),
    legend.text      = element_text(size = 9),
    plot.margin      = margin(8, 8, 60, 90)
  )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
ggsave(file.path(FIG_DIR, "module_correlation.pdf"),
       p, width = 8.5, height = 6.5, device = cairo_pdf)
ggsave(file.path(FIG_DIR, "module_correlation.png"),
       p, width = 8.5, height = 6.5, dpi = 200, device = agg_png)
message("[SAVED] module_correlation")
message("[DONE]")
