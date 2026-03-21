suppressPackageStartupMessages({
  library(data.table)
  library(matrixStats)
  library(ComplexHeatmap)
  library(circlize)
  library(BloodGen3Module)
  library(grid)
})

# ----------------------------
# User paths
# ----------------------------
expr_path <- "burn_data/plotting/preprocessed/burn_gene_matrix.tsv"
meta_path <- "burn_data/plotting/preprocessed/burn_sample_metadata.tsv"
out_pdf   <- "burn_results/plotting/heatmap/burn_fingerprint_T0.pdf"

# ----------------------------
# Parameters
# ----------------------------
FC_thresh <- 1.5
min_pct_module_response <- 10
show_ref_group <- TRUE
cutoff_display <- 0

# If TRUE: columns cluster (often separates by time naturally)
# If FALSE: keep your current column order
cluster_columns_flag <- TRUE

# If TRUE: split rows by BloodGen3 aggregate cluster (A28/A35/...)
rowSplit_flag <- TRUE

# ----------------------------
# Load data
# ----------------------------
expr <- fread(expr_path)
gene_col <- names(expr)[1]
genes <- expr[[gene_col]]
expr_mat <- as.matrix(expr[, -1, with=FALSE])
rownames(expr_mat) <- genes

meta <- fread(meta_path)
meta <- as.data.frame(meta)
rownames(meta) <- meta[[1]]  # sample_id
meta[[1]] <- NULL

# Align columns to metadata rows
common_samples <- intersect(colnames(expr_mat), rownames(meta))
expr_mat <- expr_mat[, common_samples, drop=FALSE]
meta <- meta[common_samples, , drop=FALSE]

if (!("time_bin" %in% colnames(meta))) stop("meta missing time_bin column")
if (!("is_T0" %in% colnames(meta))) stop("meta missing is_T0 column")

t0_samples <- rownames(meta)[which(as.logical(meta$is_T0))]
if (length(t0_samples) < 2) stop("Need at least 2 T0 samples to use as reference")

message("[INFO] Genes: ", nrow(expr_mat), " | Samples: ", ncol(expr_mat))
message("[INFO] T0 samples: ", length(t0_samples))

# ----------------------------
# Ensure log2 scale (heuristic)
# ----------------------------
q99 <- as.numeric(quantile(expr_mat, 0.99, na.rm=TRUE))
if (q99 > 100) {
  message("[INFO] Expression looks non-log2 (q99=", round(q99,1), "). Applying log2(x+1).")
  expr_mat <- log2(expr_mat + 1)
} else {
  message("[INFO] Expression looks log-like (q99=", round(q99,1), "). Leaving as-is.")
}

# ----------------------------
# Differential per sample vs T0 (log2FC)
# ----------------------------
t0_mean <- rowMeans(expr_mat[, t0_samples, drop=FALSE], na.rm=TRUE)
log2fc <- sweep(expr_mat, 1, t0_mean, FUN="-")
log2FC_cut <- log2(FC_thresh)

is_up   <- log2fc >=  log2FC_cut
is_down <- log2fc <= -log2FC_cut

# ----------------------------
# BloodGen3 objects
# ----------------------------
Module_listGen3_raw <- get("Module_listGen3", envir=asNamespace("BloodGen3Module"))
Gen3_ann <- get("Gen3_ann", envir=asNamespace("BloodGen3Module"))

message("[DEBUG] Module_listGen3 class: ", paste(class(Module_listGen3_raw), collapse=", "))
message("[DEBUG] Gen3_ann dim: ", paste(dim(Gen3_ann), collapse=" x "))

# IMPORTANT FIX: Module_listGen3 might be data.frame
if (is.data.frame(Module_listGen3_raw)) {
  if (!all(c("Module", "Gene") %in% colnames(Module_listGen3_raw))) {
    stop("Module_listGen3 data.frame missing expected columns Module/Gene")
  }
  Module_listGen3 <- split(as.character(Module_listGen3_raw$Gene),
                           as.character(Module_listGen3_raw$Module))
} else if (is.list(Module_listGen3_raw)) {
  Module_listGen3 <- Module_listGen3_raw
} else {
  stop("Unknown Module_listGen3 type: ", paste(class(Module_listGen3_raw), collapse=", "))
}

module_names <- names(Module_listGen3)
message("[DEBUG] n unique modules: ", length(module_names))
message("[DEBUG] first module names: ", paste(head(module_names), collapse=", "))

genes_in_data <- rownames(expr_mat)

# ----------------------------
# Module annotation table (from Gen3_ann)
# ----------------------------
Gen3_ann <- as.data.frame(Gen3_ann, stringsAsFactors = FALSE)

needed_cols <- c("Module", "Function", "Cluster", "Module_color")
missing_cols <- setdiff(needed_cols, colnames(Gen3_ann))
if (length(missing_cols) > 0) {
  stop("Gen3_ann missing columns: ", paste(missing_cols, collapse=", "))
}

anno_table <- unique(Gen3_ann[, needed_cols])
anno_table$Module <- as.character(anno_table$Module)
anno_table$Function <- as.character(anno_table$Function)
anno_table$Cluster <- as.character(anno_table$Cluster)
anno_table$Module_color <- as.character(anno_table$Module_color)
rownames(anno_table) <- anno_table$Module

# ----------------------------
# Module response: %up - %down per sample
# ----------------------------
module_response <- matrix(0, nrow=length(module_names), ncol=ncol(expr_mat))
rownames(module_response) <- module_names
colnames(module_response) <- colnames(expr_mat)

for (m in module_names) {
  gset <- intersect(Module_listGen3[[m]], genes_in_data)
  if (length(gset) == 0) next
  up_pct   <- colMeans(is_up[gset, , drop=FALSE], na.rm=TRUE)   * 100
  down_pct <- colMeans(is_down[gset, , drop=FALSE], na.rm=TRUE) * 100
  module_response[m, ] <- up_pct - down_pct
}

# ----------------------------
# Filter modules
# ----------------------------
keep_mod <- rowMaxs(abs(module_response), na.rm=TRUE) >= min_pct_module_response
module_response <- module_response[keep_mod, , drop=FALSE]
message("[INFO] Modules kept: ", nrow(module_response), " / ", length(module_names))

if (nrow(module_response) < 2) {
  stop("Too few modules after filtering. Try lowering min_pct_module_response.")
}

# ----------------------------
# Optionally drop reference group columns
# ----------------------------
if (!show_ref_group) {
  keep_cols <- setdiff(colnames(module_response), t0_samples)
  module_response <- module_response[, keep_cols, drop=FALSE]
  meta <- meta[keep_cols, , drop=FALSE]
}

# ----------------------------
# Join annotations
# ----------------------------
mod_ids <- rownames(module_response)
ann_sub <- anno_table[mod_ids, , drop=FALSE]

missing_ann <- is.na(ann_sub$Function) | is.na(ann_sub$Cluster) | is.na(ann_sub$Module_color)
if (any(missing_ann)) {
  ann_sub$Function[missing_ann] <- "TBD"
  ann_sub$Cluster[missing_ann] <- "Unk"
  ann_sub$Module_color[missing_ann] <- "#BBBBBB"
}

ann_sub$Module_func <- paste(ann_sub$Module, ann_sub$Function, sep=".")
rownames(module_response) <- ann_sub$Module_func
rownames(ann_sub) <- ann_sub$Module_func

# Drop TBD
keep_func <- ann_sub$Function != "TBD"
module_response <- module_response[keep_func, , drop=FALSE]
ann_sub <- ann_sub[keep_func, , drop=FALSE]

if (nrow(module_response) < 2) {
  stop("After dropping TBD, too few modules. Disable the TBD filter if needed.")
}

# Display cutoff
plot_mat <- module_response
plot_mat[abs(plot_mat) < cutoff_display] <- 0

message("[DEBUG] plot_mat range: ", paste(range(plot_mat, na.rm=TRUE), collapse=" .. "))
message("[DEBUG] nonzero fraction: ", round(mean(plot_mat != 0, na.rm=TRUE), 6))

# ----------------------------
# Row split
# ----------------------------
row_split <- NULL
if (rowSplit_flag) row_split <- ann_sub$Cluster

# Color mapping
col_fun <- colorRamp2(c(-100, 0, 100), c("blue", "white", "red"))

# ----------------------------
# ORDER COLUMNS BY TIME PHASE (robust)
# ----------------------------
phase_levels <- c("T0", "Early", "Mid", "Late", "FollowUp", "Other")

meta$time_bin <- trimws(as.character(meta$time_bin))
meta$time_bin[is.na(meta$time_bin) | meta$time_bin == ""] <- "Other"

# common variants
meta$time_bin[meta$time_bin %in% c("Followup", "Follow-up", "Follow_up", "Follow UP")] <- "FollowUp"
meta$time_bin[meta$time_bin %in% c("t0", "T 0", "t 0")] <- "T0"

meta$time_bin <- factor(meta$time_bin, levels = phase_levels)

ord <- order(meta$time_bin)
plot_mat <- plot_mat[, ord, drop = FALSE]
meta <- meta[ord, , drop = FALSE]

# rebuild annotation AFTER ordering (important)
time_levels_present <- phase_levels[phase_levels %in% levels(droplevels(meta$time_bin))]
time_cols <- structure(rand_color(length(time_levels_present)), names=time_levels_present)

ha <- HeatmapAnnotation(
  Time = meta$time_bin,
  col = list(Time = time_cols),
  show_annotation_name = TRUE,
  simple_anno_size = unit(0.25, "cm")
)

# ----------------------------
# Build heatmap object
# ----------------------------
ht <- Heatmap(
  plot_mat,
  name = "% Response",
  col = col_fun,

  top_annotation = ha,

  cluster_rows = TRUE,
  cluster_columns = FALSE,  # <- KEEP phase order
  row_split = row_split,

  # REMOVE labels left/right
  show_row_names = FALSE,
  show_column_names = FALSE,

  row_title_rot = 0,
  row_title_gp  = gpar(fontsize = 9),   # smaller text
  row_gap       = unit(2.5, "mm"), # space between A labels

  use_raster = FALSE
)

# ----------------------------
# Save outputs (PNG + PDF)
# ----------------------------
dir.create(dirname(out_pdf), recursive = TRUE, showWarnings = FALSE)

png_file <- sub("\\.pdf$", ".png", out_pdf)
png(png_file, width = 3200, height = 2400, res = 200)
draw(ht, heatmap_legend_side = "left", annotation_legend_side = "left")
dev.off()
message("[SAVED] ", png_file)

pdf(out_pdf, width = 22, height = 16, useDingbats = FALSE)
draw(ht, heatmap_legend_side = "left", annotation_legend_side = "left")
dev.off()
message("[SAVED] ", out_pdf)