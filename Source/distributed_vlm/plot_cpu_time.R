#!/usr/bin/env Rscript

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("^--file=", script_args)])
script_dir <- if (length(script_path) > 0) dirname(normalizePath(script_path)) else getwd()
record_dir <- file.path(script_dir, "record")

csv_files <- c(
  http = file.path(record_dir, "distributed_vlm_metrics_http.csv"),
  "https-aes-gcm" = file.path(record_dir, "distributed_vlm_metrics_https-aes-gcm.csv"),
  "https-chacha20" = file.path(record_dir, "distributed_vlm_metrics_https-chacha20.csv")
)

cpu_columns <- c(
  edge_embedding_cpu_total = "Edge embedding CPU",
  edge_transfer_cpu_total = "Edge transfer CPU",
  local_baseline_cpu_total = "Local baseline CPU",
  cloud_cpu_total = "Cloud total CPU"
)

summary_csv <- file.path(record_dir, "distributed_vlm_cpu_summary.csv")
output_png <- file.path(record_dir, "distributed_vlm_cpu_comparison.png")

read_average_cpu_time <- function(mode, csv_path) {
  if (!file.exists(csv_path)) {
    stop(paste("Missing CSV:", csv_path))
  }

  data <- read.csv(csv_path)
  result <- data.frame(stage = unname(cpu_columns), stringsAsFactors = FALSE)

  for (column in names(cpu_columns)) {
    result[result$stage == cpu_columns[[column]], mode] <- mean(data[[column]], na.rm = TRUE)
  }

  result
}

summary_table <- NULL
for (mode in names(csv_files)) {
  mode_summary <- read_average_cpu_time(mode, csv_files[[mode]])
  if (is.null(summary_table)) {
    summary_table <- mode_summary
  } else {
    summary_table <- merge(summary_table, mode_summary, by = "stage", all = TRUE)
  }
}

summary_table <- summary_table[match(unname(cpu_columns), summary_table$stage), ]

write.csv(summary_table, summary_csv, row.names = FALSE)

cat("\nAverage CPU time by stage (seconds):\n")
print_table <- summary_table
print_table[, -1] <- round(print_table[, -1], 4)
print(print_table, row.names = FALSE)
cat("\nWrote CPU summary CSV:", summary_csv, "\n")

plot_data <- as.matrix(summary_table[, -1])
rownames(plot_data) <- summary_table$stage

png(output_png, width = 1200, height = 700, res = 130)
par(mar = c(5, 13, 4, 2))
barplot(
  plot_data,
  beside = TRUE,
  horiz = TRUE,
  col = c("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"),
  xlab = "Average CPU time (seconds)",
  main = "CPU Time Comparison by Transport Mode",
  las = 1,
  legend.text = rownames(plot_data),
  args.legend = list(x = "bottomright", inset = 0.02, cex = 0.8)
)
dev.off()

cat("Wrote CPU bar chart:", output_png, "\n")
