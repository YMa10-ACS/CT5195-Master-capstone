#!/usr/bin/env Rscript

script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("^--file=", script_args)])
script_dir <- if (length(script_path) > 0) dirname(normalizePath(script_path)) else getwd()
record_dir <- file.path(script_dir, "record_all/record_tdx")

csv_files <- c(
  http = file.path(record_dir, "distributed_vlm_metrics_http.csv"),
  "https-aes-gcm" = file.path(record_dir, "distributed_vlm_metrics_https-aes-gcm.csv"),
  "https-chacha20" = file.path(record_dir, "distributed_vlm_metrics_https-chacha20.csv")
)

latency_columns <- c(
  split_latency = "Split total",
  edge_embedding_latency = "Edge embedding",
  transfer_latency = "Transfer",
  network_overhead = "Network/protocol overhead",
  cloud_total_request = "Cloud request",
  local_baseline_latency = "Local baseline"
)

summary_csv <- file.path(record_dir, "distributed_vlm_latency_summary.csv")
output_png <- file.path(record_dir, "distributed_vlm_latency_comparison.png")

read_average_latency <- function(mode, csv_path) {
  if (!file.exists(csv_path)) {
    stop(paste("Missing CSV:", csv_path))
  }

  data <- read.csv(csv_path)
  result <- data.frame(stage = unname(latency_columns), stringsAsFactors = FALSE)

  for (column in names(latency_columns)) {
    if (column %in% names(data)) {
      result[result$stage == latency_columns[[column]], mode] <- mean(data[[column]], na.rm = TRUE)
    } else {
      result[result$stage == latency_columns[[column]], mode] <- NA_real_
    }
  }

  result
}

summary_table <- NULL
for (mode in names(csv_files)) {
  mode_summary <- read_average_latency(mode, csv_files[[mode]])
  if (is.null(summary_table)) {
    summary_table <- mode_summary
  } else {
    summary_table <- merge(summary_table, mode_summary, by = "stage", all = TRUE)
  }
}

summary_table <- summary_table[match(unname(latency_columns), summary_table$stage), ]
summary_table <- summary_table[rowSums(is.na(summary_table[, -1])) < length(csv_files), ]

write.csv(summary_table, summary_csv, row.names = FALSE)

cat("\nAverage latency by stage (seconds):\n")
print_table <- summary_table
print_table[, -1] <- round(print_table[, -1], 4)
print(print_table, row.names = FALSE)
cat("\nWrote summary CSV:", summary_csv, "\n")

plot_data <- as.matrix(summary_table[, -1])
rownames(plot_data) <- summary_table$stage

png(output_png, width = 1200, height = 700, res = 130)
par(mar = c(5, 11, 4, 2))
barplot(
  plot_data,
  beside = TRUE,
  horiz = TRUE,
  col = c("#4C78A8", "#F58518", "#54A24B", "#B279A2"),
  xlab = "Average latency (seconds)",
  main = "Latency Comparison by Transport Mode",
  las = 1,
  legend.text = rownames(plot_data),
  args.legend = list(x = "bottomright", inset = 0.02, cex = 0.8)
)
dev.off()

cat("Wrote bar chart:", output_png, "\n")
