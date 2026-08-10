"""CSV persistence for per-request metrics and batch throughput."""

import csv
import os

from .names import METRICS_COLUMNS, THROUGHPUT_COLUMNS


def _write_csv_row(path, columns, row):
    if not path:
        return

    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def write_metrics_csv(path, row):
    _write_csv_row(path, METRICS_COLUMNS, row)


def write_throughput_csv(path, row):
    _write_csv_row(path, THROUGHPUT_COLUMNS, row)
