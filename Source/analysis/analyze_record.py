#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict
from statistics import mean, median, stdev


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(SCRIPT_DIR, "record")
OUTPUT_DIR = os.path.join(RECORD_DIR, "analysis")

TARGET_LABELS = {
    "10.128.0.6": "Confidential VM / TDX",
    "10.128.0.7": "Normal VM",
    "local": "Local baseline",
}

NUMERIC_COLUMNS = {
    "payload_bytes",
    "split_latency",
    "edge_embedding_latency",
    "transfer_latency",
    "network_overhead",
    "local_baseline_latency",
    "cloud_total_request",
    "cloud_reconstruct_time",
    "cloud_prompt_time",
    "cloud_combine_time",
    "cloud_generate_time",
    "cloud_generate_tokens",
    "cloud_generate_prefill_time",
    "cloud_generate_head_time",
    "cloud_generate_sample_time",
    "cloud_generate_token_embedding_time",
    "cloud_generate_attention_mask_time",
    "cloud_generate_decode_step_time",
    "cloud_generate_postprocess_time",
    "cloud_decoder_rotary_embd_time",
    "cloud_decoder_norm_time",
    "cloud_decoder_qkv_proj_time",
    "cloud_decoder_rotary_time",
    "cloud_decoder_kv_cat_time",
    "cloud_decoder_kv_repeat_time",
    "cloud_decoder_attention_mask_time",
    "cloud_decoder_attention_core_time",
    "cloud_decoder_out_proj_time",
    "cloud_decoder_mlp_time",
    "cloud_decoder_lm_head_time",
    "edge_embedding_cpu_total",
    "edge_embedding_cpu_avg_pct",
    "edge_embedding_cpu_max_pct",
    "edge_transfer_cpu_total",
    "local_baseline_cpu_total",
    "edge_generate_cpu_avg_pct",
    "edge_generate_cpu_max_pct",
    "edge_cpu_total",
    "cloud_cpu_total",
    "cloud_cpu_avg_pct",
    "cloud_cpu_max_pct",
    "image_count",
    "batch_total_seconds",
    "throughput_images_per_minute",
}

SUMMARY_METRICS = [
    "payload_mb",
    "split_latency",
    "edge_embedding_latency",
    "transfer_latency",
    "network_overhead",
    "cloud_total_request",
    "cloud_reconstruct_time",
    "cloud_prompt_time",
    "cloud_combine_time",
    "cloud_generate_time",
    "cloud_generate_tokens",
    "cloud_generate_prefill_time",
    "cloud_generate_decode_step_time",
    "cloud_generate_postprocess_time",
    "cloud_decoder_kv_cat_time",
    "cloud_decoder_kv_repeat_time",
    "cloud_decoder_attention_core_time",
    "cloud_decoder_mlp_time",
    "edge_cpu_total",
    "cloud_cpu_total",
]


def parse_float(value):
    if value is None or value == "":
        return None
    return float(value)


def fmt(value, digits=3):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def ms_per_token(time_value, token_count):
    if time_value is None or not token_count:
        return None
    return time_value / token_count * 1000


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def target_from_filename(path):
    name = os.path.basename(path)
    if "10_128_0_6" in name:
        return "10.128.0.6"
    if "10_128_0_7" in name:
        return "10.128.0.7"
    if "local" in name:
        return "local"
    return ""


def image_id(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def normalize_row(row, source_file):
    target = row.get("target") or target_from_filename(source_file)
    out = dict(row)
    out["source_file"] = os.path.basename(source_file)
    out["target"] = target
    out["cloud_type"] = TARGET_LABELS.get(target, "Unknown")
    out["image_id"] = image_id(row.get("image", ""))

    for column in NUMERIC_COLUMNS:
        if column in out:
            value = parse_float(out[column])
            out[column] = value

    if out.get("payload_bytes") is not None:
        out["payload_mb"] = out["payload_bytes"] / (1024 * 1024)
    return out


def load_metrics():
    rows = []
    for name in sorted(os.listdir(RECORD_DIR)):
        if not name.startswith("distributed_vlm_metrics_") or not name.endswith(".csv"):
            continue
        path = os.path.join(RECORD_DIR, name)
        rows.extend(normalize_row(row, path) for row in read_csv(path))
    return rows


def load_throughput():
    rows = []
    for name in sorted(os.listdir(RECORD_DIR)):
        if not name.startswith("distributed_vlm_throughput_") or not name.endswith(".csv"):
            continue
        path = os.path.join(RECORD_DIR, name)
        for row in read_csv(path):
            out = normalize_row(row, path)
            if out.get("image_count"):
                out["seconds_per_image"] = out["batch_total_seconds"] / out["image_count"]
            rows.append(out)
    return rows


def summarize(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(clean),
        "median": median(clean),
        "std": stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def grouped_summary(metrics):
    groups = defaultdict(list)
    for row in metrics:
        groups[(row["transport"], row["target"], row["cloud_type"])].append(row)

    summary_rows = []
    for (transport, target, cloud_type), rows in sorted(groups.items()):
        out = {
            "transport": transport,
            "target": target,
            "cloud_type": cloud_type,
            "n": len(rows),
        }
        for metric in SUMMARY_METRICS:
            stats = summarize([row.get(metric) for row in rows])
            for stat_name, value in stats.items():
                out[f"{metric}_{stat_name}"] = value
        summary_rows.append(out)
    return summary_rows


def paired_tdx_overhead(metrics):
    by_key = {}
    for row in metrics:
        by_key[(row["transport"], row["image_id"], row["target"])] = row

    rows = []
    for transport in sorted({row["transport"] for row in metrics}):
        pairs = []
        for image in sorted({row["image_id"] for row in metrics if row["transport"] == transport}):
            tdx = by_key.get((transport, image, "10.128.0.6"))
            normal = by_key.get((transport, image, "10.128.0.7"))
            if tdx and normal:
                pairs.append((tdx, normal))

        out = {"transport": transport, "paired_images": len(pairs)}
        for metric in [
            "split_latency",
            "transfer_latency",
            "network_overhead",
            "cloud_total_request",
            "cloud_reconstruct_time",
            "cloud_prompt_time",
            "cloud_combine_time",
            "cloud_generate_time",
            "cloud_generate_prefill_time",
            "cloud_generate_head_time",
            "cloud_generate_sample_time",
            "cloud_generate_token_embedding_time",
            "cloud_generate_attention_mask_time",
            "cloud_generate_decode_step_time",
            "cloud_generate_postprocess_time",
            "cloud_decoder_kv_cat_time",
            "cloud_decoder_kv_repeat_time",
            "cloud_decoder_attention_core_time",
            "cloud_decoder_mlp_time",
            "cloud_cpu_total",
            "edge_cpu_total",
        ]:
            deltas = [
                tdx[metric] - normal[metric]
                for tdx, normal in pairs
                if tdx.get(metric) is not None and normal.get(metric) is not None
            ]
            pct = [
                ((tdx[metric] - normal[metric]) / normal[metric] * 100)
                for tdx, normal in pairs
                if tdx.get(metric) is not None and normal.get(metric)
            ]
            out[f"{metric}_tdx_minus_normal_mean"] = mean(deltas) if deltas else None
            out[f"{metric}_tdx_vs_normal_pct_mean"] = mean(pct) if pct else None
        rows.append(out)
    return rows


def transport_overhead(metrics):
    groups = defaultdict(list)
    for row in metrics:
        groups[(row["target"], row["transport"])].append(row)

    by_target_transport = {}
    for key, rows in groups.items():
        by_target_transport[key] = {
            metric: summarize([row.get(metric) for row in rows])["mean"]
            for metric in ["split_latency", "transfer_latency", "network_overhead", "cloud_total_request"]
        }

    rows = []
    for target in sorted({row["target"] for row in metrics}):
        base = by_target_transport.get((target, "http"))
        if not base:
            continue
        for transport in ["https-aes-gcm", "https-chacha20"]:
            current = by_target_transport.get((target, transport))
            if not current:
                continue
            out = {
                "target": target,
                "cloud_type": TARGET_LABELS.get(target, "Unknown"),
                "transport": transport,
                "baseline_transport": "http",
            }
            for metric, base_value in base.items():
                delta = current[metric] - base_value
                out[f"{metric}_delta_vs_http"] = delta
                out[f"{metric}_pct_vs_http"] = delta / base_value * 100 if base_value else None
            rows.append(out)
    return rows


def throughput_comparison(rows):
    out_rows = []
    for row in sorted(rows, key=lambda r: (r["transport"], r["target"])):
        out_rows.append({
            "transport": row["transport"],
            "split": row["split"],
            "target": row["target"],
            "cloud_type": row["cloud_type"],
            "image_count": row["image_count"],
            "batch_total_seconds": row["batch_total_seconds"],
            "seconds_per_image": row["seconds_per_image"],
            "throughput_images_per_minute": row["throughput_images_per_minute"],
        })
    return out_rows


def write_report(metrics, summary, throughput, tdx_rows, transport_rows):
    total_images = len(metrics)
    transports = sorted({row["transport"] for row in metrics})
    targets = sorted({row["target"] for row in metrics})
    rows_per_file = sorted({
        sum(1 for item in metrics if item["source_file"] == source_file)
        for source_file in {row["source_file"] for row in metrics}
    })
    rows_per_file_text = ", ".join(str(count) for count in rows_per_file)

    fastest = min(summary, key=lambda row: row["split_latency_mean"])
    slowest = max(summary, key=lambda row: row["split_latency_mean"])
    best_throughput = max(throughput, key=lambda row: row["throughput_images_per_minute"])
    worst_throughput = min(throughput, key=lambda row: row["throughput_images_per_minute"])

    lines = [
        "# Distributed VLM Record Analysis",
        "",
        "## Dataset",
        "",
        f"- Metrics rows: {total_images}",
        f"- Transport modes: {', '.join(transports)}",
        f"- Targets: {', '.join(f'{target} ({TARGET_LABELS.get(target, "Unknown")})' for target in targets)}",
        f"- Metrics rows per file: {rows_per_file_text}",
        "",
        "## Key Findings",
        "",
        (
            f"- Fastest mean split latency: {fastest['transport']} on {fastest['cloud_type']} "
            f"({fastest['target']}), {fmt(fastest['split_latency_mean'])} s."
        ),
        (
            f"- Slowest mean split latency: {slowest['transport']} on {slowest['cloud_type']} "
            f"({slowest['target']}), {fmt(slowest['split_latency_mean'])} s."
        ),
        (
            f"- Best throughput: {best_throughput['transport']} on {best_throughput['cloud_type']} "
            f"({fmt(best_throughput['throughput_images_per_minute'])} images/min, "
            f"{fmt(best_throughput['seconds_per_image'])} s/image)."
        ),
        (
            f"- Lowest throughput: {worst_throughput['transport']} on {worst_throughput['cloud_type']} "
            f"({fmt(worst_throughput['throughput_images_per_minute'])} images/min, "
            f"{fmt(worst_throughput['seconds_per_image'])} s/image)."
        ),
        "",
        "## Mean Latency By Transport And Target",
        "",
        "| transport | cloud type | split s | edge embed s | transfer s | cloud request s | generated tokens | decoder generate s | prefill s | decode step s | decode ms/token | KV cat s | KV repeat s | attn core s | MLP s | network overhead ms | cloud CPU s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary:
        lines.append(
            "| {transport} | {cloud_type} | {split} | {edge} | {transfer} | {cloud} | {tokens} | {generate} | {prefill} | {decode_step} | {decode_ms_token} | {kv_cat} | {kv_repeat} | {attn_core} | {mlp} | {network_ms} | {cpu} |".format(
                transport=row["transport"],
                cloud_type=row["cloud_type"],
                split=fmt(row["split_latency_mean"]),
                edge=fmt(row["edge_embedding_latency_mean"]),
                transfer=fmt(row["transfer_latency_mean"]),
                cloud=fmt(row["cloud_total_request_mean"]),
                tokens=fmt(row.get("cloud_generate_tokens_mean"), 1),
                generate=fmt(row.get("cloud_generate_time_mean")),
                prefill=fmt(row.get("cloud_generate_prefill_time_mean")),
                decode_step=fmt(row.get("cloud_generate_decode_step_time_mean")),
                decode_ms_token=fmt(ms_per_token(
                    row.get("cloud_generate_decode_step_time_mean"),
                    row.get("cloud_generate_tokens_mean"),
                )),
                kv_cat=fmt(row.get("cloud_decoder_kv_cat_time_mean")),
                kv_repeat=fmt(row.get("cloud_decoder_kv_repeat_time_mean")),
                attn_core=fmt(row.get("cloud_decoder_attention_core_time_mean")),
                mlp=fmt(row.get("cloud_decoder_mlp_time_mean")),
                network_ms=fmt(row["network_overhead_mean"] * 1000),
                cpu=fmt(row["cloud_cpu_total_mean"]),
            )
        )

    lines.extend([
        "",
        "## TDX Overhead Compared With Normal VM",
        "",
        "| transport | paired images | split delta s | split delta % | cloud request delta s | cloud request delta % | cloud CPU delta s | cloud CPU delta % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in tdx_rows:
        lines.append(
            "| {transport} | {pairs} | {split_delta} | {split_pct}% | {cloud_delta} | {cloud_pct}% | {cpu_delta} | {cpu_pct}% |".format(
                transport=row["transport"],
                pairs=row["paired_images"],
                split_delta=fmt(row["split_latency_tdx_minus_normal_mean"]),
                split_pct=fmt(row["split_latency_tdx_vs_normal_pct_mean"]),
                cloud_delta=fmt(row["cloud_total_request_tdx_minus_normal_mean"]),
                cloud_pct=fmt(row["cloud_total_request_tdx_vs_normal_pct_mean"]),
                cpu_delta=fmt(row["cloud_cpu_total_tdx_minus_normal_mean"]),
                cpu_pct=fmt(row["cloud_cpu_total_tdx_vs_normal_pct_mean"]),
            )
        )

    lines.extend([
        "",
        "## HTTPS Overhead Compared With HTTP",
        "",
        "| cloud type | transport | split delta s | transfer delta s | network overhead delta ms | cloud request delta s |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in transport_rows:
        lines.append(
            "| {cloud_type} | {transport} | {split} | {transfer} | {network_ms} | {cloud} |".format(
                cloud_type=row["cloud_type"],
                transport=row["transport"],
                split=fmt(row["split_latency_delta_vs_http"]),
                transfer=fmt(row["transfer_latency_delta_vs_http"]),
                network_ms=fmt(row["network_overhead_delta_vs_http"] * 1000),
                cloud=fmt(row["cloud_total_request_delta_vs_http"]),
            )
        )

    lines.extend([
        "",
        "## Generated Files",
        "",
        "- metrics_combined.csv: one row per image request, with cloud_type and image_id added.",
        "- metrics_summary_by_transport_target.csv: mean/median/std/min/max by transport and target.",
        "- throughput_combined.csv: one row per throughput run, with seconds_per_image added.",
        "- tdx_overhead_vs_normal_vm.csv: paired-image TDX minus Normal VM deltas.",
        "- transport_overhead_vs_http.csv: HTTPS mode deltas against HTTP within the same target.",
    ])

    with open(os.path.join(OUTPUT_DIR, "analysis_report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = load_metrics()
    throughput = load_throughput()

    metric_fields = [
        "source_file",
        "timestamp",
        "image_id",
        "image",
        "transport",
        "target",
        "cloud_type",
        "payload_bytes",
        "payload_mb",
        "split_latency",
        "edge_embedding_latency",
        "transfer_latency",
        "network_overhead",
        "local_baseline_latency",
        "cloud_total_request",
        "cloud_reconstruct_time",
        "cloud_prompt_time",
        "cloud_combine_time",
        "cloud_generate_time",
        "cloud_generate_tokens",
        "cloud_generate_prefill_time",
        "cloud_generate_head_time",
        "cloud_generate_sample_time",
        "cloud_generate_token_embedding_time",
        "cloud_generate_attention_mask_time",
        "cloud_generate_decode_step_time",
        "cloud_generate_postprocess_time",
        "cloud_decoder_rotary_embd_time",
        "cloud_decoder_norm_time",
        "cloud_decoder_qkv_proj_time",
        "cloud_decoder_rotary_time",
        "cloud_decoder_kv_cat_time",
        "cloud_decoder_kv_repeat_time",
        "cloud_decoder_attention_mask_time",
        "cloud_decoder_attention_core_time",
        "cloud_decoder_out_proj_time",
        "cloud_decoder_mlp_time",
        "cloud_decoder_lm_head_time",
        "edge_embedding_cpu_total",
        "edge_embedding_cpu_avg_pct",
        "edge_embedding_cpu_max_pct",
        "edge_transfer_cpu_total",
        "local_baseline_cpu_total",
        "edge_cpu_total",
        "cloud_cpu_total",
    ]
    write_csv(os.path.join(OUTPUT_DIR, "metrics_combined.csv"), metrics, metric_fields)

    summary = grouped_summary(metrics)
    summary_fields = list(summary[0].keys()) if summary else []
    write_csv(os.path.join(OUTPUT_DIR, "metrics_summary_by_transport_target.csv"), summary, summary_fields)

    throughput_rows = throughput_comparison(throughput)
    throughput_fields = list(throughput_rows[0].keys()) if throughput_rows else []
    write_csv(os.path.join(OUTPUT_DIR, "throughput_combined.csv"), throughput_rows, throughput_fields)

    tdx_rows = paired_tdx_overhead(metrics)
    tdx_fields = list(tdx_rows[0].keys()) if tdx_rows else []
    write_csv(os.path.join(OUTPUT_DIR, "tdx_overhead_vs_normal_vm.csv"), tdx_rows, tdx_fields)

    transport_rows = transport_overhead(metrics)
    transport_fields = list(transport_rows[0].keys()) if transport_rows else []
    write_csv(os.path.join(OUTPUT_DIR, "transport_overhead_vs_http.csv"), transport_rows, transport_fields)

    write_report(metrics, summary, throughput_rows, tdx_rows, transport_rows)

    print(f"metrics rows: {len(metrics)}")
    print(f"throughput rows: {len(throughput_rows)}")
    print(f"analysis output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
