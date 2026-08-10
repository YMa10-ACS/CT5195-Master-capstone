"""Metric names and CSV schemas shared by the edge and cloud applications."""

TIMESTAMP = "timestamp"
IMAGE = "image"
TRANSPORT = "transport"
PAYLOAD_BYTES = "payload_bytes"
SPLIT_LATENCY = "split_latency"
EDGE_EMBEDDING_LATENCY = "edge_embedding_latency"
TRANSFER_LATENCY = "transfer_latency"
NETWORK_OVERHEAD = "network_overhead"
LOCAL_BASELINE_LATENCY = "local_baseline_latency"

CLOUD_TOTAL_REQUEST = "cloud_total_request"
CLOUD_RECONSTRUCT_TIME = "cloud_reconstruct_time"
CLOUD_PROMPT_TIME = "cloud_prompt_time"
CLOUD_COMBINE_TIME = "cloud_combine_time"
CLOUD_GENERATE_TIME = "cloud_generate_time"
CLOUD_GENERATE_TOKENS = "cloud_generate_tokens"
CLOUD_GENERATE_PREFILL_TIME = "cloud_generate_prefill_time"
CLOUD_GENERATE_HEAD_TIME = "cloud_generate_head_time"
CLOUD_GENERATE_SAMPLE_TIME = "cloud_generate_sample_time"
CLOUD_GENERATE_TOKEN_EMBEDDING_TIME = "cloud_generate_token_embedding_time"
CLOUD_GENERATE_ATTENTION_MASK_TIME = "cloud_generate_attention_mask_time"
CLOUD_GENERATE_DECODE_STEP_TIME = "cloud_generate_decode_step_time"
CLOUD_GENERATE_POSTPROCESS_TIME = "cloud_generate_postprocess_time"

CLOUD_DECODER_ROTARY_EMBD_TIME = "cloud_decoder_rotary_embd_time"
CLOUD_DECODER_NORM_TIME = "cloud_decoder_norm_time"
CLOUD_DECODER_QKV_PROJ_TIME = "cloud_decoder_qkv_proj_time"
CLOUD_DECODER_ROTARY_TIME = "cloud_decoder_rotary_time"
CLOUD_DECODER_KV_CAT_TIME = "cloud_decoder_kv_cat_time"
CLOUD_DECODER_KV_REPEAT_TIME = "cloud_decoder_kv_repeat_time"
CLOUD_DECODER_ATTENTION_MASK_TIME = "cloud_decoder_attention_mask_time"
CLOUD_DECODER_ATTENTION_CORE_TIME = "cloud_decoder_attention_core_time"
CLOUD_DECODER_OUT_PROJ_TIME = "cloud_decoder_out_proj_time"
CLOUD_DECODER_MLP_TIME = "cloud_decoder_mlp_time"
CLOUD_DECODER_LM_HEAD_TIME = "cloud_decoder_lm_head_time"

EDGE_EMBEDDING_CPU_TOTAL = "edge_embedding_cpu_total"
EDGE_EMBEDDING_CPU_AVG_PCT = "edge_embedding_cpu_avg_pct"
EDGE_EMBEDDING_CPU_MAX_PCT = "edge_embedding_cpu_max_pct"
EDGE_TRANSFER_CPU_TOTAL = "edge_transfer_cpu_total"
LOCAL_BASELINE_CPU_TOTAL = "local_baseline_cpu_total"
EDGE_GENERATE_CPU_AVG_PCT = "edge_generate_cpu_avg_pct"
EDGE_GENERATE_CPU_MAX_PCT = "edge_generate_cpu_max_pct"
CLOUD_CPU_TOTAL = "cloud_cpu_total"
CLOUD_CPU_AVG_PCT = "cloud_cpu_avg_pct"
CLOUD_CPU_MAX_PCT = "cloud_cpu_max_pct"

SPLIT = "split"
TARGET = "target"
IMAGE_COUNT = "image_count"
BATCH_TOTAL_SECONDS = "batch_total_seconds"
THROUGHPUT_IMAGES_PER_MINUTE = "throughput_images_per_minute"

METRICS_COLUMNS = [
    TIMESTAMP,
    IMAGE,
    TRANSPORT,
    PAYLOAD_BYTES,
    SPLIT_LATENCY,
    EDGE_EMBEDDING_LATENCY,
    TRANSFER_LATENCY,
    NETWORK_OVERHEAD,
    LOCAL_BASELINE_LATENCY,
    CLOUD_TOTAL_REQUEST,
    CLOUD_RECONSTRUCT_TIME,
    CLOUD_PROMPT_TIME,
    CLOUD_COMBINE_TIME,
    CLOUD_GENERATE_TIME,
    CLOUD_GENERATE_TOKENS,
    CLOUD_GENERATE_PREFILL_TIME,
    CLOUD_GENERATE_HEAD_TIME,
    CLOUD_GENERATE_SAMPLE_TIME,
    CLOUD_GENERATE_TOKEN_EMBEDDING_TIME,
    CLOUD_GENERATE_ATTENTION_MASK_TIME,
    CLOUD_GENERATE_DECODE_STEP_TIME,
    CLOUD_GENERATE_POSTPROCESS_TIME,
    CLOUD_DECODER_ROTARY_EMBD_TIME,
    CLOUD_DECODER_NORM_TIME,
    CLOUD_DECODER_QKV_PROJ_TIME,
    CLOUD_DECODER_ROTARY_TIME,
    CLOUD_DECODER_KV_CAT_TIME,
    CLOUD_DECODER_KV_REPEAT_TIME,
    CLOUD_DECODER_ATTENTION_MASK_TIME,
    CLOUD_DECODER_ATTENTION_CORE_TIME,
    CLOUD_DECODER_OUT_PROJ_TIME,
    CLOUD_DECODER_MLP_TIME,
    CLOUD_DECODER_LM_HEAD_TIME,
    EDGE_EMBEDDING_CPU_TOTAL,
    EDGE_EMBEDDING_CPU_AVG_PCT,
    EDGE_EMBEDDING_CPU_MAX_PCT,
    EDGE_TRANSFER_CPU_TOTAL,
    LOCAL_BASELINE_CPU_TOTAL,
    EDGE_GENERATE_CPU_AVG_PCT,
    EDGE_GENERATE_CPU_MAX_PCT,
    CLOUD_CPU_TOTAL,
    CLOUD_CPU_AVG_PCT,
    CLOUD_CPU_MAX_PCT,
]

THROUGHPUT_COLUMNS = [
    TRANSPORT,
    SPLIT,
    TARGET,
    IMAGE_COUNT,
    BATCH_TOTAL_SECONDS,
    THROUGHPUT_IMAGES_PER_MINUTE,
]
