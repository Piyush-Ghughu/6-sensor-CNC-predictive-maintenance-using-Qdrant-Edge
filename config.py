# config.py

QDRANT_SHARD_PATH = "./qdrant_edge_shard"   # on-device shard (Qdrant Edge local storage)
VECTOR_NAME       = "cnc_vec"
VECTOR_SIZE       = 24                       # 6 sensors × 4 features
DISTANCE          = "Cosine"

TRAINING_STEPS    = 500    # pure normal data, no faults, builds Qdrant baseline
WARMUP_STEPS      = 500    # same as training — no detection during this phase
ANOMALY_THRESHOLD = 0.72   # cosine similarity below this → anomaly
SPIKE_Z_SCORE     = 3.0    # z-score multiplier for spike detection
BASELINE_WINDOW   = 150    # rolling window size for spike stats

TOTAL_STEPS       = 0      # 0 = run forever until Ctrl+C
SLEEP_INTERVAL    = 0.08   # seconds between steps
