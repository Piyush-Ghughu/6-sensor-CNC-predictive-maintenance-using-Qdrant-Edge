# config.py

QDRANT_SHARD_PATH = "./qdrant_edge_shard"   # on-device shard (Qdrant Edge local storage)
VECTOR_NAME       = "cnc_vec"
VECTOR_SIZE       = 24                       # 6 sensors × 4 features
DISTANCE          = "Cosine"

WARMUP_STEPS      = 120    # collect baseline before anomaly detection kicks in
ANOMALY_THRESHOLD = 0.65   # cosine similarity below this → anomaly
SPIKE_Z_SCORE     = 3.0    # z-score multiplier for spike detection
BASELINE_WINDOW   = 100    # rolling window size for spike stats

TOTAL_STEPS       = 0      # 0 = run forever until Ctrl+C
SLEEP_INTERVAL    = 0.08   # seconds between steps
