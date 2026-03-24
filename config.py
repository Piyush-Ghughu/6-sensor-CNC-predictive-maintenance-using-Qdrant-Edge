# config.py

QDRANT_SHARD_PATH = "./qdrant_edge_shard"   # on-device shard (Qdrant Edge local storage)
VECTOR_NAME       = "sensor_vec"
VECTOR_SIZE       = 12                       # 3 sensors × 4 features
DISTANCE          = "Cosine"

WARMUP_STEPS      = 40     # collect baseline before anomaly detection kicks in
ANOMALY_THRESHOLD = 0.72   # cosine similarity below this → anomaly
SPIKE_Z_SCORE     = 2.5    # z-score multiplier for spike detection
BASELINE_WINDOW   = 60     # rolling window size for spike stats

TOTAL_STEPS       = 0      # 0 = run forever until Ctrl+C
SLEEP_INTERVAL    = 0.08   # seconds between steps
