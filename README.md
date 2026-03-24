# Qdrant Edge — CNC Anomaly Detection System

> **Real-time predictive maintenance for CNC machines using Qdrant Edge — a fully embedded, serverless vector database running entirely on-device with zero cloud dependency.**

---

## What Is This Project?

This is an **edge-native AI system** that monitors a CNC (Computer Numerical Control) industrial machine in real time, detects anomalies as they happen, and displays everything on a live web dashboard — all without touching the internet or a cloud server.

At its core, it uses **Qdrant Edge** (an embedded vector database) to build a library of "what normal looks like" from 6 industrial sensors, then flags anything that deviates from that library using a dual-layer detection algorithm. When a fault is detected — tool wear, bearing failure, coolant blockage, thermal runaway — the system raises an alert in milliseconds, displays it on the dashboard, and logs it for review.

---

## Why Qdrant Edge?

Most anomaly detection systems need a central server to query a vector database. **Qdrant Edge changes that entirely.**

| Feature | Traditional Setup | This Project (Qdrant Edge) |
|---|---|---|
| Vector DB location | Remote cloud / server | Embedded on the edge device |
| Network required | Yes | No |
| Latency | 10–500ms round-trip | Sub-millisecond (local disk) |
| Privacy | Data leaves the machine | Data never leaves the edge |
| Deployment | Complex infra | Single Python process |
| Persistence | Managed service | Local shard on disk |

Qdrant Edge runs as a **local file-based shard** (`qdrant_edge_shard/`). Every vector query, upsert, and similarity search happens in-process — no HTTP, no gRPC, no server process. This makes it ideal for factory floors, embedded systems, and air-gapped environments.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTEM OVERVIEW                             │
│                                                                     │
│   CNC Machine Simulator                                             │
│   ┌──────────────────────┐                                          │
│   │  6 Sensor Streams    │  80ms sampling (~12.5 Hz)               │
│   │  • Spindle Current   │                                          │
│   │  • Servo Torque      │                                          │
│   │  • Coolant Flow      │──────────────────────────────────────┐   │
│   │  • Acoustic Emission │                                       │   │
│   │  • Feed Rate Dev.    │                                       ▼   │
│   │  • Thermal Gradient  │                          CNCFeatureExtractor│
│   └──────────────────────┘                          (Online Welford │
│                                                      Normalization) │
│                                                           │         │
│                                                    24-dim L2-norm   │
│                                                      vector         │
│                                                           │         │
│                                              ┌────────────▼────────┐│
│                                              │   Qdrant Edge       ││
│                                              │   (Embedded DB)     ││
│                                              │   cosine similarity ││
│                                              │   search vs stored  ││
│                                              │   normal patterns   ││
│                                              └────────────┬────────┘│
│                                                           │         │
│                                                     similarity      │
│                                                      score          │
│                                                           │         │
│                                              ┌────────────▼────────┐│
│                                              │  AnomalyDetector    ││
│                                              │  Layer 1: similarity││
│                                              │  Layer 2: Z-score   ││
│                                              │  spike detection    ││
│                                              └────────────┬────────┘│
│                                                           │         │
│                           ┌───────────────────────────────┘         │
│                           │  is_anomaly / reason / confidence       │
│                           ▼                                         │
│              ┌──────────────────────┐    ┌────────────────────────┐ │
│              │  Thread-safe State   │───▶│  HTTP Server :8766     │ │
│              │  _history (600 pts)  │    │  /data  /latest        │ │
│              │  _alerts  (50 max)   │    │  /alerts               │ │
│              │  _latest             │    └──────────┬─────────────┘ │
│              └──────────────────────┘               │               │
│                                                     ▼               │
│                                          ┌──────────────────────┐   │
│                                          │  index.html          │   │
│                                          │  Live Dashboard      │   │
│                                          │  8 real-time charts  │   │
│                                          │  Alert notifications │   │
│                                          │  Sensor gauges       │   │
│                                          └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
qdrant_edge_anomaly_detection/
│
├── main.py                    # Entry point — HTTP server + detection loop
├── config.py                  # All tunable parameters
├── index.html                 # Live web dashboard (HTML + CSS + JS)
│
├── sensors/
│   └── cnc_sensor.py          # CNC machine simulator (6 sensors, 8 fault types)
│
├── edge/
│   ├── cnc_extractor.py       # Online Welford feature extraction → 24-dim vector
│   └── vectorizer.py          # Unused FFT vectorizer (legacy)
│
├── core/
│   └── qdrant_engine.py       # Qdrant Edge wrapper (store, search, flush)
│
├── intelligence/
│   ├── anomaly_engine.py      # Dual-layer anomaly detector
│   └── spike_tracker.py       # TP/FP metrics and precision tracking
│
├── qdrant_edge_shard/         # Persistent local vector database (auto-created)
│   └── segments/{uuid}/
│       ├── vector_storage-cnc_vec/
│       └── payload_storage/
│
└── ui/                        # Legacy placeholder (empty)
```

---

## How It Works — Step by Step

### Step 1 — Sensor Reading (every 80ms)

`CNCMachineSimulator` generates synthetic CNC sensor readings with **realistic physics-based signals** — sinusoidal oscillations, random noise, and periodic fault injections:

```
CNCReading {
    spindle_current      float   # Motor current (A)
    servo_torque         float   # Axis torque (Nm)
    coolant_flow         float   # Coolant flow rate (L/min)
    acoustic_emission    float   # Vibration / noise (kHz)
    feed_rate_deviation  float   # Feed rate error (%)
    thermal_gradient     float   # Temperature rise (°C)
    ground_truth_anomaly bool    # Was a fault actually injected?
    anomaly_label        str     # Fault type name (if any)
}
```

### Step 2 — Feature Extraction (24-dim vector)

`CNCFeatureExtractor` maintains a **rolling window of 10 samples per sensor** and applies **Welford's online normalization** to extract 4 features per sensor:

| Feature Index | What It Captures |
|---|---|
| `feature_0` | Window mean, normalized |
| `feature_1` | Window std dev, normalized |
| `feature_2` | Window max, normalized |
| `feature_3` | Rate of change (latest − previous), normalized |

6 sensors × 4 features = **24-dimensional vector**, then **L2-normalized** to unit length. This enables pure cosine similarity queries in Qdrant Edge.

**Welford's Online Algorithm** (numerically stable, single-pass, O(1) memory):
```
n += 1
delta = x − mean
mean += delta / n
M2  += delta * (x − mean)
std  = sqrt(M2 / (n−1))
```

### Step 3 — Warmup Phase (first 120 steps)

For the first 120 steps (~9.6 seconds), every extracted vector is **stored directly into Qdrant Edge** as a "normal pattern" — no anomaly detection runs. This builds the initial baseline of what healthy CNC operation looks like.

### Step 4 — Dual-Layer Anomaly Detection

After warmup, every incoming vector is checked against two independent detection layers:

#### Layer 1 — Cosine Similarity Threshold
```
similarity = qdrant_edge.search(vector, top_k=5)   # [0, 1]
anomaly_score = 1.0 - similarity

if similarity < 0.65:
    → ANOMALY (pattern never seen before)
```

#### Layer 2 — Z-Score Spike Detection
```
z_score = (anomaly_score − rolling_mean) / rolling_std
                                   ↑
                    rolling window of last 100 scores

if z_score > 3.0:
    → ANOMALY (sudden statistical jump)
```

**Final decision:**
```
is_anomaly = (similarity < 0.65) OR (z_score > 3.0)
```

This dual approach catches both **absolute anomalies** (never-seen patterns) and **relative anomalies** (sudden jumps in otherwise familiar behavior).

### Step 5 — Fault Diagnosis

When an anomaly is detected, per-sensor threshold checks generate a human-readable reason:

| Sensor | Threshold | Interpretation |
|---|---|---|
| Spindle Current | > 5.5 A | Motor load high (normal ~4.5A) |
| Servo Torque | > 16.0 Nm | Axis stress high (normal ~12Nm) |
| Coolant Flow | < 6.0 L/min | Flow dropping (normal ~8.5) |
| Acoustic Emission | > 60.0 kHz | Abnormal noise (normal ~47kHz) |
| Feed Rate Deviation | > 1.0% | Controller off (normal ~0.2%) |
| Thermal Gradient | > 4.0°C | Heat building (normal ~2.2°C) |

If no threshold is breached, it reports the most deviated sensor with its percentage deviation.

### Step 6 — Pattern Database Growth

Only **normal vectors** (non-anomalous) are stored in Qdrant Edge after warmup. Anomalous vectors are discarded. This keeps the pattern library clean — it represents exclusively healthy machine behavior. The database grows continuously, making the system smarter over time.

### Step 7 — Dashboard & Alerts

The HTTP server on port **8766** serves the live dashboard and three data endpoints:

| Endpoint | Content | Poll Rate |
|---|---|---|
| `GET /` | index.html dashboard | — |
| `GET /data` | Last 600 readings as JSON | 500ms |
| `GET /latest` | Single newest reading | on demand |
| `GET /alerts` | Up to 50 alerts as JSON | 800ms |

---

## CNC Fault Simulation

The simulator injects **8 realistic fault types** with distinct sensor signatures:

| Fault | Duration | Key Signatures |
|---|---|---|
| `tool_wear` | 20 steps | Spindle +3.5A, Servo +5Nm, Acoustic +40kHz, Thermal +3°C |
| `tool_fracture` | 1 step (spike) | Spindle +9A, Servo +22Nm, Acoustic +90kHz, Thermal +6°C, Feed +4% |
| `coolant_block` | 15 steps | Coolant −5.5 L/min, Thermal +5°C |
| `bearing_fault` | 25 steps | Servo +7Nm, Acoustic +30kHz, Thermal +3°C |
| `thermal_runaway` | 20 steps | Thermal +10°C, Spindle +2A |
| `feed_fault` | 3 steps | Servo +8Nm, Feed Rate +4.5% |
| `acoustic_crack` | 1 step (spike) | Acoustic +70kHz (instantaneous) |
| `multi_fault` | 2 steps | Spindle +6A, Servo +15Nm, Acoustic +50kHz, Thermal +7°C, Coolant −3.5 |

Each fault **linearly ramps up** over its duration and applies multiplicative noise. Fault probability is **0.08% per step** (~1 fault every ~2 minutes of operation).

**Healthy baseline signals** use sinusoidal oscillations at multiple frequencies to mimic real CNC dynamics:

| Sensor | Base Value | Oscillations | Noise |
|---|---|---|---|
| Spindle Current | 4.5 A | ±0.3A (1.5s), ±0.1A (0.3s) | ±0.06A |
| Servo Torque | 12.0 Nm | ±0.8Nm (3.0s), ±0.3Nm (0.8s) | ±0.15Nm |
| Coolant Flow | 8.5 L/min | slow drift (20s) | ±0.05 |
| Acoustic Emission | 47.0 kHz | ±3.0kHz (0.5s) | ±1.2kHz |
| Feed Rate Dev. | 0.2% | modulation (5s) | bounded |
| Thermal Gradient | 2.2°C | very slow drift (40s) | ±0.08°C |

---

## Live Dashboard

Open `http://localhost:8766` after starting the system.

### Left Panel
- **Machine Status** — WARMUP (yellow) / NORMAL (green) / FAULT DETECTED (red, animated)
- **6 Sensor Gauges** — Live values with color-coded danger highlighting
- **Detection Stats** — Step count, patterns in Qdrant Edge, similarity, anomaly score, confidence
- **Event Log** — Scrollable timestamped log of anomalies and normal samples

### Center Panel — 8 Real-Time Charts

| Chart | Color | Range |
|---|---|---|
| Anomaly Score | Purple | 0 – 1 |
| Qdrant Similarity | Cyan | 0 – 1 |
| Spindle Current | Blue | 2 – 15 A |
| Servo Torque | Purple | 5 – 40 Nm |
| Coolant Flow | Cyan | 0 – 10 L/min |
| Acoustic Emission | Yellow | 30 – 160 kHz |
| Feed Rate Deviation | Green | 0 – 6% |
| Thermal Gradient | Orange | 0 – 18°C |

Each chart shows the last 200 data points and marks anomaly spikes as red dots.

### Right Panel
- **Alert Notifications** — CRITICAL (score > 0.6) or WARNING, with full sensor values
- **Precision Ring** — SVG circular indicator: TP / Total anomalies (green > 70%, yellow > 40%, red < 40%)
- **Stats Grid** — Anomalies detected, True Positives, False Positives, Patterns stored
- **Last Fault Box** — Step, fault type, score, key sensor values

### Toast Notifications
- Slide-in popups (top-right) on every anomaly detection
- Auto-dismiss after 4 seconds
- Red border = CRITICAL, Yellow border = WARNING

---

## Configuration (`config.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `QDRANT_SHARD_PATH` | `./qdrant_edge_shard` | Where the local vector DB lives |
| `VECTOR_NAME` | `cnc_vec` | Named vector field in the shard |
| `VECTOR_SIZE` | `24` | 6 sensors × 4 features |
| `DISTANCE` | `Cosine` | Similarity metric |
| `WARMUP_STEPS` | `120` | Steps before detection starts (~9.6s) |
| `ANOMALY_THRESHOLD` | `0.65` | Cosine similarity below this = anomaly |
| `SPIKE_Z_SCORE` | `3.0` | Z-score above this = statistical spike |
| `BASELINE_WINDOW` | `100` | Rolling window size for spike detection |
| `SLEEP_INTERVAL` | `0.08` | Seconds between sensor reads (80ms) |
| `TOTAL_STEPS` | `0` | 0 = run forever |

---

## Qdrant Edge — Deep Dive

### What Is Qdrant Edge?

Qdrant Edge is the **embedded, serverless variant of Qdrant** — the high-performance vector database. Instead of running as a separate server process, it runs **inside your Python process** and reads/writes directly to a local file shard.

### How It Is Used Here

```python
# Initialize (EdgeConfig specifies vector schema)
engine = QdrantEdgeEngine(fresh=True)
# fresh=True → delete old shard, start clean

# Store a normal pattern
engine.store(vector)          # upsert Point with monotonic uint ID

# Search for nearest patterns
similarity = engine.search(vector, top_k=5)   # cosine similarity [0, 1]

# Persist on shutdown
engine.flush()
engine.close()
```

### Shard Layout on Disk

```
qdrant_edge_shard/
├── edge_config.json               # Vector schema (size=24, distance=Cosine)
└── segments/
    └── {uuid}/
        ├── segment.json           # Segment metadata
        ├── vector_storage-cnc_vec/ # The actual 24-dim vectors (mmap)
        ├── payload_storage/        # Point payloads (unused here)
        └── payload_index/          # Index metadata
```

### Vector Schema

```
Named vector: "cnc_vec"
Dimensions:   24
Distance:     Cosine
Point ID:     uint (monotonic counter, incremented per store)
Payload:      empty (anomaly metadata stored in app state, not DB)
```

### Search Semantics

```
Query: Query.Nearest(vector)  →  top-K cosine neighbors
Result: [(id, score), ...]    →  score ∈ [0, 1] (Qdrant normalizes cosine to positive)

score ≈ 1.0 → very similar to stored normal patterns → NORMAL
score ≈ 0.0 → very different from all stored patterns → ANOMALY
```

---

## Key Algorithms

### Welford's Online Normalization

**Problem**: Normalize streaming sensor data without storing all history.

**Algorithm** (per sensor, per step):
```
n     += 1
delta  = x − mean
mean  += delta / n
M2    += delta × (x − mean)
std    = sqrt(M2 / (n−1))

normalized = (x − mean) / max(std, 1e-6)
```

**Why this matters**:
- Single-pass, no batch required
- Numerically stable (Knuth/Welford method)
- O(1) memory regardless of stream length
- Prevents division-by-zero with 1e-6 floor on std

### Z-Score Spike Detection

```
history = deque of last 100 anomaly_scores

z_score = (current_score − mean(history)) / std(history)
spike   = z_score > 3.0
```

Example: if recent anomaly scores average 0.05 with std 0.01, a score of 0.35 gives z=30 → instant spike.

### Confidence Scoring

```python
if low_similarity:
    confidence = (THRESHOLD − similarity) / THRESHOLD     # distance from threshold

if z_spike:
    confidence = max(confidence, min(1.0, z_score / 6.0)) # scaled z-score

# confidence ∈ [0, 1]
# CRITICAL if anomaly_score > 0.6
# WARNING  if anomaly_score ≤ 0.6
```

---

## Getting Started

### Requirements

```
Python 3.10+
qdrant-edge-py
numpy
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

This will:
1. Delete any existing `qdrant_edge_shard/` and create a fresh one
2. Start the detection loop in a background daemon thread
3. Start the HTTP server on `localhost:8766`
4. Open the dashboard in your browser automatically

### Stop

Press `Ctrl+C`. The system will:
- Flush the Qdrant Edge shard to disk
- Close the shard cleanly
- Print a summary: `steps | anomalies | precision`

---

## Data Flow Summary

```
Every 80ms:
  CNCMachineSimulator.read()
      → CNCReading (6 floats + fault ground truth)
      → CNCFeatureExtractor.ingest()
          → Welford normalization
          → 4 features × 6 sensors = 24-dim vector
          → L2 normalize to unit sphere
      → AnomalyDetector.process(vector)
          → Qdrant Edge cosine search → similarity score
          → Layer 1: similarity < 0.65?
          → Layer 2: z-score > 3.0?
          → Per-sensor threshold diagnosis
          → AnomalyResult {is_anomaly, reason, confidence, ...}
      → SpikeTracker.record(result)
          → TP/FP counter update
      → _store(reading, result)
          → Thread-safe append to _history
          → Create alert entry if anomaly
          → Update _latest

Every 500ms (browser):
  GET /data → _history → render 8 charts + gauges + stats

Every 800ms (browser):
  GET /alerts → _alerts → update alert list + show toasts
```

---

## Metrics & Evaluation

Because the simulator injects faults with ground truth labels, the system tracks **real-time precision**:

```
Precision = True Positives / Total Anomalies Detected

True Positive  = anomaly detected AND ground_truth_anomaly = True
False Positive = anomaly detected AND ground_truth_anomaly = False
```

This is displayed on the dashboard as:
- The **Precision Ring** (SVG circular indicator)
- The **Stats Grid** (TP, FP, total count)

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Cosine distance over Euclidean | L2-normalized vectors make cosine = dot product; angle-based similarity is scale-invariant |
| Only store normal vectors | Keeps pattern DB clean; anomalies are rare outliers, not useful as search targets |
| 24-dim vector (6×4) | Enough expressiveness per sensor without excessive dimensionality |
| Warmup = 120 steps | Short enough for demos (~10s), long enough to learn stable baseline |
| Threshold = 0.65 | Conservative: requires 65% similarity before calling normal |
| Z-score = 3.0 | Standard 3σ rule — <0.3% false positive rate for normally distributed scores |
| 80ms sampling | Real-time for CNC monitoring (~12.5 Hz); comfortable for Python GIL |
| HTTP polling not WebSocket | Simpler, no extra dependencies; 500ms poll latency acceptable for dashboard |
| Fresh shard on start | Reproducible baseline every run using seed=42 simulator |

---



## Technology Stack

| Component | Technology |
|---|---|
| Vector Database | Qdrant Edge (embedded, serverless) |
| Feature Extraction | Custom online Welford normalization |
| Anomaly Detection | Cosine similarity + Z-score spike |
| Sensor Simulation | Physics-based Python simulator |
| Web Server | Python `http.server.BaseHTTPRequestHandler` |
| Dashboard | Vanilla HTML/CSS/JS + Chart.js |
| Threading | Python `threading` (daemon thread + Lock) |
| Persistence | Local file shard (mmap'd by Qdrant Edge) |
