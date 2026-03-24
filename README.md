# Qdrant Edge — CNC Anomaly Detection

> Real-time predictive maintenance for CNC machines using **Qdrant Edge** — a fully embedded, serverless vector database running entirely on-device with zero cloud dependency.

---

## What Is This?

This system monitors a CNC industrial machine in real time using **6 sensors**, detects faults as they happen, and shows everything on a live web dashboard — no internet, no cloud server, no external database.

**Qdrant Edge** is the core. It runs as an embedded vector database inside the Python process itself. Every similarity search happens on-device in milliseconds.

---

## Why Qdrant Edge?

| | Traditional Setup | This Project |
|---|---|---|
| Vector DB | Remote cloud / server | Embedded on the device |
| Network | Required | Not needed |
| Latency | 10–500ms | Sub-millisecond |
| Data privacy | Data leaves machine | Data never leaves |
| Deployment | Complex infra | Single Python process |

---

## How It Works

Every 80ms the system reads 6 sensors from the CNC machine:

- **Spindle Current** — how hard the motor is working
- **Servo Torque** — force on the machine arms
- **Coolant Flow** — is cooling water flowing properly
- **Acoustic Emission** — cracking or grinding sounds
- **Feed Rate Deviation** — is the machine moving at correct speed
- **Thermal Gradient** — is the spindle overheating

These 6 values are converted into a **24-dimensional vector** and sent to Qdrant Edge.

**Warmup (first 120 readings):** Qdrant Edge stores these as "normal" patterns — building a baseline of healthy machine behavior.

**After warmup:** Every new reading is compared to stored patterns using cosine similarity:
- Similarity close to 1.0 → machine is normal → store the pattern
- Similarity below 0.65 → pattern never seen before → **ANOMALY**
- Sudden statistical spike (z-score > 3.0) → **ANOMALY**

When a fault is detected, the system identifies **exactly which sensor caused it** and raises an alert.

---

## Fault Types Detected

| Fault | What Happens |
|---|---|
| Tool Wear | Spindle current rises, acoustic noise increases gradually |
| Tool Fracture | All sensors spike hard simultaneously |
| Coolant Blockage | Coolant flow drops, temperature rises |
| Bearing Fault | Servo torque and acoustic emission oscillate |
| Thermal Runaway | Thermal gradient builds up over time |
| Feed Rate Fault | Feed deviation spikes, servo torque rises |
| Acoustic Crack | Acoustic emission spikes instantly |
| Multi Fault | Multiple sensors fail at once |

---

## Project Structure

```
qdrant_edge_anomaly_detection/
├── main.py                  # Entry point — HTTP server + detection loop
├── config.py                # All tunable parameters
├── index.html               # Live web dashboard
├── sensors/
│   └── cnc_sensor.py        # CNC machine simulator (6 sensors, random faults)
├── edge/
│   └── cnc_extractor.py     # Feature extraction → 24-dim vector
├── core/
│   └── qdrant_engine.py     # Qdrant Edge wrapper (store, search)
├── intelligence/
│   ├── anomaly_engine.py    # Dual-layer anomaly detector
│   └── spike_tracker.py     # Precision tracking
└── qdrant_edge_shard/       # Local vector database (auto-created)
```

---

## Getting Started

**Linux / Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Using uv:**
```bash
uv add qdrant-edge-py numpy rich
uv run main.py
```

Open `http://localhost:8766` in your browser. The dashboard connects automatically.

Press `Ctrl+C` to stop.

---

## Configuration

Edit `config.py` to tune the system:

| Parameter | Default | Meaning |
|---|---|---|
| `WARMUP_STEPS` | `120` | Readings before detection starts |
| `ANOMALY_THRESHOLD` | `0.65` | Similarity below this = anomaly |
| `SPIKE_Z_SCORE` | `3.0` | Statistical spike sensitivity |
| `SLEEP_INTERVAL` | `0.08` | Seconds between sensor reads |
| `TOTAL_STEPS` | `0` | 0 = run forever |

---

## Tech Stack

| Component | Technology |
|---|---|
| Vector Database | Qdrant Edge (embedded, serverless) |
| Anomaly Detection | Cosine similarity + Z-score spike |
| Dashboard | HTML + CSS + Chart.js |
| Web Server | Python `http.server` |
