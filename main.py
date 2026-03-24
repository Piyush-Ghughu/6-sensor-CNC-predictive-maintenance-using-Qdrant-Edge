#!/usr/bin/env python3
"""
Qdrant Edge Anomaly Detection
==============================
HTTP polling — browser hits /data every 500ms to get latest results.
No WebSocket needed.
"""

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import config
from sensors.multi_sensor import MultiSensorSimulator
from edge.feature_extractor import FeatureExtractor
from core.qdrant_engine import QdrantEdgeEngine
from intelligence.anomaly_engine import AnomalyDetector, AnomalyResult
from intelligence.spike_tracker import SpikeTracker

# ── shared state ─────────────────────────────────────────────────
_lock    = threading.Lock()
_history: list = []        # all payloads so far (browser replays on load)
_latest:  dict = {}        # most recent single payload


def _store(result: AnomalyResult, pattern_count: int):
    payload = {
        "step":          int(result.step),
        "temperature":   float(result.temperature),
        "vibration":     float(result.vibration),
        "pressure":      float(result.pressure),
        "similarity":    float(result.similarity),
        "anomaly_score": float(result.anomaly_score),
        "is_anomaly":    bool(result.is_anomaly),
        "spike":         bool(result.spike),
        "confidence":    float(result.confidence),
        "reason":        str(result.reason),
        "ground_truth":  bool(result.ground_truth),
        "gt_label":      result.gt_label,
        "pattern_count": int(pattern_count),
    }
    with _lock:
        _history.append(payload)
        if len(_history) > 500:
            _history.pop(0)
        _latest.update(payload)


# ── HTTP handler ─────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent

class Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass  # silence access logs

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str):
        try:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._file(PROJECT_DIR / "index.html", "text/html")

        elif self.path == "/data":
            # return ALL history so browser can replay from beginning
            with _lock:
                data = list(_history)
            self._json(data)

        elif self.path == "/latest":
            with _lock:
                data = dict(_latest)
            self._json(data)

        else:
            self.send_response(404)
            self.end_headers()


# ── detection loop ────────────────────────────────────────────────
def detection_loop():
    sensor    = MultiSensorSimulator(seed=42)
    extractor = FeatureExtractor()
    engine    = QdrantEdgeEngine(fresh=True)
    detector  = AnomalyDetector(engine)
    tracker   = SpikeTracker(maxlen=500)

    print(f"[engine] Qdrant Edge shard: {config.QDRANT_SHARD_PATH}")
    print(f"[engine] {config.VECTOR_SIZE}D vectors | warmup={config.WARMUP_STEPS} | threshold={config.ANOMALY_THRESHOLD}")

    def _steps():
        if config.TOTAL_STEPS == 0:
            while True:
                yield
        else:
            yield from range(config.TOTAL_STEPS)

    try:
        for _ in _steps():
            reading = sensor.read()
            vector  = extractor.ingest(reading)
            if vector is None:
                time.sleep(config.SLEEP_INTERVAL)
                continue

            result = detector.process(vector, reading)
            tracker.record(result)
            _store(result, engine.pattern_count)

            if result.is_anomaly:
                print(f"  ANOMALY step={result.step:4d} score={result.anomaly_score:.4f} {result.reason}")

            time.sleep(config.SLEEP_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        engine.flush()
        engine.close()

    print(f"\n[done] steps={detector._step} anomalies={tracker.total_anomalies} precision={tracker.precision:.0%}")


# ── main ──────────────────────────────────────────────────────────
def main():
    # start detection thread
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    # start HTTP server
    server = HTTPServer(("localhost", 8766), Handler)
    print("⚡ Qdrant Edge Anomaly Detection")
    print("   Dashboard : http://localhost:8766")
    print("   (open in browser — no WebSocket needed)")
    print()
    webbrowser.open("http://localhost:8766")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
