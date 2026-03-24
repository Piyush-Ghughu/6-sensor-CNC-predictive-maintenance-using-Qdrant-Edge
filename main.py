#!/usr/bin/env python3
"""
Qdrant Edge — CNC Machine Anomaly Detection
============================================
6-sensor CNC predictive maintenance using Qdrant Edge (qdrant-edge-py).
HTTP polling dashboard. No WebSocket needed.
"""

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import config
from sensors.cnc_sensor import CNCMachineSimulator, CNCReading
from edge.cnc_extractor import CNCFeatureExtractor
from core.qdrant_engine import QdrantEdgeEngine
from intelligence.anomaly_engine import AnomalyDetector
from intelligence.spike_tracker import SpikeTracker

# ── shared state ──────────────────────────────────────────────────────
_lock      = threading.Lock()
_history:  list = []
_alerts:   list = []   # notification queue
_latest:   dict = {}


def _store(reading: CNCReading, result, pattern_count: int):
    payload = {
        "step":                 int(result.step),
        # CNC sensors
        "spindle_current":      float(reading.spindle_current),
        "servo_torque":         float(reading.servo_torque),
        "coolant_flow":         float(reading.coolant_flow),
        "acoustic_emission":    float(reading.acoustic_emission),
        "feed_rate_deviation":  float(reading.feed_rate_deviation),
        "thermal_gradient":     float(reading.thermal_gradient),
        # anomaly fields
        "similarity":           float(result.similarity),
        "anomaly_score":        float(result.anomaly_score),
        "is_anomaly":           bool(result.is_anomaly),
        "spike":                bool(result.spike),
        "confidence":           float(result.confidence),
        "reason":               str(result.reason),
        "ground_truth":         bool(result.ground_truth),
        "gt_label":             reading.anomaly_label,
        "pattern_count":        int(pattern_count),
    }

    with _lock:
        _history.append(payload)
        if len(_history) > 600:
            _history.pop(0)
        _latest.update(payload)

        # notification alert
        if result.is_anomaly:
            alert = {
                "id":        int(result.step),
                "step":      int(result.step),
                "time":      time.strftime("%H:%M:%S"),
                "reason":    str(result.reason),
                "label":     (reading.anomaly_label or "ANOMALY").replace("_", " ").upper(),
                "score":     float(result.anomaly_score),
                "severity":  "CRITICAL" if result.anomaly_score > 0.6 else "WARNING",
                "spindle":   float(reading.spindle_current),
                "servo":     float(reading.servo_torque),
                "coolant":   float(reading.coolant_flow),
                "acoustic":  float(reading.acoustic_emission),
                "thermal":   float(reading.thermal_gradient),
            }
            _alerts.append(alert)
            if len(_alerts) > 200:
                _alerts.pop(0)


# ── HTTP handler ──────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent


class Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass

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
        if self.path in ("/", "/index.html"):
            self._file(PROJECT_DIR / "index.html", "text/html")
        elif self.path == "/data":
            with _lock:
                data = list(_history)
            self._json(data)
        elif self.path == "/latest":
            with _lock:
                data = dict(_latest)
            self._json(data)
        elif self.path == "/alerts":
            with _lock:
                data = list(_alerts)
            self._json(data)
        else:
            self.send_response(404)
            self.end_headers()


# ── detection loop ────────────────────────────────────────────────────
def detection_loop():
    sensor    = CNCMachineSimulator(seed=42, training_mode=True)  # start in training mode
    extractor = CNCFeatureExtractor()
    engine    = QdrantEdgeEngine(fresh=True)
    detector  = AnomalyDetector(engine)
    tracker   = SpikeTracker(maxlen=600)

    print(f"[engine] Qdrant Edge shard: {config.QDRANT_SHARD_PATH}")
    print(f"[engine] {config.VECTOR_SIZE}D CNC vectors | training={config.TRAINING_STEPS} steps | threshold={config.ANOMALY_THRESHOLD}")
    print(f"[engine] Phase 1: TRAINING — collecting pure normal baseline ({config.TRAINING_STEPS} steps)...")

    def _steps():
        if config.TOTAL_STEPS == 0:
            while True:
                yield
        else:
            yield from range(config.TOTAL_STEPS)

    _detection_started = False

    try:
        for _ in _steps():
            reading = sensor.read()

            # switch from training → detection after TRAINING_STEPS
            if not _detection_started and reading.step > config.TRAINING_STEPS:
                sensor.set_training_mode(False)
                _detection_started = True
                print(f"[engine] Phase 2: DETECTION — faults now enabled. Qdrant Edge has {engine.pattern_count} normal patterns.")

            vector  = extractor.ingest(reading)
            if vector is None:
                time.sleep(config.SLEEP_INTERVAL)
                continue

            result = detector.process(vector, reading)
            tracker.record(result)
            _store(reading, result, engine.pattern_count)

            if result.is_anomaly:
                sev = " CRITICAL" if result.anomaly_score > 0.5 else "  WARNING"
                print(f"  {sev}  step={result.step:4d}  score={result.anomaly_score:.4f}  {result.reason}")

            time.sleep(config.SLEEP_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        engine.flush()
        engine.close()

    print(f"\n[done] steps={detector._step} anomalies={tracker.total_anomalies} precision={tracker.precision:.0%}")


# ── main ──────────────────────────────────────────────────────────────
def main():
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    server = HTTPServer(("localhost", 8766), Handler)
    print("⚡ Qdrant Edge — CNC Anomaly Detection")
    print("   Dashboard : http://localhost:8766")
    print()
    webbrowser.open("http://localhost:8766")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
