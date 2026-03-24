#!/usr/bin/env python3
"""
Qdrant Edge Anomaly Detection
==============================
Multi-sensor (temperature / vibration / pressure) real-time anomaly detection
using Qdrant Edge (qdrant-edge-py) — fully embedded, zero server required.

Pipeline:
  MultiSensorSimulator → FeatureExtractor → QdrantEdgeEngine → AnomalyDetector → Dashboard
"""

import time
from rich.console import Console

import config
from sensors.multi_sensor import MultiSensorSimulator
from edge.feature_extractor import FeatureExtractor
from core.qdrant_engine import QdrantEdgeEngine
from intelligence.anomaly_engine import AnomalyDetector
from intelligence.spike_tracker import SpikeTracker
from ui.live_plot import Dashboard

console = Console()


def main():
    console.print("[bold cyan]⚡ Qdrant Edge Anomaly Detection[/bold cyan]")
    console.print(f"[dim]Shard path : {config.QDRANT_SHARD_PATH}[/dim]")
    console.print(f"[dim]Vector size: {config.VECTOR_SIZE}D  │  "
                  f"Warmup: {config.WARMUP_STEPS} steps  │  "
                  f"Threshold: sim < {config.ANOMALY_THRESHOLD}[/dim]")
    console.print()

    sensor    = MultiSensorSimulator(seed=42)
    extractor = FeatureExtractor()
    engine    = QdrantEdgeEngine(fresh=True)
    detector  = AnomalyDetector(engine)
    tracker   = SpikeTracker(maxlen=500)

    try:
        with Dashboard(tracker) as dash:
            for _ in range(config.TOTAL_STEPS):
                reading = sensor.read()
                vector  = extractor.ingest(reading)

                if vector is None:          # window not full yet
                    time.sleep(config.SLEEP_INTERVAL)
                    continue

                result = detector.process(vector, reading)
                tracker.record(result)
                dash.update(result, engine.pattern_count)

                time.sleep(config.SLEEP_INTERVAL)

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")
    finally:
        engine.flush()
        engine.close()

    # ── summary ──────────────────────────────────────────────────────
    console.print()
    console.print("[bold cyan]══ FINAL SUMMARY ══[/bold cyan]")
    console.print(f"  Steps run        : {detector._step}")
    console.print(f"  Patterns stored  : {engine.pattern_count}")
    console.print(f"  Anomalies flagged: {tracker.total_anomalies}")
    console.print(f"  True positives   : {tracker.true_positives}")
    console.print(f"  False positives  : {tracker.false_positives}")
    console.print(f"  Precision        : {tracker.precision:.0%}")


if __name__ == "__main__":
    main()
