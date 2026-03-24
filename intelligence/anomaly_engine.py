# intelligence/anomaly_engine.py
# Two-layer anomaly detection:
#   1. Qdrant Edge similarity  — is this vector unlike anything seen before?
#   2. Statistical spike check — is the anomaly score a sudden spike vs history?

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
import config


@dataclass
class AnomalyResult:
    step: int
    similarity: float       # 0→1, higher = more similar to normal patterns
    anomaly_score: float    # 0→1, higher = more anomalous  (= 1 - similarity)
    is_anomaly: bool
    spike: bool             # statistical spike detected
    confidence: float       # 0→1
    reason: str
    # raw readings for display
    temperature: float = 0.0
    vibration: float   = 0.0
    pressure: float    = 0.0
    ground_truth: bool = False
    gt_label: Optional[str] = None


class AnomalyDetector:
    def __init__(self, engine):
        self._engine   = engine
        self._step     = 0
        self._history  = deque(maxlen=config.BASELINE_WINDOW)

    # ------------------------------------------------------------------
    def _spike_check(self, score: float) -> tuple[bool, float]:
        """Z-score spike detection on rolling anomaly score history."""
        if len(self._history) < 10:
            return False, 0.0
        arr  = np.array(self._history)
        mean = arr.mean()
        std  = arr.std()
        if std < 1e-6:
            return False, 0.0
        z = (score - mean) / std
        return z > config.SPIKE_Z_SCORE, min(1.0, z / (config.SPIKE_Z_SCORE * 2))

    # ------------------------------------------------------------------
    def process(self, vector: np.ndarray, reading) -> AnomalyResult:
        self._step += 1

        similarity    = self._engine.search(vector)
        anomaly_score = 1.0 - similarity

        spike, spike_conf = self._spike_check(anomaly_score)
        self._history.append(anomaly_score)

        # --- WARMUP: just store every vector as a "normal" baseline ---
        if self._step <= config.WARMUP_STEPS:
            self._engine.store(vector)
            return AnomalyResult(
                step=self._step, similarity=similarity,
                anomaly_score=anomaly_score, is_anomaly=False,
                spike=False, confidence=0.0,
                reason=f"WARMUP {self._step}/{config.WARMUP_STEPS}",
                temperature=reading.temperature,
                vibration=reading.vibration,
                pressure=reading.pressure,
                ground_truth=reading.ground_truth_anomaly,
                gt_label=reading.anomaly_label,
            )

        # --- DETECTION ---
        low_sim = similarity > 0 and similarity < config.ANOMALY_THRESHOLD
        is_anomaly = low_sim or spike

        if is_anomaly:
            parts = []
            if low_sim:
                parts.append(f"LOW_SIM({similarity:.3f})")
            if spike:
                parts.append(f"SPIKE(z>{config.SPIKE_Z_SCORE}σ)")
            reason = " + ".join(parts)
            confidence = max(
                (config.ANOMALY_THRESHOLD - similarity) / config.ANOMALY_THRESHOLD if low_sim else 0.0,
                spike_conf,
            )
        else:
            # Normal → store as a learned pattern (self-supervised)
            self._engine.store(vector)
            reason = f"NORMAL(sim={similarity:.3f})"
            confidence = 0.0

        return AnomalyResult(
            step=self._step, similarity=similarity,
            anomaly_score=anomaly_score, is_anomaly=is_anomaly,
            spike=spike, confidence=round(confidence, 3),
            reason=reason,
            temperature=reading.temperature,
            vibration=reading.vibration,
            pressure=reading.pressure,
            ground_truth=reading.ground_truth_anomaly,
            gt_label=reading.anomaly_label,
        )
