# edge/cnc_extractor.py
# Converts rolling CNC sensor window → 24-dim normalized feature vector
# 6 sensors × 4 features (mean, std, max, rate-of-change) = 24 dims

import numpy as np
from collections import deque
from sensors.cnc_sensor import CNCReading

WINDOW = 10


class CNCFeatureExtractor:
    """
    Online feature extraction for 6 CNC sensors.
    Uses Welford's algorithm for running normalization — no batch needed.
    Produces a 24-dim L2-normalized vector for Qdrant Edge cosine search.
    """

    SENSORS = [
        ("spindle",   4.5,   0.3),
        ("servo",     12.0,  1.5),
        ("coolant",   8.5,   0.3),
        ("acoustic",  47.0,  5.0),
        ("feed_dev",  0.2,   0.1),
        ("thermal",   2.2,   0.4),
    ]

    def __init__(self):
        self._bufs  = {s: deque(maxlen=WINDOW) for s, _, _ in self.SENSORS}
        self._n     = 0
        self._means = {s: m  for s, m, _ in self.SENSORS}
        self._m2s   = {s: v**2 for s, _, v in self.SENSORS}

    def _welford(self, key: str, val: float):
        self._n += 1
        d = val - self._means[key]
        self._means[key] += d / self._n
        self._m2s[key]   += d * (val - self._means[key])

    def _std(self, key: str) -> float:
        if self._n < 2:
            return 1.0
        return max(1e-6, (self._m2s[key] / (self._n - 1)) ** 0.5)

    def _features(self, key: str) -> list:
        arr = np.array(self._bufs[key], dtype=np.float32)
        if len(arr) < 2:
            return [0.0, 0.0, 0.0, 0.0]
        std  = self._std(key)
        mean = self._means[key]
        return [
            (float(np.mean(arr)) - mean) / std,
            float(np.std(arr)) / std,
            (float(np.max(arr)) - mean) / std,
            float(arr[-1] - arr[-2]) / std,
        ]

    def ingest(self, r: CNCReading) -> np.ndarray | None:
        vals = {
            "spindle":  r.spindle_current,
            "servo":    r.servo_torque,
            "coolant":  r.coolant_flow,
            "acoustic": r.acoustic_emission,
            "feed_dev": r.feed_rate_deviation,
            "thermal":  r.thermal_gradient,
        }
        for key, val in vals.items():
            self._bufs[key].append(val)
            self._welford(key, val)

        if len(self._bufs["spindle"]) < WINDOW:
            return None

        raw = []
        for key in ["spindle", "servo", "coolant", "acoustic", "feed_dev", "thermal"]:
            raw.extend(self._features(key))

        vec  = np.array(raw, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
