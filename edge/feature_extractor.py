# edge/feature_extractor.py
# Converts a rolling window of sensor readings → 12-dim normalized feature vector
# 3 sensors × 4 features (mean, std, max, rate-of-change) = 12 dims

import numpy as np
from collections import deque
from sensors.multi_sensor import SensorReading

WINDOW = 10   # samples in rolling window


class FeatureExtractor:
    def __init__(self):
        self._bufs = {
            "temp": deque(maxlen=WINDOW),
            "vib":  deque(maxlen=WINDOW),
            "pres": deque(maxlen=WINDOW),
        }
        # Welford running stats for online normalization
        self._n = 0
        self._means = {"temp": 45.0, "vib": 1.5, "pres": 1.2}
        self._m2s   = {"temp": 25.0, "vib": 0.25, "pres": 0.0025}

    # ------------------------------------------------------------------
    def _welford_update(self, key: str, val: float):
        self._n += 1
        delta = val - self._means[key]
        self._means[key] += delta / self._n
        self._m2s[key]   += delta * (val - self._means[key])

    def _std(self, key: str) -> float:
        if self._n < 2:
            return 1.0
        return max(1e-6, (self._m2s[key] / (self._n - 1)) ** 0.5)

    # ------------------------------------------------------------------
    def _features(self, key: str) -> list:
        arr = np.array(self._bufs[key], dtype=np.float32)
        if len(arr) < 2:
            return [0.0, 0.0, 0.0, 0.0]

        std  = self._std(key)
        mean = self._means[key]

        f_mean = (float(np.mean(arr)) - mean) / std
        f_std  = float(np.std(arr))   / std
        f_max  = (float(np.max(arr))  - mean) / std
        f_roc  = float(arr[-1] - arr[-2]) / std   # rate-of-change

        return [f_mean, f_std, f_max, f_roc]

    # ------------------------------------------------------------------
    def ingest(self, r: SensorReading) -> np.ndarray | None:
        """Returns 12-dim unit vector, or None if window not yet full."""
        self._bufs["temp"].append(r.temperature)
        self._bufs["vib"].append(r.vibration)
        self._bufs["pres"].append(r.pressure)

        self._welford_update("temp", r.temperature)
        self._welford_update("vib",  r.vibration)
        self._welford_update("pres", r.pressure)

        if len(self._bufs["temp"]) < WINDOW:
            return None

        raw = (
            self._features("temp")
            + self._features("vib")
            + self._features("pres")
        )
        vec = np.array(raw, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
