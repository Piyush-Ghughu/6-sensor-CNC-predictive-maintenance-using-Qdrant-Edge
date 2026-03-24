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
    # CNC fields (populated when using CNC sensor)
    spindle_current: float     = 0.0
    servo_torque: float        = 0.0
    coolant_flow: float        = 0.0
    acoustic_emission: float   = 0.0
    feed_rate_deviation: float = 0.0
    thermal_gradient: float    = 0.0
    ground_truth: bool = False
    gt_label: Optional[str] = None


class AnomalyDetector:
    def __init__(self, engine):
        self._engine   = engine
        self._step     = 0
        self._history  = deque(maxlen=config.BASELINE_WINDOW)

  
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

    def process(self, vector: np.ndarray, reading) -> AnomalyResult:
        self._step += 1

        similarity    = self._engine.search(vector)
        anomaly_score = 1.0 - similarity

        spike, spike_conf = self._spike_check(anomaly_score)
        self._history.append(anomaly_score)

       
        # extract CNC fields if available
        is_cnc = hasattr(reading, 'spindle_current')
        extras = dict(
            spindle_current     = float(reading.spindle_current)     if is_cnc else 0.0,
            servo_torque        = float(reading.servo_torque)        if is_cnc else 0.0,
            coolant_flow        = float(reading.coolant_flow)        if is_cnc else 0.0,
            acoustic_emission   = float(reading.acoustic_emission)   if is_cnc else 0.0,
            feed_rate_deviation = float(reading.feed_rate_deviation) if is_cnc else 0.0,
            thermal_gradient    = float(reading.thermal_gradient)    if is_cnc else 0.0,
            temperature         = float(reading.temperature)         if hasattr(reading,'temperature') else 0.0,
            vibration           = float(reading.vibration)           if hasattr(reading,'vibration')   else 0.0,
            pressure            = float(reading.pressure)            if hasattr(reading,'pressure')    else 0.0,
        )

        if self._step <= config.WARMUP_STEPS:
            self._engine.store(vector)
            return AnomalyResult(
                step=self._step, similarity=similarity,
                anomaly_score=anomaly_score, is_anomaly=False,
                spike=False, confidence=0.0,
                reason=f"WARMUP {self._step}/{config.WARMUP_STEPS}",
                ground_truth=reading.ground_truth_anomaly,
                gt_label=reading.anomaly_label,
                **extras,
            )

        # --- DETECTION ---
        low_sim    = similarity > 0 and similarity < config.ANOMALY_THRESHOLD
        is_anomaly = low_sim or spike

        if is_anomaly:
            faults = []
            if is_cnc:
                sc = extras["spindle_current"]
                st = extras["servo_torque"]
                cf = extras["coolant_flow"]
                ae = extras["acoustic_emission"]
                fd = extras["feed_rate_deviation"]
                tg = extras["thermal_gradient"]

                # use percentage deviation from known normal baseline
                if sc > 5.5:    faults.append(f"Spindle Current {sc:.2f}A — motor load high (normal ~4.5A)")
                if st > 16.0:   faults.append(f"Servo Torque {st:.2f}Nm — axis stress high (normal ~12Nm)")
                if cf < 6.0:    faults.append(f"Coolant Flow {cf:.2f} L/min — flow dropping (normal ~8.5)")
                if ae > 60.0:   faults.append(f"Acoustic Emission {ae:.1f}kHz — abnormal noise (normal ~47kHz)")
                if fd > 1.0:    faults.append(f"Feed Rate Deviation {fd:.3f}% — controller off (normal ~0.2%)")
                if tg > 4.0:    faults.append(f"Thermal Gradient {tg:.2f}C — heat building (normal ~2.2C)")

            if faults:
                reason = " | ".join(faults)
            else:
                # Qdrant Edge flagged it but no single sensor crossed threshold
                # report the most deviated sensor
                if is_cnc:
                    deviations = {
                        f"Spindle Current {extras['spindle_current']:.2f}A": abs(extras["spindle_current"] - 4.5) / 4.5,
                        f"Servo Torque {extras['servo_torque']:.2f}Nm":      abs(extras["servo_torque"] - 12.0) / 12.0,
                        f"Coolant Flow {extras['coolant_flow']:.2f}L/min":   abs(extras["coolant_flow"] - 8.5) / 8.5,
                        f"Acoustic {extras['acoustic_emission']:.1f}kHz":    abs(extras["acoustic_emission"] - 47.0) / 47.0,
                        f"Thermal {extras['thermal_gradient']:.2f}C":        abs(extras["thermal_gradient"] - 2.2) / 2.2,
                    }
                    worst = max(deviations, key=deviations.get)
                    reason = f"{worst} — unusual reading detected by Qdrant Edge"
                else:
                    reason = f"Unfamiliar pattern (similarity={similarity:.3f})"

            confidence = max(
                (config.ANOMALY_THRESHOLD - similarity) / config.ANOMALY_THRESHOLD if low_sim else 0.0,
                spike_conf,
            )
        else:
            self._engine.store(vector)
            reason = f"NORMAL(sim={similarity:.3f})"
            confidence = 0.0

        return AnomalyResult(
            step=self._step, similarity=similarity,
            anomaly_score=anomaly_score, is_anomaly=is_anomaly,
            spike=spike, confidence=round(confidence, 3),
            reason=reason,
            ground_truth=reading.ground_truth_anomaly,
            gt_label=reading.anomaly_label,
            **extras,
        )
