# sensors/multi_sensor.py
# Simulates real industrial sensors: temperature, vibration, pressure
# Injects realistic anomalies (spikes, drops, bursts) at known steps

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class SensorReading:
    step: int
    temperature: float      # °C
    vibration: float        # g-force RMS
    pressure: float         # bar
    ground_truth_anomaly: bool = False
    anomaly_label: Optional[str] = None


# Anomaly injection schedule: step → (type, magnitude)
_ANOMALY_SCHEDULE = {
    60:  ("temp_spike",      +22.0),
    130: ("vib_burst",       +7.5),
    210: ("pressure_drop",   -0.28),
    290: ("multi_spike",     None),   # all sensors spike together
    370: ("vib_oscillation", None),   # sustained vibration for 12 steps
    440: ("temp_spike",      +28.0),
    510: ("vib_burst",       +8.2),
    560: ("pressure_drop",   -0.32),
}
_OSCILLATION_RANGE = range(370, 382)


class MultiSensorSimulator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._step = 0
        self._t = 0.0

    def read(self) -> SensorReading:
        self._step += 1
        self._t += 0.1

        # --- baseline signals ---
        # Temperature: 45°C mean, slow machine-cycle sine + thermal drift + noise
        temp = (
            45.0
            + 3.0 * math.sin(2 * math.pi * self._t / 60.0)
            + 0.004 * self._t
            + random.gauss(0, 0.3)
        )

        # Vibration: 1.5g RMS, bearing harmonics + noise
        vib = max(
            0.0,
            1.5
            + 0.2 * math.sin(2 * math.pi * self._t / 2.0)
            + 0.05 * math.sin(2 * math.pi * self._t * 3 / 2.0)
            + random.gauss(0, 0.04)
        )

        # Pressure: 1.2 bar, pump cycle + noise
        pres = (
            1.2
            + 0.04 * math.sin(2 * math.pi * self._t / 30.0)
            + random.gauss(0, 0.008)
        )

        is_anomaly = False
        label = None

        # --- inject scheduled anomalies ---
        if self._step in _ANOMALY_SCHEDULE:
            atype, mag = _ANOMALY_SCHEDULE[self._step]
            is_anomaly = True
            label = atype

            if atype == "temp_spike":
                temp += mag
            elif atype == "vib_burst":
                vib += mag + random.gauss(0, 0.4)
            elif atype == "pressure_drop":
                pres += mag
            elif atype == "multi_spike":
                temp += 18.0
                vib += 6.0
                pres += 0.18

        if self._step in _OSCILLATION_RANGE:
            vib += 4.5 * abs(math.sin(self._t * 6))
            is_anomaly = True
            label = "vib_oscillation"

        return SensorReading(
            step=self._step,
            temperature=round(temp, 3),
            vibration=round(vib, 4),
            pressure=round(pres, 4),
            ground_truth_anomaly=is_anomaly,
            anomaly_label=label,
        )
