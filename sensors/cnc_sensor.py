# sensors/cnc_sensor.py
# Simulates a real CNC machine with 6 industrial sensors.
# Faults happen RANDOMLY — not at fixed steps. Just like a real machine.

import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class CNCReading:
    step:                  int
    spindle_current:       float
    servo_torque:          float
    coolant_flow:          float
    acoustic_emission:     float
    feed_rate_deviation:   float
    thermal_gradient:      float
    ground_truth_anomaly:  bool           = False
    anomaly_label:         Optional[str]  = None


# fault types and how they affect each sensor
_FAULTS = {
    "tool_wear":       dict(spindle=+3.5, servo=+5.0, acoustic=+40.0, thermal=+3.0, coolant=0,    feed=0,   duration=20),
    "tool_fracture":   dict(spindle=+9.0, servo=+22.0,acoustic=+90.0, thermal=+6.0, coolant=0,    feed=+4.0,duration=1),
    "coolant_block":   dict(spindle=0,    servo=0,    acoustic=0,      thermal=+5.0, coolant=-5.5, feed=0,   duration=15),
    "bearing_fault":   dict(spindle=0,    servo=+7.0, acoustic=+30.0,  thermal=+3.0, coolant=0,    feed=0,   duration=25),
    "thermal_runaway": dict(spindle=+2.0, servo=0,    acoustic=0,      thermal=+10.0,coolant=0,    feed=0,   duration=20),
    "feed_fault":      dict(spindle=0,    servo=+8.0, acoustic=0,      thermal=0,    coolant=0,    feed=+4.5,duration=3),
    "acoustic_crack":  dict(spindle=0,    servo=0,    acoustic=+70.0,  thermal=0,    coolant=0,    feed=0,   duration=1),
    "multi_fault":     dict(spindle=+6.0, servo=+15.0,acoustic=+50.0,  thermal=+7.0, coolant=-3.5, feed=0,   duration=2),
}

# probability of a NEW fault starting at any given step (after warmup)
_FAULT_PROB = 0.0008  # ~0.08% per step = roughly 1 fault every 1250 steps (~2 min)


class CNCMachineSimulator:

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._step        = 0
        self._t           = 0.0
        self._active_fault: Optional[str] = None
        self._fault_step  = 0   # how many steps into current fault

    def _maybe_start_fault(self):
        """Randomly decide if a new fault starts this step."""
        if self._active_fault:
            return   # already in a fault
        if random.random() < _FAULT_PROB:
            self._active_fault = random.choice(list(_FAULTS.keys()))
            self._fault_step   = 0

    def read(self) -> CNCReading:
        self._step += 1
        self._t    += 0.1

        # ── baseline healthy machine ──────────────────────────────────
        spindle = (
            4.5
            + 0.3  * math.sin(2 * math.pi * self._t / 1.5)
            + 0.1  * math.sin(2 * math.pi * self._t / 0.3)
            + random.gauss(0, 0.06)
        )
        servo = (
            12.0
            + 1.2  * math.sin(2 * math.pi * self._t / 3.0)
            + 0.3  * math.sin(2 * math.pi * self._t / 0.8)
            + random.gauss(0, 0.15)
        )
        coolant = (
            8.5
            + 0.2  * math.sin(2 * math.pi * self._t / 20.0)
            + random.gauss(0, 0.05)
        )
        acoustic = (
            47.0
            + 3.0  * math.sin(2 * math.pi * self._t / 0.5)
            + random.gauss(0, 1.2)
        )
        feed_dev = abs(
            0.2
            + 0.1  * math.sin(2 * math.pi * self._t / 5.0)
            + random.gauss(0, 0.04)
        )
        thermal = (
            2.2
            + 0.3  * math.sin(2 * math.pi * self._t / 40.0)
            + random.gauss(0, 0.08)
        )

        is_anomaly = False
        label      = None

        # ── random fault injection ────────────────────────────────────
        self._maybe_start_fault()

        if self._active_fault:
            f = _FAULTS[self._active_fault]
            self._fault_step += 1
            progress = min(1.0, self._fault_step / max(1, f["duration"]))

            spindle  += f["spindle"]  * progress + random.gauss(0, abs(f["spindle"])  * 0.05 + 0.01)
            servo    += f["servo"]    * progress + random.gauss(0, abs(f["servo"])    * 0.05 + 0.01)
            acoustic += f["acoustic"] * progress + random.gauss(0, abs(f["acoustic"]) * 0.05 + 0.01)
            thermal  += f["thermal"]  * progress + random.gauss(0, abs(f["thermal"])  * 0.05 + 0.01)
            coolant  += f["coolant"]  * progress
            feed_dev += f["feed"]     * progress

            is_anomaly = True
            label      = self._active_fault

            # end fault after its duration
            if self._fault_step >= f["duration"]:
                self._active_fault = None
                self._fault_step   = 0

        # clamp physical limits
        coolant  = max(0.1, coolant)
        feed_dev = max(0.0, feed_dev)
        thermal  = max(0.5, thermal)
        spindle  = max(0.5, spindle)

        return CNCReading(
            step                 = self._step,
            spindle_current      = round(spindle,  3),
            servo_torque         = round(servo,    3),
            coolant_flow         = round(coolant,  3),
            acoustic_emission    = round(acoustic, 2),
            feed_rate_deviation  = round(feed_dev, 4),
            thermal_gradient     = round(thermal,  3),
            ground_truth_anomaly = is_anomaly,
            anomaly_label        = label,
        )
