# sensors/vibration_sensor.py

import numpy as np

def read_sensor(step):
    t = np.linspace(0, 1, 100)

    if step % 25 == 0:
        return np.sin(2 * np.pi * 30 * t) + 0.6 * np.random.randn(100)
    else:
        return np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(100)