# intelligence/spike_tracker.py
# Keeps rolling history of AnomalyResult for dashboard display

from collections import deque
from intelligence.anomaly_engine import AnomalyResult


class SpikeTracker:
    def __init__(self, maxlen: int = 500):
        self.history: deque[AnomalyResult] = deque(maxlen=maxlen)
        self.anomaly_events: list[AnomalyResult] = []
        self.true_positives  = 0
        self.false_positives = 0
        self.total_anomalies = 0

    def record(self, result: AnomalyResult):
        self.history.append(result)
        if result.is_anomaly:
            self.total_anomalies += 1
            self.anomaly_events.append(result)
            if result.ground_truth:
                self.true_positives += 1
            else:
                self.false_positives += 1

    @property
    def scores(self) -> list[float]:
        return [r.anomaly_score for r in self.history]

    @property
    def precision(self) -> float:
        return self.true_positives / self.total_anomalies if self.total_anomalies else 0.0
