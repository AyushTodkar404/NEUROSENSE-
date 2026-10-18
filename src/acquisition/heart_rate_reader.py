import random
import numpy as np


class HeartRateReader:
    def read_heart_rate(self, state: str = "Neutral") -> int:
        """
        Simulate heart-rate sensor reading correlated with physiological state.
        - Focused: Steady, calm cardiac rate (~68 - 76 BPM)
        - Neutral: Baseline resting rate (~74 - 82 BPM)
        - Distracted: Slightly elevated or variable rate (~82 - 94 BPM)
        """
        if state == "Focused":
            bpm = int(np.clip(np.random.normal(72, 3), 64, 82))
        elif state == "Neutral":
            bpm = int(np.clip(np.random.normal(78, 4), 68, 88))
        elif state == "Distracted":
            bpm = int(np.clip(np.random.normal(87, 5), 75, 105))
        else:
            bpm = random.randint(68, 88)
        return int(bpm)


if __name__ == "__main__":
    sensor = HeartRateReader()
    for s in ["Focused", "Neutral", "Distracted"]:
        print(f"[{s}] Heart Rate:", sensor.read_heart_rate(s), "BPM")