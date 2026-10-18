import random
import numpy as np


class MotionReader:
    """
    Simulated motion sensor correlated with cognitive engagement:
    - Focused: Very quiet / still (0.05 - 0.20)
    - Neutral: Normal subtle postural shifts (0.22 - 0.45)
    - Distracted: Fidgeting, head-turning, movement (0.50 - 0.85)
    """

    def read_motion(self, state: str = "Neutral") -> float:
        if state == "Focused":
            val = float(np.clip(np.random.uniform(0.06, 0.20), 0.02, 0.25))
        elif state == "Neutral":
            val = float(np.clip(np.random.uniform(0.25, 0.45), 0.15, 0.55))
        elif state == "Distracted":
            val = float(np.clip(np.random.uniform(0.55, 0.85), 0.40, 1.0))
        else:
            val = random.uniform(0.0, 1.0)
        return round(val, 2)


if __name__ == "__main__":
    sensor = MotionReader()
    for s in ["Focused", "Neutral", "Distracted"]:
        print(f"[{s}] Motion Level:", sensor.read_motion(s))

