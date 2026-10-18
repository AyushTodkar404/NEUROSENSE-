import json
import time
from pathlib import Path

import numpy as np

from acquisition.eeg_reader import EEGReader
from acquisition.heart_rate_reader import HeartRateReader
from acquisition.motion_reader import MotionReader


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "sample" / "replay_session.jsonl"


def generate_replay(seconds=60, sampling_rate=256):
    """
    Generate an authentic multi-phase cognitive session for replay:
    - 0s to 20s: Focused State (Elevated Beta, calm HR, low motion)
    - 20s to 40s: Neutral State (Prominent Alpha, resting HR, moderate motion)
    - 40s to 60s: Distracted State (Alpha surge, elevated HR, restless motion)
    """
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    eeg_reader = EEGReader(sampling_rate=sampling_rate)
    hr_reader = HeartRateReader()
    motion_reader = MotionReader()

    with OUTPUT.open("w") as f:
        start = time.time()

        for second in range(seconds):
            if second < 20:
                state = "Focused"
            elif second < 40:
                state = "Neutral"
            else:
                state = "Distracted"

            _, eeg_chunk = eeg_reader.generate_eeg(duration=1, state_override=state)
            hr = hr_reader.read_heart_rate(state)
            motion = motion_reader.read_motion(state)

            packet = {
                "timestamp": start + second,
                "target_state": state,
                "heart_rate": hr,
                "motion_level": motion,
                "eeg": eeg_chunk.tolist()
            }

            f.write(json.dumps(packet) + "\n")

    print(f"Realistic replay file created ({seconds} seconds):")
    print(OUTPUT)


if __name__ == "__main__":
    generate_replay()

