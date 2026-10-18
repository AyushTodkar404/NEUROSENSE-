import time
import numpy as np


class EEGReader:
    """
    Biophysically-grounded EEG signal generator.
    Simulates authentic human cortical rhythms:
    - 1/f pink noise electrophysiological background
    - Alpha rhythm with waxing/waning spindle envelope
    - Beta activity during cognitive focus
    - Frontal theta oscillations
    - Periodic ocular micro-artifacts (blinks)
    - Time-varying or scenario-driven cognitive states:
      'Focused', 'Neutral', 'Distracted', or 'Auto'
    """

    def __init__(self, sampling_rate=256, initial_state="Auto"):
        self.sampling_rate = sampling_rate
        self.current_state = initial_state
        self._start_time = time.time()
        self._sample_counter = 0

    def set_state(self, state: str):
        """Set cognitive state: 'Focused', 'Neutral', 'Distracted', or 'Auto'."""
        self.current_state = state

    def get_effective_state(self) -> str:
        """Resolve 'Auto' into time-varying cyclic cognitive states."""
        if self.current_state != "Auto":
            return self.current_state
        # Auto cycle: 16s Focused -> 14s Neutral -> 14s Distracted -> repeat (44s cycle)
        elapsed = (time.time() - self._start_time) % 44.0
        if elapsed < 16.0:
            return "Focused"
        elif elapsed < 30.0:
            return "Neutral"
        else:
            return "Distracted"

    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """Generate 1/f neural background noise."""
        white = np.random.normal(0, 1, n_samples)
        fft_white = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / self.sampling_rate)
        freqs[0] = 1.0  # Avoid division by zero
        pink_fft = fft_white / np.sqrt(freqs)
        pink = np.fft.irfft(pink_fft, n=n_samples)
        pink = 10.0 * (pink - np.mean(pink)) / (np.std(pink) + 1e-6)
        return pink

    def generate_eeg(self, duration=2, state_override=None):
        """
        Generate physiological EEG signal calibrated to the NeuroSense
        feature extractor and trained XGBoost classification space.
        """
        n_samples = int(self.sampling_rate * duration)
        start_idx = self._sample_counter
        self._sample_counter += n_samples

        t = np.linspace(
            start_idx / self.sampling_rate,
            (start_idx + n_samples) / self.sampling_rate,
            n_samples,
            endpoint=False
        )

        state = state_override or self.get_effective_state()

        # 1. 1/f Neural background noise
        pink_noise = self._generate_pink_noise(n_samples)

        # 2. State-dependent band dynamics
        if state == "Focused":
            # High Beta (18-24 Hz), Suppressed Alpha (8-12 Hz), Low Theta
            theta = 8.0 * np.sin(2 * np.pi * 6.0 * t)
            # Alpha spindle amplitude modulation
            spindle_env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t)
            alpha = 7.5 * spindle_env * np.sin(2 * np.pi * 10.0 * t)
            beta = (
                18.0 * np.sin(2 * np.pi * 20.0 * t)
                + 12.0 * np.sin(2 * np.pi * 23.5 * t)
                + np.random.normal(0, 2.0, n_samples)
            )
        elif state == "Neutral":
            # Balanced Alpha and Beta
            theta = 11.5 * np.sin(2 * np.pi * 6.0 * t)
            spindle_env = 0.8 + 0.5 * np.sin(2 * np.pi * 0.35 * t)
            alpha = 18.0 * spindle_env * np.sin(2 * np.pi * 10.2 * t)
            beta = 14.0 * np.sin(2 * np.pi * 20.0 * t) + np.random.normal(0, 2.5, n_samples)
        else:  # Distracted / Relaxed / Drowsy
            # Dominant Alpha bursts, elevated Theta, low Beta
            theta = 14.0 * np.sin(2 * np.pi * 5.5 * t)
            spindle_env = 1.0 + 0.6 * np.sin(2 * np.pi * 0.4 * t)
            alpha = 24.0 * spindle_env * np.sin(2 * np.pi * 9.8 * t)
            beta = 8.0 * np.sin(2 * np.pi * 19.0 * t) + np.random.normal(0, 2.0, n_samples)

        # 3. Micro ocular artifact (simulating natural blinks every ~7 seconds)
        blink_period = 7.0
        time_in_period = t % blink_period
        blink_mask = (time_in_period > 3.0) & (time_in_period < 3.35)
        blink_pulse = np.zeros(n_samples)
        if np.any(blink_mask):
            tau = (time_in_period[blink_mask] - 3.175) / 0.08
            blink_pulse[blink_mask] = 35.0 * np.exp(-0.5 * (tau ** 2))

        # Composite signal
        eeg_signal = pink_noise + theta + alpha + beta + blink_pulse

        # Return relative time axis starting from 0 for the window
        rel_t = np.linspace(0, duration, n_samples, endpoint=False)
        return rel_t, eeg_signal


if __name__ == "__main__":
    reader = EEGReader()

    for s in ["Focused", "Neutral", "Distracted"]:
        time_axis, eeg = reader.generate_eeg(state_override=s)
        print(f"[{s}] samples: {len(eeg)}, std: {eeg.std():.2f}")