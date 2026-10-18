def calculate_fusion_score(eeg_score, heart_rate, motion_level):
    """
    Experimental multi-sensor fusion score.

    Fuses cortical EEG rhythm (60%), cardiovascular stability (20%),
    and behavioral motion/fidgeting (20%).
    """

    eeg_component = eeg_score * 0.60

    if 60 <= heart_rate <= 100:
        heart_component = 20.0
    else:
        heart_component = 10.0

    motion_component = max(0.0, (1.0 - motion_level) * 20.0)
    excess_motion_penalty = max(0.0, (motion_level - 0.35) * 35.0)

    score = eeg_component + heart_component + motion_component - excess_motion_penalty

    return round(max(0.0, min(score, 100.0)), 1)


def fusion_state(score):
    if score >= 70:
        return "Focused"
    elif score >= 40:
        return "Moderate"
    else:
        return "Distracted"


if __name__ == "__main__":
    score = calculate_fusion_score(
        eeg_score=65,
        heart_rate=78,
        motion_level=0.25
    )

    print("Fusion Score:", score)
    print("State:", fusion_state(score))
